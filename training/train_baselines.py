"""Benchmark or train CNN baselines. Test audio is never loaded here."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import time

import librosa
import numpy as np
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, f1_score
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import yaml

from common.baseline_models import make_model
from prepare_data import environment

ROOT = Path(__file__).resolve().parents[1]


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_audio(rows, config):
    root = ROOT / config['dataset_root'] / 'audio'
    cache = ROOT / 'data' / 'cache' / f"esc50_{config['sample_rate']}hz"
    cache.mkdir(parents=True, exist_ok=True)
    samples = int(config['sample_rate'] * config['duration_seconds'])
    result = []
    for row in rows:
        source = root / row['filename']
        # Cache key includes content hash and preprocessing parameters.
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        row['audio_sha256'] = digest
        cached = cache / f'{digest}_{samples}.npy'
        if cached.exists():
            wave = np.load(cached, allow_pickle=False)
        else:
            wave, sr = sf.read(source, dtype='float32', always_2d=True)
            wave = librosa.resample(wave.mean(axis=1), orig_sr=sr, target_sr=config['sample_rate'])
            wave = np.pad(wave[:samples], (0, max(0, samples - len(wave))))
            np.save(cached, wave, allow_pickle=False)
        if wave.shape != (samples,) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid audio: {source}')
        result.append(torch.from_numpy(wave))
    return TensorDataset(torch.stack(result), torch.tensor([r['target'] for r in rows]))


def augment(wave, config):
    b, length = wave.shape
    max_shift = int(config['time_shift_seconds'] * config['sample_rate'])
    shifts = torch.randint(-max_shift, max_shift + 1, (b, 1), device=wave.device)
    indices = (torch.arange(length, device=wave.device)[None] - shifts) % length
    wave = wave.gather(1, indices)
    snr = torch.empty((b, 1), device=wave.device).uniform_(*config['noise_snr_db'])
    noise = torch.randn_like(wave)
    rms = wave.square().mean(1, keepdim=True).sqrt()
    noise_rms = noise.square().mean(1, keepdim=True).sqrt().clamp_min(1e-10)
    return wave + noise * rms / noise_rms * torch.pow(10., -snr / 20)


@torch.inference_mode()
def validate(model, loader, device, amp):
    model.eval()
    actual, predicted = [], []
    for wave, label in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(wave.to(device))
        actual.extend(label.tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
    return dict(accuracy=accuracy_score(actual, predicted),
                macro_f1=f1_score(actual, predicted, labels=list(range(50)), average='macro', zero_division=0))


def run(config, mode, model_name, fold):
    seed_all(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    env = environment()
    device = torch.device(env['device'])
    amp = config['mixed_precision'] and device.type == 'cuda'
    name = f"{mode}_{model_name}_fold{fold}_seed{config['seed']}_{time.strftime('%Y%m%d_%H%M%S')}"
    output = ROOT / 'artifacts' / 'report_esc50' / name
    output.mkdir(parents=True, exist_ok=False)
    effective = dict(config, mode=mode, model=model_name, test_fold=fold)
    (output / 'config.yaml').write_text(yaml.safe_dump(effective, sort_keys=False), encoding='utf-8')
    for args, filename in [(['rev-parse', 'HEAD'], 'git_commit.txt'), (['status', '--short'], 'git_status.txt')]:
        (output / filename).write_text(subprocess.check_output(['git', *args], cwd=ROOT, text=True), encoding='utf-8')
    # Preserve exact training code even when Claude commits or edits elsewhere.
    code_dir = output / 'source'
    code_dir.mkdir()
    for source in [Path(__file__), ROOT / 'training/common/baseline_models.py', ROOT / 'training/prepare_data.py']:
        (code_dir / source.name).write_bytes(source.read_bytes())
    versions = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (output / 'env.txt').write_text('\n'.join(versions) + '\n', encoding='utf-8')
    json_write(output / 'environment.json', env)
    splits = json.loads((ROOT / config['split_manifest']).read_text(encoding='utf-8'))[str(fold)]
    start_load = time.perf_counter()
    train = load_audio(splits['train'], config)
    val = load_audio(splits['validation'], config)
    load_seconds = time.perf_counter() - start_load
    files = sorted(r['filename'] for part in ['train', 'validation'] for r in splits[part])
    json_write(output / 'data_manifest.json', dict(
        train=splits['train'], validation=splits['validation'],
        held_out_test_files=[r['filename'] for r in splits['test']],
        file_list_sha256=hashlib.sha256(('\n'.join(files) + '\n').encode()).hexdigest(),
        test_audio_loaded=False))
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    model = make_model(model_name, config).to(device)
    count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    total_steps = len(loader) * config['epochs']
    warmup = max(1, int(total_steps * config['warmup_fraction']))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    writer = SummaryWriter(str(ROOT / 'runs' / name))
    times, history = [], []
    best, stale, step = -1., 0, 0
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    print(f'{name}: {count:,} parameters; {device}; data load {load_seconds:.1f}s', flush=True)
    try:
        for epoch in range(config['epochs']):
            model.train()
            losses = []
            for wave, label in loader:
                sync(device)
                start_step = time.perf_counter()
                wave, label = wave.to(device), label.to(device)
                wave = augment(wave, config)
                mix = float(np.random.beta(config['mixup_alpha'], config['mixup_alpha']))
                permutation = torch.randperm(len(wave), device=device)
                mixed = mix * wave + (1 - mix) * wave[permutation]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(mixed)
                    loss = mix * nn.functional.cross_entropy(logits, label) + (1 - mix) * nn.functional.cross_entropy(logits, label[permutation])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                sync(device)
                times.append(time.perf_counter() - start_step)
                step += 1
                losses.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)
                if mode == 'benchmark' and step >= config['benchmark_steps']:
                    break
            sync(device)
            val_start = time.perf_counter()
            scores = validate(model, val_loader, device, amp)
            sync(device)
            val_seconds = time.perf_counter() - val_start
            record = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), **scores)
            history.append(record)
            for key, value in scores.items():
                writer.add_scalar(f'validation/{key}', value, epoch + 1)
            print(json.dumps(record), flush=True)
            if mode == 'benchmark':
                steady = times[config['benchmark_warmup_steps']:]
                seconds = float(np.mean(steady))
                metrics = dict(mode=mode, model=model_name, parameters=count, steps=step,
                               seconds_per_step=seconds, validation_seconds=val_seconds,
                               data_loading_seconds=load_seconds,
                               estimated_max_training_seconds=config['epochs'] * (len(loader) * seconds + val_seconds),
                               max_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                               validation=scores, test_evaluated=False)
                json_write(output / 'metrics.json', metrics)
                print(json.dumps(metrics, indent=2), flush=True)
                return metrics
            if scores['macro_f1'] > best:
                best, stale = scores['macro_f1'], 0
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output / 'model.safetensors'))
                json_write(output / 'best_validation.json', record)
            else:
                stale += 1
            json_write(output / 'metrics.json', dict(mode=mode, model=model_name, parameters=count,
                       best_validation_macro_f1=best, history=history,
                       elapsed_seconds=time.perf_counter() - start, test_evaluated=False,
                       complete=stale >= config['patience'] or epoch + 1 == config['epochs']))
            if stale >= config['patience']:
                break
    finally:
        writer.close()
    return dict(output=str(output), best_validation_macro_f1=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'train'], default='benchmark')
    parser.add_argument('--model', choices=['wave_cnn', 'mel_cnn'])
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    models = [args.model] if args.model else config['models']
    folds = [args.fold] if args.fold else ([1] if args.mode == 'benchmark' else config['folds'])
    results = []
    for name in models:
        for fold in folds:
            results.append(run(config, args.mode, name, fold))
    if args.mode == 'train':
        index = ROOT / 'artifacts' / 'report_esc50' / f"training_index_{time.strftime('%Y%m%d_%H%M%S')}.json"
        json_write(index, results)
        print(f'Training index: {index}', flush=True)


if __name__ == '__main__':
    main()
