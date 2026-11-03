"""Memory-bounded AST training and 50-optimizer-step benchmarks on ESC-50.

Streaming waveforms; offline pinned weights; no test audio in this script.
Each phase is benchmarked independently. Benchmark weights are never reused.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import ASTForAudioClassification
import yaml

from common.ast_model import create_ast, loading_metadata
from common.checkpoint import atomic_json, load_checkpoint, restore_rng, save_checkpoint
from prepare_data import environment
from train_baselines import ROOT, augment, json_write, seed_all, sync


def available_ram():
    if not hasattr(ctypes, 'windll'):
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ['total_phys', 'avail_phys', 'total_page',
                'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended']]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_phys


class StreamingAudio(Dataset):
    def __init__(self, rows, config):
        self.rows, self.config = rows, config
        self.samples = int(config['sample_rate'] * config['duration_seconds'])
        self.cache = ROOT / 'data/cache' / f"esc50_{config['sample_rate']}hz"
        self.cache.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = ROOT / config['dataset_root'] / 'audio' / row['filename']
            with source.open('rb') as handle:
                row['audio_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cached = self.cache / f"{row['audio_sha256']}_{self.samples}.npy"
        if cached.exists():
            wave = np.load(cached, allow_pickle=False)
        else:
            wave, sr = sf.read(ROOT / self.config['dataset_root'] / 'audio' / row['filename'],
                               dtype='float32', always_2d=True)
            wave = librosa.resample(wave.mean(1), orig_sr=sr, target_sr=self.config['sample_rate'])
            wave = np.pad(wave[:self.samples], (0, max(0, self.samples - len(wave))))
            np.save(cached, wave, allow_pickle=False)
        return torch.from_numpy(wave), row['target']


def features(wave, extractor, config, training, waveform_augmented=False):
    if training and not waveform_augmented:
        wave = augment(wave, config)
    values = extractor([sample.numpy() for sample in wave], sampling_rate=config['sample_rate'],
                       return_tensors='pt')['input_values']
    if training:
        for sample in values:
            for axis, key in [(0, 'specaugment_time_width'), (1, 'specaugment_frequency_width')]:
                width = int(torch.randint(config[key] + 1, ()).item())
                start = int(torch.randint(sample.shape[axis] - width + 1, ()).item())
                if axis == 0:
                    sample[start:start + width, :] = 0
                else:
                    sample[:, start:start + width] = 0
    return values


def set_phase(model, phase, config):
    for parameter in model.parameters():
        parameter.requires_grad_(phase == 'finetune')
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if phase == 'finetune' and config['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.gradient_checkpointing_disable()


@torch.inference_mode()
def validation(model, extractor, loader, config, device, amp, limit=None):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    actual, predicted, times = [], [], []
    for index, (wave, labels) in enumerate(loader):
        if limit is not None and index >= limit:
            break
        sync(device)
        start = time.perf_counter()
        values = features(wave, extractor, config, False).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(input_values=values).logits
        predicted.extend(logits.argmax(1).cpu().tolist())
        actual.extend(labels.tolist())
        sync(device)
        times.append(time.perf_counter() - start)
    return dict(accuracy=accuracy_score(actual, predicted),
                macro_f1=f1_score(actual, predicted, labels=list(range(config['num_classes'])),
                                  average='macro', zero_division=0),
                samples=len(actual), seconds_per_batch=float(np.mean(times)),
                estimated_full_validation_seconds=float(np.mean(times)) * len(loader))


def provenance(output, config, splits):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (output / 'split_snapshot.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')
    for command, name in [(['rev-parse', 'HEAD'], 'git_commit.txt'), (['status', '--short'], 'git_status.txt')]:
        (output / name).write_text(subprocess.check_output(['git', *command], cwd=ROOT, text=True), encoding='utf-8')
    versions = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (output / 'env.txt').write_text('\n'.join(versions) + '\n', encoding='utf-8')
    source_dir = output / 'source'
    source_dir.mkdir()
    for source in [Path(__file__), ROOT / 'training/common/ast_model.py',
                   ROOT / 'training/common/checkpoint.py',
                   ROOT / 'training/train_baselines.py', ROOT / 'training/prepare_data.py']:
        (source_dir / source.name).write_bytes(source.read_bytes())


def run(config, fold, mode, resume_dir=None, on_progress=None):
    seed_all(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    env = environment()
    device = torch.device(env['device'])
    amp = config['mixed_precision'] and device.type == 'cuda'
    config = dict(config, mode=mode, model='ast', test_fold=fold)
    name = f"{mode}_ast_fold{fold}_seed{config['seed']}_{time.strftime('%Y%m%d_%H%M%S')}"
    output = Path(resume_dir) if resume_dir else ROOT / 'artifacts/report_esc50' / name
    name = output.name
    splits = json.loads((ROOT / config['split_manifest']).read_text(encoding='utf-8'))[str(fold)]
    if config['num_workers'] != 0:
        raise ValueError('Exact epoch resume currently requires num_workers=0')
    if resume_dir:
        if yaml.safe_load((output / 'config.yaml').read_text(encoding='utf-8')) != config:
            raise ValueError('Resume config differs from saved config')
        for saved in (output / 'source').glob('*.py'):
            source = ROOT / 'training' / saved.name
            if not source.exists():
                source = ROOT / 'training/common' / saved.name
            if source.read_bytes() != saved.read_bytes():
                raise ValueError(f'Resume code changed: {source}')
        if json.loads((output / 'environment.json').read_text()) != env:
            raise ValueError('Resume environment differs from original')
    else:
        provenance(output, config, splits)
        json_write(output / 'environment.json', env)
    train = StreamingAudio(splits['train'], config)
    val = StreamingAudio(splits['validation'], config)
    files = sorted(row['filename'] for part in ['train', 'validation'] for row in splits[part])
    manifest = dict(train=splits['train'], validation=splits['validation'],
        held_out_test_files=[r['filename'] for r in splits['test']], test_audio_loaded=False,
        file_list_sha256=hashlib.sha256(('\n'.join(files) + '\n').encode()).hexdigest())
    if resume_dir and json.loads((output / 'data_manifest.json').read_text()) != manifest:
        raise ValueError('Resume data differs from original')
    if not resume_dir:
        json_write(output / 'data_manifest.json', manifest)
    if on_progress:
        on_progress(output, {'phase': 'loading'})
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], pin_memory=False)
    val_loader = DataLoader(val, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=False)
    labels = {r['target']: r['category'] for r in splits['train']}
    model, extractor, layout, initial_loading = create_ast(config, labels)
    json_write(output / 'position_layout.json', layout)
    json_write(output / 'initial_loading.json', initial_loading)
    model.to(device)
    writer = SummaryWriter(str(ROOT / 'runs' / name))
    params = sum(p.numel() for p in model.parameters())
    print(f'{name}: {params:,} parameters, layout={layout}, available_RAM={available_ram()}', flush=True)
    history, benchmarks = [], {}
    best, best_epoch, global_step = -1., None, 0
    best_path = None
    elapsed_before = 0.
    resumed = load_checkpoint(output / 'checkpoint') if resume_dir else None
    if resumed:
        model.load_state_dict(resumed.pop('model'), strict=True)
        progress = resumed['progress']
        history, best, best_epoch, global_step = [progress[key] for key in ['history', 'best', 'best_epoch', 'global_step']]
        best_path, elapsed_before = progress['best_path'], progress['elapsed_seconds']
        print(f'Resuming after {progress["phase"]} epoch {progress["epoch"]}', flush=True)
    run_start = time.perf_counter()
    last_optimizer = None
    try:
        for phase_index, phase in enumerate(['head', 'finetune']):
            if resumed and phase_index < ['head', 'finetune'].index(resumed['progress']['phase']):
                continue
            set_phase(model, phase, config)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=config[f'{phase}_learning_rate'], weight_decay=config['weight_decay'], foreach=False)
            last_optimizer = optimizer
            epochs = config[f'{phase}_epochs']
            accumulation = config['gradient_accumulation']
            steps_per_epoch = math.ceil(len(loader) / accumulation)
            total_steps = steps_per_epoch * epochs
            warmup = max(1, int(total_steps * config['warmup_fraction']))
            def lr_factor(step):
                return (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(
                    math.pi * min(1., (step - warmup) / max(1, total_steps - warmup))))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            scaler = torch.amp.GradScaler('cuda', enabled=amp)
            phase_steps, stale, phase_best = 0, 0, -1.
            first_epoch = 0
            if resumed:
                optimizer.load_state_dict(resumed.pop('optimizer'))
                scheduler.load_state_dict(resumed.pop('scheduler'))
                scaler.load_state_dict(resumed.pop('scaler'))
                progress = resumed['progress']
                first_epoch, stale, phase_best = progress['epoch'], progress['stale'], progress['phase_best']
                restore_rng(resumed.pop('rng'))
                del resumed
                resumed = None
                gc.collect()
            times, ram_samples = [], []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for epoch in range(first_epoch, epochs):
                if phase == 'finetune' and stale >= config['patience']:
                    break
                model.train()
                if phase == 'head':
                    model.audio_spectrogram_transformer.eval()
                iterator = iter(loader)
                losses = []
                for group in range(steps_per_epoch):
                    sync(device)
                    start = time.perf_counter()
                    group_size = min(accumulation, len(loader) - group * accumulation)
                    group_samples = min(config['batch_size'] * group_size,
                        len(train) - group * accumulation * config['batch_size'])
                    optimizer.zero_grad(set_to_none=True)
                    for _ in range(group_size):
                        wave, label = next(iterator)
                        # All common augmentations, including mixup, act on waveforms.
                        wave = augment(wave, config)
                        mix = float(np.random.beta(config['mixup_alpha'], config['mixup_alpha']))
                        order = torch.randperm(len(wave))
                        mixed_wave = mix * wave + (1 - mix) * wave[order]
                        values = features(mixed_wave, extractor, config, True, waveform_augmented=True).to(device)
                        label, order = label.to(device), order.to(device)
                        with torch.autocast(device_type=device.type, enabled=amp):
                            logits = model(input_values=values).logits
                            loss = mix * torch.nn.functional.cross_entropy(logits, label) + (
                                1 - mix) * torch.nn.functional.cross_entropy(logits, label[order])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite AST loss')
                        scaler.scale(loss * len(label) / group_samples).backward()
                        losses.append(loss.item())
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    sync(device)
                    times.append(time.perf_counter() - start)
                    phase_steps += 1
                    global_step += 1
                    ram_samples.append(available_ram())
                    writer.add_scalar(f'{phase}/loss', float(np.mean(losses[-group_size:])), global_step)
                    writer.add_scalar(f'{phase}/lr', optimizer.param_groups[0]['lr'], global_step)
                    if mode == 'benchmark' and phase_steps % 10 == 0:
                        print(f'{phase}: {phase_steps}/{config["benchmark_steps"]} optimizer steps', flush=True)
                    if mode == 'benchmark' and phase_steps >= config['benchmark_steps']:
                        break
                scores = validation(model, extractor, val_loader, config, device, amp,
                                    config['benchmark_validation_batches'] if mode == 'benchmark' else None)
                if mode == 'benchmark':
                    seconds = float(np.mean(times[config['benchmark_warmup_steps']:]))
                    benchmarks[phase] = dict(optimizer_steps=phase_steps, microbatches_per_step=accumulation,
                        seconds_per_optimizer_step=seconds, seconds_per_clip=seconds / (config['batch_size'] * accumulation),
                        optimizer_steps_per_epoch=steps_per_epoch, validation=scores,
                        estimated_phase_seconds=epochs * (steps_per_epoch * seconds + scores['estimated_full_validation_seconds']),
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                        minimum_system_available_ram_bytes=min(v for v in ram_samples if v is not None) if any(v is not None for v in ram_samples) else None,
                        effective_batch=config['batch_size'] * accumulation)
                    print(json.dumps(benchmarks[phase], indent=2), flush=True)
                    break
                record = dict(phase=phase, epoch=epoch + 1, train_loss=float(np.mean(losses)), **scores)
                history.append(record)
                writer.add_scalar('validation/macro_f1', scores['macro_f1'], len(history))
                writer.add_scalar('validation/accuracy', scores['accuracy'], len(history))
                if scores['macro_f1'] > best:
                    best, best_epoch = scores['macro_f1'], len(history)
                    best_path = f'best_models/epoch_{len(history)}'
                    model.save_pretrained(output / best_path, safe_serialization=True)
                    extractor.save_pretrained(output / best_path)
                if scores['macro_f1'] > phase_best:
                    phase_best, stale = scores['macro_f1'], 0
                else:
                    stale += 1
                optimizer.zero_grad(set_to_none=True)
                checkpoint_start = time.perf_counter()
                save_checkpoint(output / 'checkpoint', model, optimizer, scheduler, scaler,
                    dict(phase=phase, epoch=epoch + 1, stale=stale, phase_best=phase_best,
                         history=history, best=best, best_epoch=best_epoch, best_path=best_path,
                         global_step=global_step, elapsed_seconds=elapsed_before + time.perf_counter() - run_start))
                record['checkpoint_seconds'] = time.perf_counter() - checkpoint_start
                atomic_json(output / 'metrics.json', dict(model='ast', parameters=params, complete=False,
                    best_validation_macro_f1=best, best_epoch=best_epoch, history=history,
                    best_model_path=best_path,
                    elapsed_seconds=elapsed_before + time.perf_counter() - run_start, test_evaluated=False))
                # The committed checkpoint and metrics now reference the new best export.
                # Remove only known files in our own obsolete export directories.
                export_root = (output / 'best_models').resolve()
                for obsolete in export_root.glob('epoch_*'):
                    if (obsolete.relative_to(export_root).as_posix() != Path(best_path).name
                            and obsolete.resolve().is_relative_to(export_root)
                            and obsolete.name.removeprefix('epoch_').isdigit()):
                        for filename in ['model.safetensors', 'config.json', 'preprocessor_config.json']:
                            (obsolete / filename).unlink(missing_ok=True)
                        if not any(obsolete.iterdir()):
                            obsolete.rmdir()
                if on_progress:
                    on_progress(output, dict(phase=phase, epoch=epoch + 1, best_validation_macro_f1=best))
                print(json.dumps(record), flush=True)
                if phase == 'finetune' and stale >= config['patience']:
                    break
            optimizer.zero_grad(set_to_none=True)
            del optimizer, scheduler, scaler
            last_optimizer = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if mode == 'benchmark':
            model.eval()
            probe = features(next(iter(val_loader))[0][:1], extractor, config, False)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                expected = model(input_values=probe.to(device)).logits.cpu()
            model.save_pretrained(output / 'roundtrip_model', safe_serialization=True)
            extractor.save_pretrained(output / 'roundtrip_model')
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            restored, loading = ASTForAudioClassification.from_pretrained(output / 'roundtrip_model',
                local_files_only=True, output_loading_info=True,
                attn_implementation=config['attention_implementation'])
            loading = loading_metadata(loading)
            for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
                if loading.get(key):
                    raise ValueError(f'Roundtrip load error: {loading}')
            restored.to(device).eval()
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                actual = restored(input_values=probe.to(device)).logits.cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            metrics = dict(model='ast', mode=mode, parameters=params, phases=benchmarks,
                estimated_seconds_per_fold=sum(p['estimated_phase_seconds'] for p in benchmarks.values()),
                estimated_seconds_all_folds=len(config['folds']) * sum(p['estimated_phase_seconds'] for p in benchmarks.values()),
                roundtrip_loading=loading, roundtrip_passed=True, test_evaluated=False,
                position_layout=layout, elapsed_seconds=time.perf_counter() - run_start)
            json_write(output / 'metrics.json', metrics)
            print(json.dumps(metrics, indent=2), flush=True)
        else:
            metrics = dict(model='ast', parameters=params, complete=True, best_validation_macro_f1=best,
                best_epoch=best_epoch, best_model_path=best_path, history=history,
                elapsed_seconds=elapsed_before + time.perf_counter() - run_start,
                test_evaluated=False)
            atomic_json(output / 'metrics.json', metrics)
    except Exception as error:
        json_write(output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        writer.close()
    return dict(output=str(output), **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'train'], default='benchmark')
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--job-status', help='Optional status JSON within artifacts/')
    parser.add_argument('--resume-job', action='store_true', help='Resume active fold and skip completed folds')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    folds = [args.fold] if args.fold else ([1] if args.mode == 'benchmark' else config['folds'])
    job_path = (ROOT / args.job_status).resolve() if args.job_status else None
    if job_path is not None and not job_path.is_relative_to((ROOT / 'artifacts').resolve()):
        parser.error('job-status must be inside artifacts/')
    results = []
    prior = None
    if args.resume_job:
        if job_path is None or not job_path.is_file() or args.mode != 'train':
            parser.error('resume-job requires existing job-status and mode=train')
        prior = json.loads(job_path.read_text(encoding='utf-8'))
        if prior['config'] != args.config or prior['folds'] != folds:
            parser.error('Job config/folds differ')
        for directory in prior.get('outputs', []):
            metrics = json.loads((Path(directory) / 'metrics.json').read_text())
            if not metrics.get('complete'):
                parser.error('Completed fold has incomplete metrics')
            results.append(dict(output=directory, **metrics))
    def job_status(status, **extra):
        if job_path is not None:
            atomic_json(job_path, dict(status=status, pid=os.getpid(), mode=args.mode,
                config=args.config, folds=folds, completed_folds=len(results),
                heartbeat_unix=time.time(),
                outputs=[r['output'] for r in results], **extra))
    active_output = None
    active_progress = {}
    def progress(output, detail):
        nonlocal active_output, active_progress
        active_output, active_progress = str(output), detail
        job_status('running', current_fold=fold, active_output=active_output, progress=detail)
    try:
        for fold in folds[len(results):]:
            resume_dir = None
            if prior and prior.get('current_fold') == fold:
                candidate = prior.get('active_output')
                if candidate:
                    if not (Path(candidate) / 'checkpoint/latest.json').is_file():
                        raise ValueError('No resumable checkpoint; restart this fold as a new job')
                    resume_dir = candidate
            job_status('running', current_fold=fold, active_output=resume_dir)
            result = run(config, fold, args.mode, resume_dir=resume_dir, on_progress=progress)
            results.append(result)
            active_output, active_progress = None, {}
            job_status('running', current_fold=None)
    except Exception as error:
        job_status('failed', current_fold=fold, active_output=active_output,
                   progress=active_progress, error_type=type(error).__name__, error=str(error))
        raise
    path = ROOT / 'artifacts/report_esc50' / f"ast_{args.mode}_index_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json_write(path, results)
    job_status('complete', index=str(path))
    print(f'Index: {path}', flush=True)


if __name__ == '__main__':
    main()
