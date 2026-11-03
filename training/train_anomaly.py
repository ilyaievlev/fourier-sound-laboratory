"""Pump machine-ID CNN and persisted AST+LOF, trained on normal audio only."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import shutil
import time

import joblib
import numpy as np
from safetensors.torch import load_file, save_file
from sklearn.neighbors import LocalOutlierFactor
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import ASTFeatureExtractor, ASTModel
import yaml

from common.anomaly_model import from_config, true_id_score
from common.anomaly_metrics import evaluate_scores
from common.features import DEFAULT_CONFIG, pump_logmel, sliding_crops
from common.checkpoint import atomic_json, load_checkpoint, restore_rng, save_checkpoint
from prepare_data import environment
from train_baselines import ROOT, seed_all, sync


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PumpData(Dataset):
    def __init__(self, rows, config):
        self.rows, self.config = rows, config
        self.cache = ROOT / 'data/cache/pump_logmel_v1'
        self.cache.mkdir(parents=True, exist_ok=True)

    def logmel(self, row):
        path = ROOT / row['file']
        # Split-relative filename prevents train/test name collisions.
        key = hashlib.sha256(row['file'].encode()).hexdigest()
        cached = self.cache / f'{key}.npy'
        if not cached.exists():
            wave, sr = sf.read(path, dtype='float32')
            if wave.ndim != 1 or sr != 16000:
                raise ValueError('Unexpected pump audio format')
            np.save(cached, pump_logmel(wave, sr), allow_pickle=False)
        return np.load(cached, allow_pickle=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mel = self.logmel(row)
        start = int(np.random.randint(max(1, mel.shape[1] - self.config['crop_frames'] + 1)))
        return torch.from_numpy(mel[:, start:start + self.config['crop_frames']]), self.config['id_list'].index(row['machine_id'])


@torch.inference_mode()
def cnn_scores(model, dataset, config, device):
    model.eval()
    result, correct = [], 0
    for row in dataset.rows:
        crops = torch.from_numpy(sliding_crops(dataset.logmel(row), config['crop_frames'], config['crop_hop_frames'])).to(device)
        logits = model(crops)
        true = config['id_list'].index(row['machine_id'])
        correct += int(logits.softmax(-1).mean(0).argmax().item() == true)
        result.append(dict(file=row['file'], machine_id=row['machine_id'], label=row['label'],
                           score=float(true_id_score(logits, true))))
    return result, correct / len(dataset)


def provenance(output, config, manifest):
    output.mkdir(parents=True, exist_ok=True)
    packages = '\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())) + '\n'
    if (output / 'config.yaml').exists() and yaml.safe_load((output / 'config.yaml').read_text()) != config:
        raise ValueError('Refusing to change pump config in an existing experiment')
    if (output / 'data_manifest.json').exists() and json.loads((output / 'data_manifest.json').read_text()) != manifest:
        raise ValueError('Refusing to change pump data in an existing experiment')
    sources = [Path(__file__), *list((ROOT / 'training/common').glob('anomaly_*.py')),
               ROOT / 'training/common/features.py', ROOT / 'training/common/checkpoint.py']
    if (output / 'env.txt').exists() and (output / 'data_manifest.json').exists():
        if (output / 'env.txt').read_text() != packages:
            raise ValueError('Pump environment changed; exact training resume is not valid')
        for path in sources:
            saved = output / 'source' / path.name
            if not saved.exists() or saved.read_bytes() != path.read_bytes():
                raise ValueError('Pump experiment source changed; do not mix existing embeddings/models with new code')
        return
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (output / 'git_commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True))
    (output / 'git_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True))
    (output / 'env.txt').write_text(packages)
    atomic_json(output / 'data_manifest.json', manifest)
    atomic_json(output / 'environment.json', environment())
    source = output / 'source'
    source.mkdir(exist_ok=True)
    for path in sources:
        (source / path.name).write_bytes(path.read_bytes())


def thresholds_from_val(scores, config):
    return {machine: float(np.quantile([row['score'] for row in scores if row['machine_id'] == machine],
                                       config['threshold_quantile'])) for machine in config['id_list']}


def train_cnn(output, manifest, config, device):
    model_config = dict(architecture='PumpIDCNN', num_ids=len(config['id_list']), channels=[32, 64, 128, 192],
                        input_offset=40., input_scale=40., crop_frames=config['crop_frames'],
                        crop_hop_frames=config['crop_hop_frames'])
    atomic_json(output / 'model_config.json', model_config)
    atomic_json(output / 'feature_config.json', DEFAULT_CONFIG)
    atomic_json(output / 'id_list.json', config['id_list'])
    if (output / 'cnn_complete.json').exists():
        return
    seed_all(config['seed'])
    train, val = [PumpData(manifest['splits'][part], config) for part in ['train', 'validation']]
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    # Benchmark a separate initialization; include data loading and full normal validation.
    if not (output / 'benchmark.json').exists():
        model = from_config(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        model.train()
        iterator, times = iter(loader), []
        for step in range(config['benchmark_steps']):
            sync(device)
            start = time.perf_counter()
            try:
                values, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                values, labels = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(values.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            sync(device)
            times.append(time.perf_counter() - start)
        start = time.perf_counter()
        cnn_scores(model, val, config, device)
        sync(device)
        validation_seconds = time.perf_counter() - start
        seconds_per_step = float(np.mean(times[5:]))
        estimate = (seconds_per_step * len(loader) + validation_seconds) * config['epochs']
        atomic_json(output / 'benchmark.json', dict(steps=len(times), seconds_per_step=seconds_per_step,
            validation_seconds=validation_seconds, estimated_training_seconds=estimate, test_evaluated=False))
        print(yaml.safe_dump(config, sort_keys=False), flush=True)
        print(f'Pump CNN 50-step estimate including data loading and validation: {estimate:.0f}s', flush=True)
        del model, optimizer, iterator
    seed_all(config['seed'])
    model = from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    writer = SummaryWriter(str(ROOT / 'runs/anomaly_pump_id'))
    best, stale, history, first_epoch, elapsed_before, best_path = -1., 0, [], 0, 0., None
    if (output / 'checkpoint/latest.json').exists():
        saved = load_checkpoint(output / 'checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        progress = saved['progress']
        best, stale, history = progress['best'], progress['stale'], progress['history']
        first_epoch, elapsed_before, best_path = progress['epoch'], progress['elapsed_seconds'], progress['best_path']
        restore_rng(saved['rng'])
        del saved
        print(f'Resuming pump CNN after epoch {first_epoch}', flush=True)
    exports = output / 'cnn_best_models'
    exports.mkdir(exist_ok=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, config['epochs']):
        if stale >= config['patience']:
            break
        model.train()
        losses = []
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(values.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pump loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        _, accuracy = cnn_scores(model, val, config, device)
        record = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_id_accuracy=accuracy)
        history.append(record)
        writer.add_scalar('train/loss', record['train_loss'], epoch)
        writer.add_scalar('validation/id_accuracy', accuracy, epoch)
        if accuracy > best:
            best, stale = accuracy, 0
            best_path = f'cnn_best_models/epoch_{epoch + 1}.safetensors'
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(output / best_path))
        else:
            stale += 1
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = elapsed_before + time.perf_counter() - started
        save_checkpoint(output / 'checkpoint', model, optimizer, scheduler, scaler,
            dict(epoch=epoch + 1, best=best, stale=stale, history=history, best_path=best_path, elapsed_seconds=elapsed))
        atomic_json(output / 'training_metrics.json', dict(history=history, best_validation_id_accuracy=best,
                    elapsed_seconds=elapsed, parameters=sum(p.numel() for p in model.parameters()), best_model_path=best_path))
        print(json.dumps(record), flush=True)
        if stale >= config['patience']:
            break
    writer.close()
    if best_path is None:
        raise ValueError('Pump training has no committed best checkpoint')
    pending = output / 'model.safetensors.tmp'
    shutil.copyfile(output / best_path, pending)
    pending.replace(output / 'model.safetensors')
    model.load_state_dict(load_file(str(output / 'model.safetensors'), device=str(device)))
    val_scores, _ = cnn_scores(model, val, config, device)
    thresholds = thresholds_from_val(val_scores, config)
    atomic_json(output / 'threshold.json', dict(by_id=thresholds, how='Quantile of normal validation clip mean negative-log true-ID probability', quantile=config['threshold_quantile']))
    csv_write(output / 'validation_scores.csv', val_scores)
    # Only now open final test audio, after weights and thresholds are frozen.
    scores, _ = cnn_scores(model, PumpData(manifest['splits']['test'], config), config, device)
    metrics, roc = evaluate_scores(scores, thresholds, config['id_list'])
    atomic_json(output / 'metrics.json', metrics)
    atomic_json(output / 'roc.json', roc)
    csv_write(output / 'test_scores.csv', scores)
    atomic_json(output / 'cnn_complete.json', dict(complete=True, timestamp=time.time()))


@torch.inference_mode()
def ast_embeddings(rows, model, extractor, device, target):
    if target.exists():
        return np.load(target, allow_pickle=False)
    vectors = []
    for start in range(0, len(rows), 4):
        waveforms = []
        for row in rows[start:start + 4]:
            wave, sr = sf.read(ROOT / row['file'], dtype='float32')
            if sr != 16000:
                raise ValueError('Expected 16 kHz pump')
            waveforms.append(wave)
        values = extractor(waveforms, sampling_rate=16000, return_tensors='pt')['input_values'].to(device)
        embeddings = model(input_values=values).last_hidden_state.mean(1)
        vectors.append(embeddings.float().cpu().numpy())
        if start % 400 == 0:
            print(f'Embeddings {target.stem}: {min(start + 4, len(rows))}/{len(rows)}', flush=True)
    result = np.concatenate(vectors)
    pending = target.with_suffix('.partial.npy')
    np.save(pending, result, allow_pickle=False)
    pending.replace(target)
    return result


def train_lof(output, manifest, config, device):
    directory = output / 'lof'
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        return
    atomic_json(directory / 'embedding_config.json', dict(model=config['ast_model_id'], revision=config['ast_revision'],
        max_length=1024, dtype='float32', pooling='mean last_hidden_state over all tokens including special/padded tokens', dimensions=768,
        id_list=config['id_list'], neighbors=config['lof_neighbors'], novelty=True, score='-score_samples',
        sklearn=importlib.metadata.version('scikit-learn'), joblib=importlib.metadata.version('joblib')))
    atomic_json(directory / 'embedding_manifest.json', manifest['splits'])
    extractor = ASTFeatureExtractor.from_pretrained(config['ast_model_id'], revision=config['ast_revision'], local_files_only=True)
    model = ASTModel.from_pretrained(config['ast_model_id'], revision=config['ast_revision'], local_files_only=True,
                                     attn_implementation='sdpa').to(device).eval()
    train = ast_embeddings(manifest['splits']['train'], model, extractor, device, directory / 'train_embeddings.npy')
    val = ast_embeddings(manifest['splits']['validation'], model, extractor, device, directory / 'validation_embeddings.npy')
    models, thresholds = {}, {}
    validation_scores = []
    for machine in config['id_list']:
        indices = [index for index, row in enumerate(manifest['splits']['train']) if row['machine_id'] == machine]
        lof = LocalOutlierFactor(n_neighbors=config['lof_neighbors'], novelty=True, n_jobs=2)
        lof.fit(train[indices])
        models[machine] = lof
        path = directory / f'{machine}.joblib'
        joblib.dump(lof, path)
        vi = [index for index, row in enumerate(manifest['splits']['validation']) if row['machine_id'] == machine]
        scores = -lof.score_samples(val[vi])
        thresholds[machine] = float(np.quantile(scores, config['threshold_quantile']))
        validation_scores.extend(dict(manifest['splits']['validation'][index], score=float(score)) for index, score in zip(vi, scores))
        # Persistence is mandatory: verify scores from the serialized model.
        np.testing.assert_allclose(joblib.load(path).score_samples(val[vi]), -scores, rtol=0, atol=0)
    atomic_json(directory / 'threshold.json', dict(by_id=thresholds, quantile=config['threshold_quantile'], how='Quantile of normal validation -score_samples by ID'))
    csv_write(directory / 'validation_scores.csv', validation_scores)
    test = ast_embeddings(manifest['splits']['test'], model, extractor, device, directory / 'test_embeddings.npy')
    rows = []
    for index, row in enumerate(manifest['splits']['test']):
        rows.append(dict(file=row['file'], machine_id=row['machine_id'], label=row['label'],
                         score=float(-models[row['machine_id']].score_samples(test[index:index + 1])[0])))
    metrics, roc = evaluate_scores(rows, thresholds, config['id_list'])
    atomic_json(directory / 'metrics.json', metrics)
    atomic_json(directory / 'roc.json', roc)
    csv_write(directory / 'test_scores.csv', rows)
    atomic_json(directory / 'complete.json', dict(complete=True, timestamp=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training/configs/pump.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    seed_all(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(environment()['device'])
    output = ROOT / config['output_dir']
    manifest = json.loads((ROOT / config['manifest']).read_text())
    provenance(output, config, manifest)
    train_cnn(output, manifest, config, device)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    train_lof(output, manifest, config, device)
    cnn = json.loads((output / 'metrics.json').read_text())
    lof = json.loads((output / 'lof/metrics.json').read_text())
    winner = 'lof' if lof['mean_auc'] > cnn['mean_auc'] else 'id_cnn'
    atomic_json(output / 'deployment.json', dict(approach=winner, id_cnn_mean_auc=cnn['mean_auc'], lof_mean_auc=lof['mean_auc'],
        selection='Predeclared comparison by mean final test AUC; both approaches reported, no test tuning',
        license='CC-BY-NC-SA-4.0', intended_use='Educational demo on DCASE2020 pump IDs only'))
    (output / 'LICENSE_NOTES.md').write_text('DCASE 2020 Task 2 dev: CC BY-NC-SA 4.0.\nSource: https://zenodo.org/records/3678171\nNot published. Educational pump-only detector; no reliability claim for real equipment.\n', encoding='utf-8')
    print('Pump complete:', winner, cnn['mean_auc'], lof['mean_auc'], flush=True)


if __name__ == '__main__':
    main()
