"""One final ESC-50 test evaluation of completed CNN runs; no model selection."""
import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader
import yaml

from common.baseline_models import make_model
from train_baselines import ROOT, json_write, load_audio, seed_all, sync
from prepare_data import environment


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def predict(model, dataset, device, amp, snr, seed):
    actual, predicted = [], []
    elapsed = 0.
    rng = torch.Generator(device='cpu').manual_seed(seed)
    # Batch size 1 measures per-clip latency, including transfer to/from GPU.
    for wave, label in DataLoader(dataset, batch_size=1, shuffle=False):
        if snr is not None:
            noise = torch.randn(wave.shape, generator=rng)
            noise_rms = noise.square().mean(1, keepdim=True).sqrt().clamp_min(1e-10)
            signal_rms = wave.square().mean(1, keepdim=True).sqrt()
            wave = wave + noise * signal_rms / noise_rms * (10 ** (-snr / 20))
        sync(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp):
            result = model(wave.to(device)).argmax(1).cpu()
        sync(device)
        elapsed += time.perf_counter() - start
        actual.extend(label.tolist())
        predicted.extend(result.tolist())
    return dict(accuracy=accuracy_score(actual, predicted),
                macro_f1=f1_score(actual, predicted, labels=list(range(50)), average='macro', zero_division=0),
                confusion=confusion_matrix(actual, predicted, labels=list(range(50))).tolist(),
                inference_seconds_per_clip=elapsed / len(dataset), batch_size=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-index', required=True)
    args = parser.parse_args()
    index = Path(args.training_index)
    if not index.is_absolute():
        index = ROOT / index
    entries = json.loads(index.read_text(encoding='utf-8'))
    # Validate every run before opening any held-out test audio.
    prepared = []
    seen = set()
    for entry in entries:
        directory = Path(entry['output'])
        metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
        config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
        if not metrics.get('complete') or config['mode'] != 'train':
            raise ValueError(f'Incomplete training: {directory}')
        key = (config['model'], config['test_fold'])
        if key in seen:
            raise ValueError(f'Duplicate model/fold: {key}')
        seen.add(key)
        if (directory / 'final_test').exists():
            raise FileExistsError(f'Test already evaluated or started: {directory}')
        prepared.append((directory, config, metrics))
    report = ROOT / 'artifacts/report_esc50' / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
    report.mkdir(exist_ok=False)
    (report / 'training_index.json').write_bytes(index.read_bytes())
    (report / 'evaluate_source.py').write_bytes(Path(__file__).read_bytes())
    json_write(report / 'environment.json', environment())
    json_write(report / 'evaluation_config.json', dict(noise_seed=2026, snr_db=[20, 10, 5, 0],
               noise_seed_rule='seed + fold; identical standard-normal noise across models/SNRs',
               std_ddof=1, test_used_for_selection=False, batch_size=1,
               split_protocol='official test folds; source overlap purged from train/validation'))
    rows, snr_rows = [], []
    for directory, config, train_metrics in prepared:
        seed_all(config['seed'])
        torch.set_num_threads(config['cpu_threads'])
        device = torch.device(environment()['device'])
        amp = config['mixed_precision'] and device.type == 'cuda'
        model = make_model(config['model'], config).to(device)
        model.load_state_dict(load_file(str(directory / 'model.safetensors'), device=str(device)), strict=True)
        model.eval()
        final = directory / 'final_test'
        final.mkdir(exist_ok=False)
        splits = json.loads((ROOT / config['split_manifest']).read_text(encoding='utf-8'))
        test_rows = splits[str(config['test_fold'])]['test']
        dataset = load_audio(test_rows, config)
        json_write(final / 'test_manifest.json', test_rows)
        clean = predict(model, dataset, device, amp, None, config['seed'])
        json_write(final / 'metrics.json', dict(clean, test_evaluated=True, model=config['model'],
                   fold=config['test_fold'], parameters=train_metrics['parameters'],
                   training_seconds=train_metrics['elapsed_seconds']))
        json_write(final / 'confusion_clean.json', clean['confusion'])
        fig, ax = plt.subplots(figsize=(12, 10))
        plot = ax.imshow(clean['confusion'], cmap='Blues')
        ax.set(title=f"{config['model']} / test fold {config['test_fold']}",
               xlabel='Predicted class ID', ylabel='True class ID')
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(final / 'confusion_clean.png', dpi=160)
        plt.close(fig)
        row = dict(model=config['model'], fold=config['test_fold'], accuracy=clean['accuracy'],
                   macro_f1=clean['macro_f1'], parameters=train_metrics['parameters'],
                   training_seconds=train_metrics['elapsed_seconds'],
                   inference_seconds_per_clip=clean['inference_seconds_per_clip'])
        rows.append(row)
        for snr in [20, 10, 5, 0]:
            # Same noise sequence for both models and all SNR levels in each fold.
            noisy = predict(model, dataset, device, amp, snr, config['seed'] + config['test_fold'])
            json_write(final / f'snr_{snr}.json', noisy)
            snr_rows.append(dict(model=config['model'], fold=config['test_fold'], snr_db=snr,
                                 accuracy=noisy['accuracy'], macro_f1=noisy['macro_f1']))
        print(json.dumps(row), flush=True)
    write_csv(report / 'fold_metrics.csv', rows)
    write_csv(report / 'snr.csv', snr_rows)
    summary = []
    for model in sorted({r['model'] for r in rows}):
        group = [r for r in rows if r['model'] == model]
        record = dict(model=model, folds=len(group))
        for key in ['accuracy', 'macro_f1', 'training_seconds', 'inference_seconds_per_clip']:
            values = [r[key] for r in group]
            record[f'{key}_mean'] = float(np.mean(values))
            record[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        summary.append(record)
    write_csv(report / 'summary.csv', summary)
    for filename in ['summary.csv', 'fold_metrics.csv', 'snr.csv']:
        (report.parent / filename).write_bytes((report / filename).read_bytes())
    json_write(report.parent / 'latest_evaluation.json', dict(report=str(report)))
    print(f'Final evaluation: {report}', flush=True)


if __name__ == '__main__':
    main()
