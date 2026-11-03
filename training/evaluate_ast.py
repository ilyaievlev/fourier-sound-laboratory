"""Final AST evaluation only after all five folds are complete; resumable by condition."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import yaml

from common.checkpoint import atomic_json
from evaluate import write_csv
from prepare_data import environment
from train_ast import StreamingAudio, features
from train_baselines import ROOT, seed_all, sync


@torch.inference_mode()
def predict(model, extractor, dataset, config, device, snr):
    actual, predicted = [], []
    elapsed = 0.
    rng = torch.Generator().manual_seed(config['seed'] + config['test_fold'])
    for wave, label in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        if snr is not None:
            noise = torch.randn(wave.shape, generator=rng)
            wave = wave + noise * wave.square().mean(1, keepdim=True).sqrt() / (
                noise.square().mean(1, keepdim=True).sqrt().clamp_min(1e-10)) * 10 ** (-snr / 20)
        sync(device)
        start = time.perf_counter()
        values = features(wave, extractor, config, False).to(device)
        with torch.autocast(device_type=device.type, enabled=config['mixed_precision'] and device.type == 'cuda'):
            result = model(input_values=values).logits.argmax(1).cpu()
        sync(device)
        elapsed += time.perf_counter() - start
        actual.extend(label.tolist())
        predicted.extend(result.tolist())
    return dict(accuracy=accuracy_score(actual, predicted),
        macro_f1=f1_score(actual, predicted, labels=list(range(config['num_classes'])), average='macro', zero_division=0),
        confusion=confusion_matrix(actual, predicted, labels=list(range(config['num_classes']))).tolist(),
        inference_seconds_per_clip=elapsed / len(dataset), batch_size=1,
        actual=actual, predicted=predicted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-index', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cnn-report', default='artifacts/report_esc50/evaluation_20261102_201546')
    args = parser.parse_args()
    index = ROOT / args.training_index
    entries = json.loads(index.read_text())
    prepared = []
    for entry in entries:
        directory = Path(entry['output'])
        config = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
        metrics = json.loads((directory / 'metrics.json').read_text())
        if not metrics.get('complete') or config['mode'] != 'train' or config['model'] != 'ast':
            raise ValueError(f'Incomplete AST run: {directory}')
        prepared.append((directory, config, metrics))
    if len(prepared) != 5 or {config['test_fold'] for _, config, _ in prepared} != set(range(1, 6)):
        raise ValueError('Final evaluation requires exactly five unique completed AST folds')
    report = (ROOT / args.output_dir).resolve()
    if not report.is_relative_to((ROOT / 'artifacts').resolve()):
        parser.error('output-dir must be inside artifacts/')
    report.mkdir(parents=True, exist_ok=True)
    protocol = dict(training_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snr_db=[20, 10, 5, 0], noise_seed_rule='seed + fold; same standard-normal sequence as CNN',
        std_ddof=1, test_used_for_selection=False, batch_size=1,
        split_protocol='official test folds; overlapping sources purged from train/validation')
    protocol_path = report / 'evaluation_config.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Cannot resume evaluation with changed protocol/code/index')
    atomic_json(protocol_path, protocol)
    (report / 'evaluate_source.py').write_bytes(Path(__file__).read_bytes())
    (report / 'training_index.json').write_bytes(index.read_bytes())
    rows, snr_rows = [], []
    for directory, config, metrics in prepared:
        seed_all(config['seed'])
        torch.set_num_threads(config['cpu_threads'])
        device = torch.device(environment()['device'])
        best = directory / metrics['best_model_path']
        model, loading = ASTForAudioClassification.from_pretrained(best, local_files_only=True,
            attn_implementation=config['attention_implementation'], output_loading_info=True)
        if any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError(f'Invalid exported AST: {loading}')
        extractor = ASTFeatureExtractor.from_pretrained(best, local_files_only=True)
        model.to(device).eval()
        final = report / f'fold_{config["test_fold"]}'
        final.mkdir(exist_ok=True)
        split = json.loads((directory / 'split_snapshot.json').read_text())
        dataset = StreamingAudio(split['test'], config)
        atomic_json(final / 'test_manifest.json', split['test'])
        for snr in [None, 20, 10, 5, 0]:
            condition = 'clean' if snr is None else f'snr_{snr}'
            path = final / f'{condition}.json'
            if path.exists():
                scores = json.loads(path.read_text())
            else:
                scores = predict(model, extractor, dataset, config, device, snr)
                atomic_json(path, scores)
            if snr is None:
                rows.append(dict(model='ast', fold=config['test_fold'], accuracy=scores['accuracy'],
                    macro_f1=scores['macro_f1'], parameters=metrics['parameters'],
                    training_seconds=metrics['elapsed_seconds'],
                    inference_seconds_per_clip=scores['inference_seconds_per_clip']))
            else:
                snr_rows.append(dict(model='ast', fold=config['test_fold'], snr_db=snr,
                    accuracy=scores['accuracy'], macro_f1=scores['macro_f1']))
            fig, ax = plt.subplots(figsize=(12, 10))
            plot = ax.imshow(scores['confusion'], cmap='Blues')
            ax.set(title=f'AST / fold {config["test_fold"]} / {condition}', xlabel='Predicted ID', ylabel='True ID')
            fig.colorbar(plot, ax=ax)
            fig.tight_layout()
            fig.savefig(final / f'confusion_{condition}.png', dpi=150)
            plt.close(fig)
        print(json.dumps(rows[-1]), flush=True)
        del model, dataset
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    with (ROOT / args.cnn_report / 'fold_metrics.csv').open(newline='') as handle:
        cnn_rows = list(csv.DictReader(handle))
    with (ROOT / args.cnn_report / 'snr.csv').open(newline='') as handle:
        cnn_snr = list(csv.DictReader(handle))
    all_rows, all_snr = cnn_rows + rows, cnn_snr + snr_rows
    write_csv(report / 'fold_metrics.csv', all_rows)
    write_csv(report / 'snr.csv', all_snr)
    summaries = []
    for name in ['wave_cnn', 'mel_cnn', 'ast']:
        group = [row for row in all_rows if row['model'] == name]
        record = dict(model=name, folds=len(group))
        for key in ['accuracy', 'macro_f1', 'training_seconds', 'inference_seconds_per_clip']:
            values = [float(row[key]) for row in group]
            record[key + '_mean'], record[key + '_std'] = float(np.mean(values)), float(np.std(values, ddof=1))
        summaries.append(record)
    write_csv(report / 'summary.csv', summaries)
    for filename in ['fold_metrics.csv', 'snr.csv', 'summary.csv']:
        pending = report.parent / (filename + '.tmp')
        pending.write_bytes((report / filename).read_bytes())
        pending.replace(report.parent / filename)
    atomic_json(report / 'complete.json', dict(complete=True, test_evaluated=True, timestamp=time.time()))
    atomic_json(report.parent / 'latest_evaluation.json', dict(report=str(report)))
    print(f'Completed comparison: {report}', flush=True)


if __name__ == '__main__':
    main()
