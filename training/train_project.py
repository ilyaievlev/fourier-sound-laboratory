"""Public-data-only project AST using the tested resumable two-phase trainer."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import yaml

from common.checkpoint import atomic_json
from common import project_audio
from evaluate_ast import predict
from prepare_data import environment
from train_ast import StreamingAudio as BaseStreamingAudio, features, run
from train_baselines import ROOT


class StreamingAudio(BaseStreamingAudio):
    def __init__(self, rows, config):
        super().__init__(rows, config)
        self.cache = ROOT / 'data/cache/project_waveform_v1'
        self.cache.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index):
        row = self.rows[index]
        target = self.cache / f"{row['audio_sha256']}.npy"
        if not target.exists():
            wave, sr = sf.read(ROOT / self.config['dataset_root'] / 'audio' / row['filename'],
                               dtype='float32', always_2d=True)
            np.save(target, project_audio.prepare(wave, sr), allow_pickle=False)
        return torch.from_numpy(np.load(target, allow_pickle=False)), row['target']


def balanced_loader(dataset, *args, **kwargs):
    """Trainer adapter: balance training draws; validation remains sequential and unique."""
    if kwargs.pop('shuffle', False):
        targets = np.array([row['target'] for row in dataset.rows])
        counts = np.bincount(targets)
        weights = torch.from_numpy(1. / counts[targets])
        kwargs['sampler'] = WeightedRandomSampler(weights, len(targets), replacement=True)
    return DataLoader(dataset, *args, shuffle=False, **kwargs)


@torch.inference_mode()
def validation_threshold(model, extractor, rows, config, device):
    confident_correct = []
    dataset = StreamingAudio(rows, config)
    for wave, label in DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0):
        values = features(wave, extractor, config, False).to(device)
        with torch.autocast(device_type=device.type, enabled=config['mixed_precision'] and device.type == 'cuda'):
            probabilities = model(input_values=values).logits.softmax(-1)
        maxima, predicted = probabilities.max(-1)
        confident_correct.extend(maxima[predicted == label.to(device)].float().cpu().tolist())
    if not confident_correct:
        raise ValueError('No correct validation predictions; cannot estimate a meaningful confidence threshold')
    return dict(max_prob_threshold=float(np.quantile(confident_correct, .05)),
        how='5th percentile of max probability among correctly classified validation clips; retains approx 95%',
        validation_correct_clips=len(confident_correct))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training/configs/ast_project.yaml')
    parser.add_argument('--mode', choices=['benchmark', 'train'], default='benchmark')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / config['project_manifest']).read_text(encoding='utf-8'))
    if manifest['own_recordings'] or manifest['labels'][1] != 'whistling':
        raise ValueError('Unexpected project dataset contract')
    if config['sampling'] != 'inverse_class_frequency_replacement':
        raise ValueError('Unsupported project sampling policy')
    # Inject a loader in this dedicated process, without editing the running A trainer.
    import train_ast
    train_ast.DataLoader = balanced_loader
    train_ast.StreamingAudio = StreamingAudio
    output = ROOT / 'artifacts/ast_project'
    output.mkdir(exist_ok=True)
    state_path = output / 'job_status.json'
    wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    preprocessing_hash = hashlib.sha256(Path(project_audio.__file__).read_bytes()).hexdigest()
    atomic_json(output / 'audio_preprocessing.json', dict(project_audio.CONFIG, source_sha256=preprocessing_hash))
    atomic_json(output / 'sampling.json', dict(policy=config['sampling'],
        unique_train_counts={label: sum(row['label'] == label for row in manifest['splits']['train']) for label in manifest['labels']},
        note='Equal expected training draws per class using replacement; unique sample counts and held-out splits are unchanged'))
    def progress(path, detail):
        atomic_json(state_path, dict(status='training', mode=args.mode, output=str(path), progress=detail,
                                   pid=os.getpid(), wrapper_sha256=wrapper_hash,
                                   preprocessing_sha256=preprocessing_hash, heartbeat_unix=time.time()))
    previous = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
    if previous and previous.get('wrapper_sha256') != wrapper_hash:
        raise ValueError('Project wrapper changed; exact resume is not valid')
    if previous and previous.get('preprocessing_sha256') != preprocessing_hash:
        raise ValueError('Project waveform preprocessing changed; exact resume is not valid')
    if args.mode == 'benchmark':
        print('Project configuration before the 50-optimizer-step benchmark:', flush=True)
        print(yaml.safe_dump(config, sort_keys=False), flush=True)
        result = run(config, 1, 'benchmark')
        atomic_json(output / 'benchmark.json', dict(result, config_snapshot=config,
            wrapper_sha256=wrapper_hash, preprocessing_sha256=preprocessing_hash))
        print('Project 50-step estimate:', result['estimated_seconds_per_fold'], 'seconds', flush=True)
        return
    if (output / 'complete.json').exists():
        print('Project already complete', flush=True)
        return
    benchmark = json.loads((output / 'benchmark.json').read_text())
    if (benchmark.get('config_snapshot') != config or benchmark.get('wrapper_sha256') != wrapper_hash
            or benchmark.get('preprocessing_sha256') != preprocessing_hash):
        raise ValueError('Project benchmark no longer matches config/source; rerun --mode benchmark before training')
    resume_dir = previous.get('output')
    prior_metrics = Path(resume_dir) / 'metrics.json' if resume_dir else None
    if prior_metrics and prior_metrics.exists() and json.loads(prior_metrics.read_text()).get('complete'):
        trained = dict(output=resume_dir, **json.loads(prior_metrics.read_text()))
    else:
        if resume_dir and not (Path(resume_dir) / 'checkpoint/latest.json').exists():
            print('No committed epoch in interrupted project run; restarting from the pinned source weights.', flush=True)
            resume_dir = None
        trained = run(config, 1, 'train', resume_dir=resume_dir, on_progress=progress)
    directory = Path(trained['output'])
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != wrapper_hash:
        raise ValueError('Project wrapper changed while training; refusing a misleading source snapshot')
    if hashlib.sha256(Path(project_audio.__file__).read_bytes()).hexdigest() != preprocessing_hash:
        raise ValueError('Project waveform preprocessing changed while training')
    best = directory / trained['best_model_path']
    for name in ['model.safetensors', 'config.json', 'preprocessor_config.json']:
        shutil.copyfile(best / name, output / name)
    for name in ['config.yaml', 'git_commit.txt', 'env.txt', 'environment.json', 'data_manifest.json']:
        shutil.copyfile(directory / name, output / name)
    (output / 'training_wrapper_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'training_project_audio_source.py').write_bytes(Path(project_audio.__file__).read_bytes())
    atomic_json(output / 'labels_ru.json', manifest['labels_ru'])
    atomic_json(output / 'dataset_protocol.json', {key: value for key, value in manifest.items() if key != 'splits'})
    with (output / 'attributions.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = ['freesound_id', 'uploader', 'title', 'license', 'source_url', 'split', 'label', 'sha256']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for part in manifest['splits'].values():
            for row in part:
                writer.writerow({key: row[key] for key in fields})
    (output / 'LICENSE_NOTES.md').write_text(
        '# Data and model licensing notes\n\nFSD50K source metadata from https://zenodo.org/records/4060432. '
        'Only clip licenses CC0 and CC BY were selected. Per-clip authors, licenses and sources are in attributions.csv. '
        'Dataset compilation: CC BY. Base AST: BSD-3-Clause, pinned revision in config.yaml. '
        'No ESC-50 and no personal recordings used. Dataset pooling/group split is a project-specific protocol, not the official FSD50K benchmark. '
        'Final distribution license and upstream notices must be reviewed before any publication; nothing is uploaded by this script.\n', encoding='utf-8')
    device = torch.device(environment()['device'])
    model = ASTForAudioClassification.from_pretrained(output, local_files_only=True,
        attn_implementation=config['attention_implementation']).to(device).eval()
    extractor = ASTFeatureExtractor.from_pretrained(output, local_files_only=True)
    splits = json.loads((directory / 'split_snapshot.json').read_text())
    eval_config = dict(config, test_fold=1)
    # Freeze the confidence threshold using validation before decoding test waveforms.
    if not (output / 'threshold.json').exists():
        atomic_json(output / 'threshold.json', validation_threshold(model, extractor, splits['validation'], config, device))
    atomic_json(state_path, dict(status='evaluation', output=str(directory), pid=os.getpid(),
                                 wrapper_sha256=wrapper_hash, preprocessing_sha256=preprocessing_hash,
                                 heartbeat_unix=time.time()))
    dataset = StreamingAudio(splits['test'], config)
    conditions, deployed_conditions = {}, {}
    probability_threshold = json.loads((output / 'threshold.json').read_text())['max_prob_threshold']
    for snr in [None, 20, 10, 5, 0]:
        key = 'clean' if snr is None else f'snr_{snr}'
        path = output / f'test_{key}.json'
        if path.exists():
            scores = json.loads(path.read_text())
        else:
            confidences = []
            class WithConfidence:
                def __call__(self, **kwargs):
                    result = model(**kwargs)
                    confidences.extend(result.logits.softmax(-1).max(-1).values.float().cpu().tolist())
                    return result
            scores = predict(WithConfidence(), extractor, dataset, eval_config, device, snr)
            scores['max_probabilities'] = confidences
            atomic_json(path, scores)
        conditions[key] = {key: scores[key] for key in ['accuracy', 'macro_f1', 'inference_seconds_per_clip']}
        deployed = np.where(np.array(scores['max_probabilities']) < probability_threshold,
                            manifest['labels'].index('other'), np.array(scores['predicted']))
        deployed_conditions[key] = dict(accuracy=float(accuracy_score(scores['actual'], deployed)),
            macro_f1=float(f1_score(scores['actual'], deployed, labels=list(range(10)), average='macro', zero_division=0)),
            confusion=confusion_matrix(scores['actual'], deployed, labels=list(range(10))).tolist())
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(scores['confusion'], cmap='Blues')
        ax.set(xticks=range(10), yticks=range(10), xticklabels=manifest['labels'], yticklabels=manifest['labels'],
               xlabel='Predicted', ylabel='True', title=f'Project AST / {key}')
        plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
        fig.tight_layout()
        fig.savefig(output / ('confusion.png' if snr is None else f'confusion_{key}.png'), dpi=160)
        plt.close(fig)
    atomic_json(output / 'metrics.json', dict(task='public project subset', test_evaluated=True, own_recordings=False,
        protocol=manifest['protocol'], training=trained, conditions=conditions,
        conditions_definition='Raw argmax before the confidence threshold; deployed_conditions applies validation threshold -> other',
        deployed_conditions=deployed_conditions,
        class_test_counts={label: sum(row['category'] == label for row in splits['test']) for label in manifest['labels']}))
    atomic_json(output / 'complete.json', dict(complete=True, timestamp=time.time()))
    atomic_json(state_path, dict(status='complete', output=str(directory), pid=os.getpid(),
                                 wrapper_sha256=wrapper_hash, preprocessing_sha256=preprocessing_hash,
                                 heartbeat_unix=time.time()))
    print('Project complete:', conditions['clean'], flush=True)


if __name__ == '__main__':
    main()
