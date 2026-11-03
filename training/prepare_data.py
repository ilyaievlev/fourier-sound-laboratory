"""Validate local ESC-50 metadata and create five leakage-free split manifests.

Reads metadata and checks file existence only: held-out audio is never decoded.
Does not download data or start training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_splits(rows, expected_clips=2000, expected_classes=50, expected_folds=5,
                 source_overlap_policy='error'):
    if source_overlap_policy not in {'error', 'purge_training_and_validation'}:
        raise ValueError('Unknown source overlap policy')
    if len(rows) != expected_clips:
        raise ValueError(f"Expected {expected_clips} clips, got {len(rows)}")
    names, sources, labels, folds = set(), {}, {}, set()
    normalized = []
    for row in rows:
        name = row['filename']
        if not name or '/' in name or '\\' in name or name in names:
            raise ValueError(f"Invalid or duplicate filename: {name!r}")
        names.add(name)
        fold, target = int(row['fold']), int(row['target'])
        source = row['src_file']
        if not source:
            raise ValueError('Missing source ID')
        if source in sources and sources[source] != fold and source_overlap_policy == 'error':
            raise ValueError(f"Source {source} occurs in multiple folds")
        sources[source] = fold
        if target in labels and labels[target] != row['category']:
            raise ValueError(f"Inconsistent label {target}")
        labels[target] = row['category']
        folds.add(fold)
        normalized.append(dict(filename=name, fold=fold, target=target,
                               category=row['category'], src_file=source))
    if folds != set(range(1, expected_folds + 1)):
        raise ValueError(f"Unexpected folds: {folds}")
    if set(labels) != set(range(expected_classes)):
        raise ValueError('Unexpected class IDs')
    for fold in folds:
        if {r['target'] for r in normalized if r['fold'] == fold} != set(labels):
            raise ValueError(f"Missing classes in fold {fold}")
    normalized.sort(key=lambda r: r['filename'])
    splits = {
        str(k): {
            part: [r for r in normalized if r['fold'] in selected]
            for part, selected in {
                'train': folds - {k, k % expected_folds + 1},
                'validation': {k % expected_folds + 1}, 'test': {k},
            }.items()
        } for k in sorted(folds)
    }
    for split in splits.values():
        test_sources = {r['src_file'] for r in split['test']}
        original_val, original_train = split['validation'], split['train']
        split['validation'] = [r for r in original_val if r['src_file'] not in test_sources]
        held_out = test_sources | {r['src_file'] for r in split['validation']}
        split['train'] = [r for r in original_train if r['src_file'] not in held_out]
        split['excluded'] = [dict(r, excluded_from=part) for part, original in
                             [('train', original_train), ('validation', original_val)]
                             for r in original if r not in split[part]]
    return splits


def environment():
    import torch
    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')
    packages = {}
    for name in ['torch', 'transformers', 'librosa', 'scikit-learn',
                 'PyYAML', 'tensorboard', 'matplotlib', 'safetensors']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), device=device, packages=packages,
                gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                cuda_build=torch.version.cuda)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check-env', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if args.check_env:
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
        return
    dataset = ROOT / config['dataset_root']
    metadata = dataset / 'meta' / 'esc50.csv'
    if not metadata.is_file():
        parser.error(f'Missing {metadata}. Place extracted ESC-50 here first.')
    with metadata.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    splits = build_splits(rows, config['expected_clips'], config['expected_classes'],
                          config['expected_folds'], config.get('source_overlap_policy', 'error'))
    files = sorted(r['filename'] for r in rows)
    missing = [name for name in files if not (dataset / 'audio' / name).is_file()]
    if missing:
        parser.error(f'{len(missing)} audio files missing; first: {missing[0]}')
    output = (ROOT / config['output_dir']).resolve()
    if not output.is_relative_to((ROOT / 'artifacts').resolve()):
        parser.error('output_dir must be inside artifacts/')
    # Refuse to overwrite previous provenance.
    output.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'config.yaml').write_text(config_path.read_text(encoding='utf-8'), encoding='utf-8')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True)
    (output / 'git_commit.txt').write_text(commit, encoding='utf-8')
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    (output / 'git_status.txt').write_text(status, encoding='utf-8')
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (output / 'env.txt').write_text('\n'.join(packages) + '\n', encoding='utf-8')
    write_json('environment.json', environment())
    write_json('data_manifest.json', {
        'dataset_root': str(dataset), 'files': files,
        'file_list_sha256': hashlib.sha256(('\n'.join(files) + '\n').encode()).hexdigest(),
        'metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'audio_content_hashed': False,
    })
    write_json('splits.json', splits)
    write_json('split_audit.json', {
        'policy': config.get('source_overlap_policy', 'error'),
        'note': 'Official test folds retained; overlapping sources purged from train/validation. Modified benchmark protocol.',
        'folds': {k: {part: len(rows) for part, rows in split.items()} for k, split in splits.items()},
        'excluded': {k: split['excluded'] for k, split in splits.items()},
    })
    write_json('metrics.json', {'stage': 'data_preparation', 'clips': len(files),
                               'model_metrics': None})
    print(f'Validated {len(files)} clips; split manifests saved to {output}')


if __name__ == '__main__':
    main()
