"""Build a public FSD50K project subset including collection-level Whistling.

The split is newly fixed by uploader over dev+eval; it is not the official
200-class FSD50K benchmark. Audio is downloaded file-by-file from a pinned mirror.
Only headers/checksums are inspected here, never test waveform content.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import soundfile as sf
import yaml

from common.downloads import download

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['speech', 'whistling', 'clapping', 'siren', 'music', 'knock', 'cough', 'dog', 'footsteps', 'other']
RUSSIAN = ['голос', 'свист', 'хлопки', 'сирена', 'музыка', 'стук', 'кашель', 'собака', 'шаги', 'другое']
LABELS = {'Speech': 'speech', 'Clapping': 'clapping', 'Siren': 'siren', 'Music': 'music',
          'Knock': 'knock', 'Cough': 'cough', 'Dog': 'dog', 'Bark': 'dog', 'Walk_and_footsteps': 'footsteps'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-per-class', type=int, default=300)
    args = parser.parse_args()
    root = ROOT / 'data/fsd50k'
    revision = json.loads((ROOT / 'data/source_metadata/hf_fsd.json').read_text())['sha']
    esc = {row['src_file'] for row in csv.DictReader((ROOT / 'data/ESC-50-master/meta/esc50.csv').open())}
    candidates = []
    for original in ['dev', 'eval']:
        metadata = json.loads((root / f'FSD50K.metadata/{original}_clips_info_FSD50K.json').read_text())
        collection = {row['fname']: set(row['labels'].split(',')) for row in csv.DictReader(
            (root / f'FSD50K.metadata/collection/collection_{original}.csv').open())}
        for row in csv.DictReader((root / f'FSD50K.ground_truth/{original}.csv').open()):
            fid, labels = row['fname'], set(row['labels'].split(','))
            if not fid.isdigit():
                raise ValueError('Invalid Freesound ID')
            info = metadata[fid]
            license = info['license']
            if fid in esc or not ('/publicdomain/zero/' in license or '/licenses/by/' in license):
                continue
            targets = {LABELS[name] for name in labels if name in LABELS}
            if 'Whistling' in collection.get(fid, set()):
                targets.add('whistling')
            if len(targets) > 1:
                continue
            target = next(iter(targets)) if targets else 'other'
            candidates.append(dict(freesound_id=fid, label=target, target=CLASSES.index(target),
                original_split=original, uploader=info['uploader'], license=license, title=info['title'],
                label_source='FSD50K.collection/Whistling' if target == 'whistling' else 'FSD50K.ground_truth',
                source_url=f'https://freesound.org/s/{fid}/', mirror_path=f'clips/{original}/{fid}.wav'))
    # Deterministic candidate cap before any waveform is opened or any model is trained.
    selected_candidates = []
    for label in CLASSES:
        group = [row for row in candidates if row['label'] == label]
        group.sort(key=lambda row: hashlib.sha256(('2026:' + row['freesound_id']).encode()).hexdigest())
        selected_candidates.extend(group[:args.max_per_class * 3])
    cached = ROOT / 'data/source_metadata/fsd_project_file_info.json'
    file_info = json.loads(cached.read_text()) if cached.exists() else {}
    missing = [row['mirror_path'] for row in selected_candidates if row['mirror_path'] not in file_info]
    for start in range(0, len(missing), 200):
        req = urllib.request.Request(f'https://huggingface.co/api/datasets/Fhrozen/FSD50k/paths-info/{revision}',
            data=json.dumps({'paths': missing[start:start + 200]}).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=90) as response:
            for item in json.load(response):
                file_info[item['path']] = item
        cached.write_text(json.dumps(file_info), encoding='utf-8')
    selected = []
    for label in CLASSES:
        group = [row for row in selected_candidates if row['label'] == label and
                 row['mirror_path'] in file_info and file_info[row['mirror_path']]['size'] <= 2700000]
        selected.extend(group[:args.max_per_class])
    expected_bytes = sum(file_info[row['mirror_path']]['size'] for row in selected)
    if expected_bytes > 2500 * 1024**2:
        raise ValueError('Project subset exceeds the 2.5 GiB download budget')
    print('Selected metadata:', {label: sum(row['label'] == label for row in selected) for label in CLASSES},
          'bytes:', expected_bytes, flush=True)
    def fetch(row):
        info = file_info[row['mirror_path']]
        sha = info.get('lfs', {}).get('oid')
        if not sha:
            raise ValueError('Missing pinned mirror SHA256')
        path = ROOT / 'data/project_audio' / f'{row["freesound_id"]}.wav'
        url = f'https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/{revision}/{row["mirror_path"]}'
        download(url, path, 3 * 1024**2, 'sha256:' + sha)
        header = sf.info(path)
        if header.duration > 30.1 or header.duration < .1:
            return None
        # Model input is the first <=10 seconds. Grouping remains by original uploader/source.
        return dict(row, file=path.relative_to(ROOT).as_posix(), source_duration=header.duration,
                    duration=min(header.duration, 10.), start_seconds=0.,
                    sample_rate=header.samplerate, sha256=sha, mirror_revision=revision)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(fetch, selected)):
            if row:
                rows.append(row)
            if (index + 1) % 100 == 0:
                print(f'Files verified: {index + 1}/{len(selected)}', flush=True)
    targets = np.array([row['target'] for row in rows])
    groups = np.array([row['uploader'] for row in rows])
    assignments = np.full(len(rows), -1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(rows)), targets, groups)):
        assignments[indices] = fold
    splits = {part: [dict(row, split=part) for index, row in enumerate(rows) if assignments[index] in folds]
              for part, folds in [('train', [2, 3, 4]), ('validation', [1]), ('test', [0])]}
    for part, members in splits.items():
        if {row['target'] for row in members} != set(range(10)):
            raise ValueError(f'Missing class in fixed {part} split; record metadata before changing split policy')
    authors = [{row['uploader'] for row in splits[part]} for part in ['train', 'validation', 'test']]
    if authors[0] & authors[1] or authors[0] & authors[2] or authors[1] & authors[2]:
        raise ValueError('Uploader leakage')
    manifest = dict(seed=2026, labels=CLASSES, labels_ru=dict(zip(CLASSES, RUSSIAN)),
        protocol='New project subset pooled over FSD50K dev+eval; StratifiedGroupKFold(5,seed=2026) by uploader: test=0,val=1,train=2,3,4; not official benchmark',
        whistling_note='37 CC0/CC-BY collection-level Whistling candidates before duration filtering; no personal recordings',
        mirror_revision=revision, bytes_selected=expected_bytes, own_recordings=False,
        file_list_sha256=hashlib.sha256(('\n'.join(sorted(row['file'] for row in rows)) + '\n').encode()).hexdigest(),
        test_waveforms_decoded=False, splits=splits)
    manifest['segment_policy'] = 'First min(10,duration) seconds; no content-based selection; grouped by original uploader before training'
    destination = ROOT / 'data/project_manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError('Refusing to change an existing project split')
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    audio_dir = ROOT / 'data/fsd_project/audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    converted = {}
    for part, members in splits.items():
        converted[part] = []
        for row in members:
            source, target = ROOT / row['file'], audio_dir / f'{row["freesound_id"]}.wav'
            if not target.exists():
                os.link(source, target)
            converted[part].append(dict(filename=target.name, target=row['target'], category=row['label'],
                                       src_file=row['freesound_id'], uploader=row['uploader']))
    (ROOT / 'data/project_ast_splits.json').write_text(json.dumps({'1': converted}, indent=2), encoding='utf-8')
    print('Manifest saved:', {part: {label: sum(row['label'] == label for row in members) for label in CLASSES}
          for part, members in splits.items()}, flush=True)


if __name__ == '__main__':
    main()
