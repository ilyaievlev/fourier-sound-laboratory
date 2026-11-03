"""Prepare the official DCASE2020 development pump dataset; test audio stays unopened."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from sklearn.model_selection import train_test_split
import yaml

from common.downloads import download, extract, record_info

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training/configs/pump.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    data = ROOT / config['dataset_root']
    info = record_info(3678171, ROOT / 'data/source_metadata')
    entry = next(row for row in info['files'] if row['key'] == 'dev_data_pump.zip')
    archive = download(entry['links']['self'], ROOT / 'data/archives/dev_data_pump.zip',
                       1200 * 1024**2, entry['checksum'])
    if not (data / 'pump/train').exists():
        extract(archive, data)
    candidates = list(data.rglob('pump/train'))
    if len(candidates) != 1:
        raise ValueError(f'Unexpected pump layout: {candidates}')
    pump = candidates[0].parent
    rows = []
    for part in ['train', 'test']:
        for path in sorted((pump / part).glob('*.wav')):
            match = re.fullmatch(r'(normal|anomaly)_id_(\d\d)_\d+\.wav', path.name)
            if not match:
                raise ValueError(path.name)
            condition, machine = match.groups()
            if 'id_' + machine not in config['id_list']:
                raise ValueError('Unexpected machine ID')
            if part == 'train' and condition != 'normal':
                raise ValueError('Anomaly in training data')
            rows.append(dict(file=path.relative_to(ROOT).as_posix(), machine_id='id_' + machine,
                             label=int(condition == 'anomaly'), original_split=part))
    normal = [row for row in rows if row['original_split'] == 'train']
    train, val = train_test_split(normal, test_size=.1, random_state=config['seed'],
                                 stratify=[row['machine_id'] for row in normal])
    parts = dict(train=sorted(train, key=lambda row: row['file']), validation=sorted(val, key=lambda row: row['file']),
                 test=[row for row in rows if row['original_split'] == 'test'])
    names = sorted(row['file'] for row in rows)
    manifest = dict(seed=config['seed'], source='https://zenodo.org/records/3678171',
        license='CC-BY-NC-SA-4.0', archive_checksum=entry['checksum'],
        file_list_sha256=hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest(),
        test_audio_opened=False, splits=parts)
    output = ROOT / 'data/pump_manifest.json'
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError('Existing pump manifest differs')
    output.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: len(value) for key, value in parts.items()}), flush=True)


if __name__ == '__main__':
    main()
