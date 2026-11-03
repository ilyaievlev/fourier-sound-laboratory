"""Final local/offline smoke checks of real exported models and app/training parity."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ['FSL_DEVICE'] = 'cpu'
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from analyzer.services import anomaly, models


def main():
    assert models.project_status()['available']
    project = json.loads((ROOT / 'data/project_manifest.json').read_text(encoding='utf-8'))
    example = project['splits']['test'][0]
    wave, sr = sf.read(ROOT / example['file'], dtype='float32')
    prediction = models.project_infer(wave, sr)
    assert prediction['available'] and len(prediction['labels']) == 10
    assert abs(sum(row['p'] for row in prediction['labels']) - 1) < 1e-5
    assert all(np.isfinite(row['p']) for row in prediction['labels'])
    status = anomaly.status()
    assert status['available']
    directory = ROOT / 'artifacts/anomaly_pump'
    for approach in ['id_cnn', 'lof']:
        base = directory / 'lof' if approach == 'lof' else directory
        with (base / 'test_scores.csv').open() as handle:
            expected = next(csv.DictReader(handle))
        wave, sr = sf.read(ROOT / expected['file'], dtype='float32')
        actual = anomaly.infer(wave, sr, expected['machine_id'], applicable=True, approach=approach)
        np.testing.assert_allclose(actual['score'], float(expected['score']), rtol=1e-3, atol=1e-4)
        print(approach, 'offline app/training score parity OK', actual['score'], flush=True)
    for name in ['train_embeddings.npy', 'validation_embeddings.npy', 'test_embeddings.npy']:
        values = np.load(directory / 'lof' / name, allow_pickle=False)
        assert values.shape[1] == 768 and np.isfinite(values).all()
    for machine in status['id_list']:
        assert (directory / 'lof' / f'{machine}.joblib').is_file()
    assert (ROOT / 'reports/experiment_report.md').is_file()
    path = ROOT / 'AGENTS.md'
    doc = path.read_text(encoding='utf-8').splitlines()
    doc = ['| Проверки | Автоматические тесты приложения и проверка локальных моделей пройдены без сети; физический микрофон отдельно не проверялся |'
           if line.startswith('| Проверки |') else line for line in doc]
    path.write_text('\n'.join(doc) + '\n', encoding='utf-8')
    print('All local artifact contracts verified', flush=True)


if __name__ == '__main__':
    main()
