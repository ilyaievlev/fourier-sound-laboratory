import csv
import json

import numpy as np
import pytest

from analyzer.services import anomaly
from analyzer.services.feedback import describe


def test_arbitrary_microphone_sound_is_not_diagnosed():
    result = anomaly.infer(np.zeros(16000), 16000, 'id_00', applicable=False)
    assert result['applicable'] is False
    assert 'score' not in result and 'аномал' not in result.get('verdict', '')


def test_slider_changes_confusion_without_changing_validation_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(anomaly, 'ARTIFACTS', tmp_path)
    (tmp_path / 'metrics.json').write_text(json.dumps({'id_00': {'auc': 1., 'pauc': 1.}}))
    (tmp_path / 'roc.json').write_text(json.dumps({'id_00': {'fpr': [0, 0, 1], 'tpr': [0, 1, 1], 'thresholds': [None, .8, .1]}}))
    path = tmp_path / 'threshold.json'
    path.write_text(json.dumps({'by_id': {'id_00': .5}}))
    with (tmp_path / 'test_scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['machine_id', 'label', 'score'])
        writer.writeheader()
        writer.writerows([{'machine_id': 'id_00', 'label': label, 'score': score}
                          for label, score in [(0, .1), (0, .2), (1, .8), (1, .9)]])
    assert anomaly.evaluate_threshold('id_cnn', 'id_00')['confusion'] == [[2, 0], [0, 2]]
    moved = anomaly.evaluate_threshold('id_cnn', 'id_00', .85)
    assert moved['confusion'] == [[2, 0], [1, 1]]
    assert moved['validation_threshold'] == .5
    assert json.loads(path.read_text())['by_id']['id_00'] == .5


def test_feedback_reports_measured_tone_and_silence():
    sr = 16000
    quiet = describe(np.zeros(sr), sr)
    assert quiet['quality']['is_silent']
    tone = describe(.2 * np.sin(2 * np.pi * 1000 * np.arange(sr) / sr), sr)
    assert tone['peak_hz'] == pytest.approx(1000.)
    assert tone['spectral_flatness'] < .1
    assert not tone['quality']['is_silent']


def test_lof_explanation_uses_actual_normal_neighbor_distances(tmp_path, monkeypatch):
    import joblib
    import torch
    from sklearn.neighbors import LocalOutlierFactor
    from types import SimpleNamespace
    from analyzer.services import models
    directory = tmp_path / 'lof'
    directory.mkdir()
    reference = np.random.default_rng(2026).normal(size=(40, 768)).astype('float32')
    vector = np.full((1, 768), .2, dtype='float32')
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(reference)
    joblib.dump(lof, directory / 'id_00.joblib')
    (tmp_path / 'deployment.json').write_text(json.dumps({'approach': 'lof'}))
    (directory / 'threshold.json').write_text(json.dumps({'by_id': {'id_00': 2.}}))
    entry = dict(device='cpu', proc=lambda *a, **k: {'input_values': torch.zeros(1, 2, 128)},
                 model=SimpleNamespace(audio_spectrogram_transformer=lambda values:
                    SimpleNamespace(last_hidden_state=torch.from_numpy(vector[:, None, :]).repeat(1, 2, 1))))
    monkeypatch.setattr(models, '_load', lambda key: entry)
    monkeypatch.setattr(anomaly, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(anomaly, '_cache', {})
    result = anomaly.infer(np.zeros(16000), 16000, 'id_00', applicable=True, approach='lof')
    expected = np.sort(np.linalg.norm(reference - vector, axis=1))[:20]
    assert result['score'] == pytest.approx(-lof.score_samples(vector)[0], rel=1e-6)
    assert result['explanation']['embedding_dimensions'] == 768
    assert result['explanation']['neighbors'] == 20
    assert result['explanation']['nearest_normal_distance'] == pytest.approx(expected[0], rel=1e-6)
    assert result['explanation']['mean_neighbor_distance'] == pytest.approx(expected.mean(), rel=1e-6)


def test_new_http_endpoints_and_applicability_guard(tmp_path, monkeypatch):
    import os
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    import django
    django.setup()
    from django.conf import settings
    from django.test import Client
    monkeypatch.setattr(settings, 'SESSIONS_ROOT', tmp_path / 'sessions')
    monkeypatch.setattr(settings, 'ALLOWED_HOSTS', [*settings.ALLOWED_HOSTS, 'testserver'])
    client = Client()
    recording = client.post('/api/sample', {'id': 'whistle'}).json()
    response = client.get(f'/api/rec/{recording["rid"]}/feedback')
    assert response.status_code == 200
    assert response.json()['peak_hz'] > 1000
    status = client.get('/api/anomaly/status')
    assert status.status_code == 200 and status.json()['id_list'] == ['id_00', 'id_02', 'id_04', 'id_06']
    refused = client.post(f'/api/rec/{recording["rid"]}/anomaly',
                          json.dumps({'machine_id': 'id_00', 'applicable': False}), content_type='application/json')
    assert refused.status_code == 200 and refused.json()['applicable'] is False
