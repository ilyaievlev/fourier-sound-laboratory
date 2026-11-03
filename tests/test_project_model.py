import json
from types import SimpleNamespace

import numpy as np
import pytest

from analyzer.services import models


def test_project_inference_uses_full_waveform_contract_and_validation_threshold(tmp_path, monkeypatch):
    import torch
    from training.common.project_audio import prepare
    labels = ['speech', 'whistling', 'clapping', 'siren', 'music', 'knock', 'cough', 'dog', 'footsteps', 'other']
    (tmp_path / 'complete.json').write_text('{}')
    (tmp_path / 'labels_ru.json').write_text(json.dumps({key: key for key in labels}))
    (tmp_path / 'threshold.json').write_text(json.dumps({'max_prob_threshold': .2}))
    monkeypatch.setattr(models, 'PROJECT_DIR', tmp_path)
    captured = []
    def processor(wave, **kwargs):
        captured.append(wave)
        return {'input_values': torch.zeros(1, 1024, 128)}
    def model(**kwargs):
        return SimpleNamespace(logits=torch.zeros(1, 10))
    model.config = SimpleNamespace(id2label=dict(enumerate(labels)))
    monkeypatch.setattr(models, '_load', lambda key: dict(proc=processor, model=model, device='cpu'))
    sr = 44100
    wave = (.2 * np.sin(2 * np.pi * 1200 * np.arange(sr // 4) / sr)).astype('float32')
    result = models.project_infer(wave, sr)
    np.testing.assert_array_equal(captured[0], prepare(wave, sr))
    assert captured[0].shape == (160000,)
    assert sum(row['p'] for row in result['labels']) == pytest.approx(1.)
    assert result['uncertain'] and result['verdict'] == 'другое'
    assert result['entropy'] == pytest.approx(np.log(10))
