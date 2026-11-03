"""Local persisted pump models, metrics and threshold exploration. No Django."""
from pathlib import Path
import json
import threading

import numpy as np

from training.common.features import pump_logmel, sliding_crops

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/anomaly_pump'
_lock = threading.RLock()
_cache = {}


def status():
    ready = (ARTIFACTS / 'deployment.json').is_file()
    return dict(available=ready, machine_type='pump', id_list=['id_00', 'id_02', 'id_04', 'id_06'],
        note='Учебный детектор для насосов DCASE 2020 и тех же ID; к произвольному звуку результат неприменим.',
        deployment=json.loads((ARTIFACTS / 'deployment.json').read_text()) if ready else None)


def evaluate_threshold(approach, machine_id, threshold=None):
    import csv
    from sklearn.metrics import confusion_matrix
    if approach not in ['id_cnn', 'lof'] or machine_id not in status()['id_list']:
        raise ValueError('Неизвестный подход или ID насоса')
    directory = ARTIFACTS / 'lof' if approach == 'lof' else ARTIFACTS
    metrics = json.loads((directory / 'metrics.json').read_text())
    roc = json.loads((directory / 'roc.json').read_text())[machine_id]
    chosen = json.loads((directory / 'threshold.json').read_text())['by_id'][machine_id]
    threshold = chosen if threshold is None else float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('Порог должен быть конечным числом')
    with (directory / 'test_scores.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['machine_id'] == machine_id]
    labels = [int(row['label']) for row in rows]
    scores = [float(row['score']) for row in rows]
    matrix = confusion_matrix(labels, np.array(scores) > threshold, labels=[0, 1]).tolist()
    return dict(approach=approach, machine_id=machine_id, auc=metrics[machine_id]['auc'],
        pauc=metrics[machine_id]['pauc'], roc=roc, threshold=threshold, validation_threshold=chosen,
        confusion=matrix, score_range=[min(scores), max(scores)], test_clips=len(rows),
        threshold_note='Исходный порог выбран по normal validation. Ползунок — исследование тестовых результатов; модель и сохранённый порог не переобучаются.')


def infer(wave, sr, machine_id, applicable=False, threshold=None, approach=None):
    import torch
    from . import models as MD
    if not applicable:
        return dict(applicable=False, message=status()['note'])
    st = status()
    if not st['available']:
        raise FileNotFoundError('Артефакты детектора насоса ещё не готовы')
    if machine_id not in st['id_list']:
        raise ValueError('Выберите ID из DCASE 2020 pump')
    approach = approach or st['deployment']['approach']
    if approach not in ['lof', 'id_cnn']:
        raise ValueError('Неизвестный подход')
    explanation = {}
    with _lock, torch.inference_mode():
        if approach == 'lof':
            import joblib
            # Only locally trained, user-owned artifact files are loaded.
            key = ('lof', machine_id)
            if key not in _cache:
                _cache[key] = joblib.load(ARTIFACTS / 'lof' / f'{machine_id}.joblib')
            entry = MD._load('ast')
            from .sampling import resample
            values = entry['proc'](resample(np.asarray(wave, dtype=np.float32), sr, 16000)[:160000],
                                   sampling_rate=16000, return_tensors='pt')['input_values'].to(entry['device'])
            vector = entry['model'].audio_spectrogram_transformer(values).last_hidden_state.mean(1).float().cpu().numpy()
            score = float(-_cache[key].score_samples(vector)[0])
            distances, _ = _cache[key].kneighbors(vector)
            explanation = dict(embedding_dimensions=int(vector.shape[1]), neighbors=int(distances.shape[1]),
                nearest_normal_distance=float(distances[0, 0]), mean_neighbor_distance=float(distances.mean()),
                distance_metric=_cache[key].effective_metric_)
            directory = ARTIFACTS / 'lof'
        else:
            from safetensors.torch import load_file
            from training.common.anomaly_model import from_config, true_id_score
            if 'id_cnn' not in _cache:
                model = from_config(json.loads((ARTIFACTS / 'model_config.json').read_text()))
                model.load_state_dict(load_file(str(ARTIFACTS / 'model.safetensors')))
                _cache['id_cnn'] = model.to(MD.pick_device()).eval()
            model = _cache['id_cnn']
            model_config = json.loads((ARTIFACTS / 'model_config.json').read_text())
            cfg = json.loads((ARTIFACTS / 'feature_config.json').read_text())
            mel = pump_logmel(np.asarray(wave)[:int(10 * sr)], sr, cfg)
            crops = torch.from_numpy(sliding_crops(mel, model_config['crop_frames'], model_config['crop_hop_frames'])).to(next(model.parameters()).device)
            score = float(true_id_score(model(crops), st['id_list'].index(machine_id)))
            explanation = dict(crops=int(len(crops)), crop_frames=model_config['crop_frames'],
                               crop_hop_frames=model_config['crop_hop_frames'])
            directory = ARTIFACTS
    chosen = json.loads((directory / 'threshold.json').read_text())['by_id'][machine_id]
    threshold = chosen if threshold is None else float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('Порог должен быть конечным числом')
    return dict(applicable=True, approach=approach, machine_id=machine_id, score=score,
        threshold=threshold, anomaly=score > threshold, note=st['note'], explanation=explanation,
        verdict='Есть отличие от нормальных образцов; требуется проверка.' if score > threshold else 'Близко к нормальным образцам этого ID.')
