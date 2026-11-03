"""Shared pump log-mel representation used by training and the application."""
import librosa
import numpy as np


DEFAULT_CONFIG = dict(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512,
    n_mels=128, power=2, window='hann', center=True, pad_mode='constant',
    fmin=0.0, fmax=8000.0, htk=False, norm='slaney', epsilon=1e-10)


def pump_logmel(wave, sample_rate=16000, config=None):
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Expected a finite nonempty mono waveform')
    if sample_rate != cfg['sample_rate']:
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=cfg['sample_rate'])
    power = librosa.feature.melspectrogram(y=wave, sr=cfg['sample_rate'], n_fft=cfg['n_fft'],
        win_length=cfg['win_length'], hop_length=cfg['hop_length'], n_mels=cfg['n_mels'],
        power=cfg['power'], window=cfg['window'], center=cfg['center'], pad_mode=cfg['pad_mode'],
        fmin=cfg['fmin'], fmax=cfg['fmax'], htk=cfg['htk'], norm=cfg['norm'])
    return (10 * np.log10(power + cfg['epsilon'])).astype(np.float32)


def sliding_crops(logmel, frames=64, hop=32):
    logmel = np.asarray(logmel, dtype=np.float32)
    if logmel.shape[1] < frames:
        logmel = np.pad(logmel, ((0, 0), (0, frames - logmel.shape[1])), constant_values=-100.)
    starts = list(range(0, logmel.shape[1] - frames + 1, hop))
    if starts[-1] != logmel.shape[1] - frames:
        starts.append(logmel.shape[1] - frames)
    return np.stack([logmel[:, start:start + frames] for start in starts])
