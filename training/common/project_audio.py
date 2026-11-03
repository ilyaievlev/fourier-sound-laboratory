"""Project AST waveform contract shared by dataset training and application inference."""
from fractions import Fraction

import librosa
import numpy as np
from scipy.signal import resample_poly


CONFIG = dict(version=1, mono='arithmetic mean', maximum_input_sample_rate=48000,
    high_rate_resampler='scipy resample_poly Kaiser default; ratio denominator <= 1000',
    maximum_seconds=10, minimum_input_seconds=1, sample_rate=16000,
    model_resampler='librosa soxr_hq', padding='waveform zero padding to 160000 samples before ASTFeatureExtractor',
    dtype='float32')


def prepare(wave, sample_rate):
    """Match app ingest, then resample the first up to 10s and zero-pad the waveform."""
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or sample_rate <= 0:
        raise ValueError('Expected finite nonempty mono or frames-by-channels audio')
    values = values.astype(np.float64)
    if sample_rate > 48000:
        ratio = Fraction(48000, int(sample_rate)).limit_denominator(1000)
        values = resample_poly(values, ratio.numerator, ratio.denominator)
        sample_rate = 48000
    values = values[:10 * sample_rate]
    values = np.pad(values, (0, max(0, sample_rate - len(values))))
    # The app persists its ingested waveform as float32 WAV; emulate that boundary.
    values = values.astype(np.float32)
    values = librosa.resample(values, orig_sr=sample_rate, target_sr=16000, res_type='soxr_hq')
    return np.pad(values[:160000], (0, max(0, 160000 - len(values)))).astype(np.float32)
