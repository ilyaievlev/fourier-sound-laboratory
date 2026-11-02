"""Тесты дискретизации (шаг 1) и подготовки записи."""
import numpy as np
import pytest

from analyzer.services import audio, sampling
from analyzer.services import fourier as F


def tone(f, sr, dur=1.0, A=0.5):
    t = np.arange(int(dur * sr)) / sr
    return A * np.sin(2 * np.pi * f * t)


def peak_freq(x, sr):
    s = F.amplitude_spectrum(x, sr, "hann")
    return s.freqs[np.argmax(s.amplitude)], s.amplitude.max()


def test_alias_without_filter():
    """5 кГц при fs = 8 кГц без фильтра превращаются в |5 − 8| = 3 кГц."""
    x = tone(5000, 48000)
    y = sampling.sample_without_filter(x, 48000, 8000)
    f, A = peak_freq(y, 8000)
    assert f == pytest.approx(3000.0)
    assert sampling.alias_frequency(5000, 8000) == 3000
    assert A == pytest.approx(0.5, rel=1e-6)


def test_antialias_filter_suppresses():
    x = tone(5000, 48000) + tone(1000, 48000)
    y = sampling.resample(x, 48000, 8000)
    s = F.amplitude_spectrum(y, 8000, "hann")
    a1000 = s.amplitude[np.argmin(np.abs(s.freqs - 1000))]
    a3000 = s.amplitude[np.argmin(np.abs(s.freqs - 3000))]
    assert a1000 == pytest.approx(0.5, rel=0.02)
    assert a3000 < 0.005


def test_non_integer_ratio_without_filter():
    """44,1 кГц из 48 кГц: интерполяция между отсчётами, тон на месте."""
    x = tone(1000, 48000)
    y = sampling.sample_without_filter(x, 48000, 44100)
    assert y.size == 44100
    f, _ = peak_freq(y, 44100)
    assert f == pytest.approx(1000.0, abs=1.0)


def test_reconstruct_keeps_tone():
    y = tone(700, 8000)
    z = sampling.reconstruct(y, 8000, 48000)
    assert z.size == 48000
    f, A = peak_freq(z[4800:-4800], 48000)
    assert f == pytest.approx(700, abs=2) and A == pytest.approx(0.5, rel=0.02)


def test_prepare_mono_trim_pad():
    sr = 44100
    st = np.stack([tone(440, sr, 12), -tone(440, sr, 12) + 0.2])
    p = audio.prepare(st, sr)
    assert p.trimmed and not p.padded and p.x.size == 10 * sr
    assert np.allclose(p.x, 0.1)  # среднее каналов
    q = audio.prepare(tone(440, 96000, 0.3)[None, :], 96000)
    assert q.sr == 48000 and q.padded and q.x.size == 48000


def test_quality_report():
    sr = 16000
    silent = np.zeros(sr)
    assert audio.quality_report(silent, sr)["is_silent"]
    clipped = np.clip(3 * tone(200, sr), -1, 1)
    r = audio.quality_report(clipped, sr)
    assert r["is_clipped"] and not r["is_silent"]
    rng = np.random.default_rng(0)
    noisy = tone(200, sr) + 0.3 * rng.standard_normal(sr)
    assert audio.quality_report(noisy, sr)["maybe_noisy"]
    burst = np.concatenate([np.zeros(sr // 2), tone(200, sr, 0.5)]) + 1e-4 * rng.standard_normal(sr)
    assert audio.quality_report(burst, sr)["frame_contrast_db"] > 40


def test_decode_roundtrip(tmp_path):
    sr = 22050
    x = tone(440, sr, 1.5)
    p = tmp_path / "in.wav"
    audio.write_wav(p, x, sr)
    data, sr2 = audio.decode(p)
    assert sr2 == sr and data.shape == (1, x.size)
    assert np.max(np.abs(data[0] - x)) < 1e-4


def test_loudest_time():
    sr = 8000
    x = np.zeros(sr * 2)
    x[sr:sr + 400] = 1.0
    assert audio.loudest_time(x, sr) == pytest.approx(1.0, abs=1e-3)
