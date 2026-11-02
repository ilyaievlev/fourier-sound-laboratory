"""Аналитические тесты STFT и мел-шкалы (шаг 5)."""
import librosa
import numpy as np
import pytest

from analyzer.services import stft as ST

SR = 16000


def test_tone_horizontal_line():
    """Тон 1 кГц: в каждом кадре максимум на 1 кГц."""
    t = np.arange(SR) / SR
    r = ST.stft(np.sin(2 * np.pi * 1000 * t), SR, 512, 128)
    peaks = r.freqs[np.argmax(r.power[:-3], axis=1)]
    assert np.all(peaks == pytest.approx(1000.0))


def test_chirp_rising_line():
    """Линейный свип 500→4500 Гц: частота пика в кадре ≈ мгновенной частоте в центре кадра."""
    t = np.arange(2 * SR) / SR
    f0, k = 500.0, 2000.0  # f(t) = f0 + k t
    x = np.sin(2 * np.pi * (f0 * t + k * t ** 2 / 2))
    r = ST.stft(x, SR, 512, 256)
    M = r.X.shape[0] - 2
    fpk = r.freqs[np.argmax(r.power[:M], axis=1)]
    f_inst = f0 + k * r.times[:M]
    assert np.max(np.abs(fpk - f_inst)) <= SR / 512  # в пределах шага сетки
    assert np.all(np.diff(fpk) >= 0)


def test_click_vertical_line():
    """Щелчок: в его кадре спектр плоский, вдали от щелчка энергии нет."""
    x = np.zeros(SR)
    x[8000] = 1.0
    r = ST.stft(x, SR, 256, 128, window="hann")
    P = r.power
    m = int(np.argmax(P.sum(axis=1)))
    assert r.starts[m] <= 8000 < r.starts[m] + 256
    assert ST.spectral_flatness(P[m]) > 0.99
    assert P[:m - 2].sum() == 0 and P[m + 3:].sum() == 0


def test_stft_parseval_hann_quarter_hop():
    """Σ_m Σ_k |X|² (полный спектр) = N_fft · Σ_n x² · Σ_m w²[n − mH]; для Ханна при H = N/4 сумма w² = 3/2."""
    rng = np.random.default_rng(0)
    N, H = 512, 128
    x = np.zeros(20 * N)
    x[N:-N] = rng.standard_normal(18 * N)
    r = ST.stft(x, SR, N, H, "hann")
    idx = np.arange(r.X.shape[0])[:, None] * H + np.arange(N)[None, :]
    full = np.abs(np.fft.fft(x[idx] * r.w, axis=1)) ** 2
    assert full.sum() / N == pytest.approx(1.5 * np.sum(x ** 2), rel=1e-10)
    # то же через односторонний спектр: удвоить всё, кроме k = 0 и k = N/2
    P = r.power.copy()
    P[:, 1:-1] *= 2
    assert P.sum() / N == pytest.approx(1.5 * np.sum(x ** 2), rel=1e-10)


def test_matches_librosa_magnitude():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR)
    r = ST.stft(x, SR, 1024, 256, "hann")
    L = np.abs(librosa.stft(x, n_fft=1024, hop_length=256, window="hann", center=False)).T
    m = min(L.shape[0], r.X.shape[0])
    assert np.allclose(np.abs(r.X[:m]), L[:m], atol=1e-9)


@pytest.mark.parametrize("norm", [None, "slaney"])
def test_mel_filterbank_matches_librosa(norm):
    fb = ST.mel_filterbank(SR, 512, 40, 0, 8000, norm=norm)
    ref = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40, fmin=0, fmax=8000, htk=True, norm=norm)
    assert np.allclose(fb, ref, atol=1e-6)


def test_mel_scale_values():
    assert ST.hz_to_mel(0) == 0
    assert ST.hz_to_mel(1000) == pytest.approx(1000.0, abs=0.05)
    f = np.array([20, 440, 7999.0])
    assert ST.mel_to_hz(ST.hz_to_mel(f)) == pytest.approx(f)
    c = ST.mel_centers(SR, 64)
    assert np.all(np.diff(np.diff(c)) > 0)  # по герцам полосы расширяются


def test_gabor_gaussian_attains_bound():
    s = ST.window_spreads(ST.gaussian_window(1024, 0.08), SR)
    assert s["product"] == pytest.approx(ST.GABOR_BOUND, rel=1e-3)


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "rect"])
def test_gabor_bound_holds(name):
    s = ST.window_spreads(ST.make_window(name, 1024), SR)
    assert s["product"] >= ST.GABOR_BOUND * (1 - 1e-3)


def test_longer_window_trades_time_for_frequency():
    a = ST.window_spreads(ST.make_window("hann", 256), SR)
    b = ST.window_spreads(ST.make_window("hann", 1024), SR)
    assert b["sigma_t"] == pytest.approx(4 * a["sigma_t"], rel=0.01)
    assert b["sigma_f"] == pytest.approx(a["sigma_f"] / 4, rel=0.01)
    assert b["product"] == pytest.approx(a["product"], rel=0.01)


def test_frequency_resolution_depends_on_window():
    """Тоны 1000 и 1060 Гц различимы окном 4096 (Δf ≈ 3,9 Гц) и сливаются при окне 256 (Δf = 62,5 Гц)."""
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1060 * t)

    def n_peaks(N):
        P = ST.stft(x, SR, N, N // 2, "hann").power.mean(axis=0)
        db = 10 * np.log10(P / P.max())
        from scipy.signal import find_peaks
        idx, _ = find_peaks(db, height=-6, prominence=3)
        return idx.size

    assert n_peaks(4096) == 2
    assert n_peaks(256) == 1
