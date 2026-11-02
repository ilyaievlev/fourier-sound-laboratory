"""Аналитические тесты математики (раздел 7 ТЗ). Django не нужен."""
import numpy as np
import pytest

from analyzer.services import fourier as F

FS = 8000.0


# --- ДПФ ----------------------------------------------------------------------

@pytest.mark.parametrize("window", ["rect", "hann"])
def test_sine_peak_frequency_and_amplitude(window):
    """Синус частоты f (ровно на бине) даёт пик на f с амплитудой A."""
    N, A, k0 = 4000, 0.7, 440 * 4000 // 8000  # f = k0·fs/N = 440 Гц
    f = k0 * FS / N
    n = np.arange(N)
    x = A * np.sin(2 * np.pi * f * n / FS + 0.3)
    s = F.amplitude_spectrum(x, FS, window)
    k = int(np.argmax(s.amplitude))
    assert s.freqs[k] == pytest.approx(440.0)
    assert s.amplitude[k] == pytest.approx(A, rel=1e-9)


def test_two_sines_two_peaks():
    N = 8000
    n = np.arange(N)
    x = 1.0 * np.sin(2 * np.pi * 300 * n / FS) + 0.25 * np.sin(2 * np.pi * 1250 * n / FS)
    s = F.amplitude_spectrum(x, FS, "hann")
    peaks = F.find_spectral_peaks(s)
    assert [p["f"] for p in peaks] == pytest.approx([300.0, 1250.0])
    assert [p["amp"] for p in peaks] == pytest.approx([1.0, 0.25], rel=1e-9)


def test_off_bin_sine_leaks():
    """Между бинами энергия растекается: пик ниже A, но Парсеваль всё равно верен."""
    N = 1000
    n = np.arange(N)
    x = np.sin(2 * np.pi * 100.5 * (FS / N) * n / FS)
    s = F.amplitude_spectrum(x, FS, "rect")
    assert s.amplitude.max() < 0.7
    assert F.parseval(x)["rel_error"] < 1e-12


def test_inverse_dft_restores_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    assert np.allclose(np.fft.ifft(np.fft.fft(x)).real, x, atol=1e-12)
    x_small = rng.standard_normal(256)
    assert np.allclose(F.idft_naive(F.dft_naive(x_small)).real, x_small, atol=1e-10)


def test_parseval_random_signal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3001)  # нечётная длина тоже
    p = F.parseval(x)
    assert p["rel_error"] < 1e-12
    assert p["time"] == pytest.approx(np.sum(x ** 2))


@pytest.mark.parametrize("N", [1, 2, 7, 64, 255, 512])
def test_naive_dft_matches_numpy(N):
    rng = np.random.default_rng(N)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    X = F.dft_naive(x)
    assert np.max(np.abs(X - np.fft.fft(x))) < 1e-9 * max(1, N)


def test_real_signal_spectrum_symmetry():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(128)
    X = np.fft.fft(x)
    k = np.arange(1, 128)
    assert np.allclose(X[128 - k], np.conj(X[k]))


def test_spectral_edit_remove_and_keep():
    N = 8000
    n = np.arange(N)
    a = np.sin(2 * np.pi * 300 * n / FS)
    b = 0.5 * np.sin(2 * np.pi * 2000 * n / FS)
    y, kept = F.apply_spectral_edit(a + b, FS, [{"type": "remove_band", "f_lo": 1900, "f_hi": 2100}])
    assert np.allclose(y, a, atol=1e-10)
    assert kept == pytest.approx(1 / 1.25, rel=1e-9)
    y2, _ = F.apply_spectral_edit(a + b, FS, [{"type": "keep_band", "f_lo": 1990, "f_hi": 2010}])
    assert np.allclose(y2, b, atol=1e-10)
    y3, _ = F.apply_spectral_edit(a + b, FS, [{"type": "lowpass", "f_c": 1000}])
    assert np.allclose(y3, a, atol=1e-10)
    y4, _ = F.apply_spectral_edit(a + b, FS, [{"type": "highpass", "f_c": 1000}])
    assert np.allclose(y4, b, atol=1e-10)


def test_energy_below():
    N = 8000
    n = np.arange(N)
    x = np.sin(2 * np.pi * 300 * n / FS) + np.sin(2 * np.pi * 1000 * n / FS)
    assert F.energy_below(x, FS, 0.99) == pytest.approx(1000.0)
    assert F.energy_below(x, FS, 0.4) == pytest.approx(300.0)


def test_energy_fraction_above():
    N = 8000
    n = np.arange(N)
    x = np.sin(2 * np.pi * 300 * n / FS) + np.sin(2 * np.pi * 1000 * n / FS)
    assert F.energy_fraction_above(x, FS, 500) == pytest.approx(0.5)
    assert F.energy_fraction_above(x, FS, 2000) == pytest.approx(0.0, abs=1e-20)


# --- Ряд Фурье ------------------------------------------------------------------

def test_square_wave_coefficients_fft():
    """b_n = 4/(πn) для нечётных n, остальные 0 (по БПФ на сетке периода)."""
    T, M, n_max = 1.0, 4096, 50
    t = np.arange(M) * T / M
    a, b = F.coeffs_fft(F.square_wave(t, T), n_max)
    a_ex, b_ex = F.analytic_coeffs("square", n_max)
    assert np.max(np.abs(a - a_ex)) < 1e-12
    assert np.max(np.abs(b - b_ex)) < 1e-4
    assert np.all(np.abs(b[2::2]) < 1e-12)


def test_square_wave_coefficients_trapezoid_converge():
    """Метод трапеций сходится к аналитическим коэффициентам при росте M."""
    T, n_max = 1.0, 15
    _, b_ex = F.analytic_coeffs("square", n_max)
    errs = []
    for M in (256, 1024, 4096):
        t = np.linspace(0, T, M + 1)
        _, b = F.coeffs_trapezoid(F.square_wave(t, T), T, n_max)
        errs.append(np.max(np.abs(b - b_ex)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-4


@pytest.mark.parametrize("kind", ["saw", "abs"])
def test_other_analytic_coefficients(kind):
    T, M, n_max = 2.0, 8192, 40
    t = np.arange(M) * T / M
    a, b = F.coeffs_fft(F.ANALYTIC_WAVES[kind](t, T), n_max)
    a_ex, b_ex = F.analytic_coeffs(kind, n_max)
    assert np.max(np.abs(a - a_ex)) < 1e-4
    assert np.max(np.abs(b - b_ex)) < 1e-4


def test_trapezoid_equals_fft_for_periodic_samples():
    """Для периодической функции (f(T) = f(0)) трапеции = прямоугольники = БПФ."""
    T, M = 0.02, 960
    t = np.linspace(0, T, M + 1)
    w = 2 * np.pi / T
    f = 0.3 + np.cos(3 * w * t) - 0.5 * np.sin(7 * w * t)
    a1, b1 = F.coeffs_trapezoid(f, T, 20)
    a2, b2 = F.coeffs_fft(f[:-1], 20)
    assert np.allclose(a1, a2, atol=1e-12) and np.allclose(b1, b2, atol=1e-12)
    assert a1[0] == pytest.approx(0.6) and a1[3] == pytest.approx(1.0) and b1[7] == pytest.approx(-0.5)


def test_gibbs_overshoot_about_9_percent():
    """Выброс ≈ 8,95% скачка и не исчезает при росте N."""
    vals = [F.gibbs_overshoot_fraction(N) for N in (51, 101, 199)]
    for v in vals:
        assert v == pytest.approx(F.GIBBS_FRACTION, abs=1.5e-3)
    from scipy.special import sici
    assert F.GIBBS_FRACTION == pytest.approx(sici(np.pi)[0] / np.pi - 0.5, abs=1e-14)


def test_gibbs_saw_converges_slower():
    """Для пилы max S_N − 1 → 8,95% скачка, но с поправкой O(1/N) из-за наклона f."""
    errs = []
    for N in (50, 200, 800):
        a, b = F.analytic_coeffs("saw", N)
        t = 0.5 - np.linspace(0, 4 / N, 40001)[1:]
        errs.append(F.GIBBS_FRACTION - (F.partial_sum(a, b, 1.0, t, N).max() - 1) / 2)
    assert errs[0] > errs[1] > errs[2] > 0
    assert errs[1] * 200 == pytest.approx(0.5, rel=0.05)  # ≈ 1/(2N)


def test_gibbs_overshoot_narrows():
    """Ширина выброса сужается: положение максимума ~ T/(2N)."""
    T = 1.0
    pos = []
    for N in (21, 81):
        a, b = F.analytic_coeffs("square", N)
        t = np.linspace(0, 0.25, 200001)[1:]
        pos.append(t[np.argmax(F.partial_sum(a, b, T, t, N))])
    assert pos[1] < pos[0] / 3


def test_rmse_nonincreasing_smooth_signal():
    """Для гладкого сигнала ошибка S_N не растёт с N (S_N — ортопроекция)."""
    T, M = 1.0, 2048
    t = np.arange(M) * T / M
    f = np.exp(np.sin(2 * np.pi * t / T))  # гладкая периодическая
    a, b = F.coeffs_fft(f, 60)
    r = F.rmse_curve(f, a, b, T, t)
    assert np.all(np.diff(r) <= 1e-12)
    assert r[-1] < 1e-12


def test_rmse_matches_parseval_form():
    """RMSE²(N) = mean(f²) − (a0²/4 + ½Σ(a_n²+b_n²)) — дискретный Парсеваль."""
    rng = np.random.default_rng(3)
    M, T = 501, 1.0
    t = np.arange(M) * T / M
    f = rng.standard_normal(M)
    a, b = F.coeffs_fft(f, F.max_harmonic(M))
    r = F.rmse_curve(f, a, b, T, t)
    s = F.bessel_sums(a, b) / 2
    assert np.allclose(r ** 2, np.mean(f ** 2) - s, atol=1e-10)
    assert r[-1] < 1e-10  # при N = (M−1)/2 сетка восстановлена точно


def test_bessel_inequality_and_parseval_square():
    """Σ ≤ (2/T)∫f² = 2 для меандра и стремится к 2."""
    a, b = F.analytic_coeffs("square", 2001)
    s = F.bessel_sums(a, b)
    assert np.all(s <= 2.0 + 1e-12)
    assert s[-1] == pytest.approx(2.0, abs=1e-3)


def test_fft_coeffs_reject_too_high_harmonic():
    with pytest.raises(ValueError):
        F.coeffs_fft(np.zeros(100), 50)


def test_dirichlet_midpoint_value():
    """В точке разрыва сумма ряда = (f(t−0)+f(t+0))/2 = 0 для меандра."""
    a, b = F.analytic_coeffs("square", 501)
    assert F.partial_sum(a, b, 1.0, np.array([0.0, 0.5]), 501) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_orthogonality_numeric():
    T, M = 1.0, 4000
    t = np.linspace(0, T, M + 1)
    w = 2 * np.pi / T
    assert np.trapezoid(np.cos(3 * w * t) * np.cos(5 * w * t), t) == pytest.approx(0, abs=1e-12)
    assert np.trapezoid(np.cos(4 * w * t) ** 2, t) == pytest.approx(T / 2)
    assert np.trapezoid(np.sin(4 * w * t) * np.cos(4 * w * t), t) == pytest.approx(0, abs=1e-12)


def test_period_estimate():
    sr = 48000
    t = np.arange(int(0.06 * sr)) / sr
    x = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t + 1)
    assert F.estimate_period_autocorr(x, sr) == pytest.approx(1 / 200, rel=2e-3)
    rng = np.random.default_rng(0)
    assert F.estimate_period_autocorr(rng.standard_normal(t.size), sr) is None
