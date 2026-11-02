"""Ряды Фурье, ДПФ и спектральные операции.

Модуль не зависит от Django: только NumPy/SciPy. Все функции проверяются
аналитическими тестами в tests/test_fourier.py.

Обозначения совпадают с интерфейсом:
    период T, ω = 2π/T,
    f(t) ~ a0/2 + Σ_{n≥1} (a_n cos nωt + b_n sin nωt),
    c_n = (1/T) ∫_0^T f(t) e^{-inωt} dt,  a_n = 2 Re c_n,  b_n = -2 Im c_n,
    X[k] = Σ_{n=0}^{N-1} x[n] e^{-2πikn/N}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks as _find_peaks
from scipy.signal import get_window

# ---------------------------------------------------------------------------
# Ряд Фурье на периоде T
# ---------------------------------------------------------------------------

#: Константа Уилбрахама — Гиббса: доля от скачка, на которую частичная сумма
#: перескакивает предел у точки разрыва при N → ∞.  (1/π)·Si(π) − 1/2.
GIBBS_FRACTION = 0.08948987223608362


def coeffs_trapezoid(f_vals: np.ndarray, T: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """a_n, b_n (n = 0..n_max) методом трапеций.

    f_vals — значения f на равномерной сетке t_j = jT/M, j = 0..M
    (M+1 точек, включая оба конца отрезка [0, T]).
    """
    f_vals = np.asarray(f_vals, dtype=float)
    M = f_vals.size - 1
    if M < 1:
        raise ValueError("нужно хотя бы две точки")
    t = np.linspace(0.0, T, M + 1)
    omega = 2 * np.pi / T
    n = np.arange(n_max + 1)[:, None]
    phase = n * omega * t[None, :]
    a = (2.0 / T) * np.trapezoid(f_vals[None, :] * np.cos(phase), t, axis=1)
    b = (2.0 / T) * np.trapezoid(f_vals[None, :] * np.sin(phase), t, axis=1)
    b[0] = 0.0
    return a, b


def complex_coeffs_fft(x: np.ndarray, n_max: int) -> np.ndarray:
    """c_n, n = 0..n_max, через БПФ: c_n ≈ (1/M) Σ_j x[j] e^{-2πinj/M}.

    x — M отсчётов периода на сетке t_j = jT/M, j = 0..M-1 (правый конец не
    включается: для периодической функции он совпадает с t = 0). Это формула
    прямоугольников для интеграла c_n; для периодической функции она совпадает
    с формулой трапеций.
    """
    x = np.asarray(x, dtype=float)
    M = x.size
    if n_max > (M - 1) // 2:
        raise ValueError(f"n_max={n_max} больше (M-1)/2={(M - 1) // 2}: старшие гармоники неотличимы (алиасинг)")
    return np.fft.rfft(x)[: n_max + 1] / M


def coeffs_fft(x: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """a_n, b_n (n = 0..n_max) из c_n: a_n = 2 Re c_n, b_n = −2 Im c_n."""
    c = complex_coeffs_fft(x, n_max)
    a = 2 * c.real
    b = -2 * c.imag
    b[0] = 0.0
    return a, b


def max_harmonic(M: int) -> int:
    """Наибольший номер гармоники, различимый по M отсчётам периода."""
    return (M - 1) // 2


def harmonic(a: np.ndarray, b: np.ndarray, n: int, T: float, t: np.ndarray) -> np.ndarray:
    """Отдельное слагаемое ряда: a_n cos nωt + b_n sin nωt (для n=0 — a0/2)."""
    if n == 0:
        return np.full_like(np.asarray(t, dtype=float), a[0] / 2)
    w = 2 * np.pi / T
    return a[n] * np.cos(n * w * t) + b[n] * np.sin(n * w * t)


def partial_sum(a: np.ndarray, b: np.ndarray, T: float, t: np.ndarray, N: int) -> np.ndarray:
    """S_N(t) = a0/2 + Σ_{n=1}^{N} (a_n cos nωt + b_n sin nωt)."""
    t = np.asarray(t, dtype=float)
    s = np.full_like(t, a[0] / 2)
    w = 2 * np.pi / T
    for n in range(1, N + 1):
        s += a[n] * np.cos(n * w * t) + b[n] * np.sin(n * w * t)
    return s


def rmse_curve(f_vals: np.ndarray, a: np.ndarray, b: np.ndarray, T: float, t: np.ndarray) -> np.ndarray:
    """RMSE(N) = sqrt(mean((f − S_N)²)) на сетке t для N = 0..len(a)-1.

    Считается напрямую (накопление S_N), без формулы Парсеваля, чтобы
    тесты могли сравнить оба способа.
    """
    f_vals = np.asarray(f_vals, dtype=float)
    s = np.full_like(f_vals, a[0] / 2)
    out = [np.sqrt(np.mean((f_vals - s) ** 2))]
    for n in range(1, len(a)):
        s = s + harmonic(a, b, n, T, t)
        out.append(np.sqrt(np.mean((f_vals - s) ** 2)))
    return np.array(out)


def bessel_sums(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Частичные суммы a0²/2 + Σ_{n=1}^{N} (a_n² + b_n²), N = 0..len(a)-1.

    Неравенство Бесселя: каждая не превосходит (2/T)∫_0^T f² dt,
    равенство Парсеваля — предел при N → ∞.
    """
    terms = a ** 2 + b ** 2
    terms[0] = a[0] ** 2 / 2
    return np.cumsum(terms)


# --- Эталонные периодические функции с известными коэффициентами ----------
# Все определены через x = ωt, приведённый в [−π, π), амплитуда 1.
# В точках разрыва задано среднее односторонних пределов (условие Дирихле).

def _x_of_t(t: np.ndarray, T: float) -> np.ndarray:
    x = 2 * np.pi * np.asarray(t, dtype=float) / T
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def square_wave(t: np.ndarray, T: float) -> np.ndarray:
    """Меандр sign(sin ωt): +1 на (0, T/2), −1 на (T/2, T), 0 в разрывах."""
    x = _x_of_t(t, T)
    y = np.sign(x).astype(float)
    y[np.isclose(np.abs(x), np.pi, atol=1e-12) | np.isclose(x, 0.0, atol=1e-12)] = 0.0
    return y


def sawtooth_wave(t: np.ndarray, T: float) -> np.ndarray:
    """Пила x/π, x ∈ (−π, π); в разрыве x = ±π значение 0."""
    x = _x_of_t(t, T)
    y = x / np.pi
    y[np.isclose(np.abs(x), np.pi, atol=1e-12)] = 0.0
    return y


def abs_wave(t: np.ndarray, T: float) -> np.ndarray:
    """|x|/π, x ∈ [−π, π): непрерывная «треугольная» волна."""
    return np.abs(_x_of_t(t, T)) / np.pi


def analytic_coeffs(kind: str, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Точные a_n, b_n для эталонных функций (n = 0..n_max)."""
    n = np.arange(n_max + 1, dtype=float)
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    odd = (np.arange(n_max + 1) % 2 == 1)
    if kind == "square":
        b[odd] = 4 / (np.pi * n[odd])
    elif kind == "saw":
        nz = n[1:]
        b[1:] = 2 * (-1) ** (nz + 1) / (np.pi * nz)
    elif kind == "abs":
        a[0] = 1.0
        a[odd] = -4 / (np.pi ** 2 * n[odd] ** 2)
    else:
        raise ValueError(kind)
    return a, b


ANALYTIC_WAVES = {"square": square_wave, "saw": sawtooth_wave, "abs": abs_wave}


def gibbs_overshoot_fraction(N: int, samples: int = 20001) -> float:
    """Выброс S_N меандра над пределом 1 у разрыва t=0, в долях скачка (=2).

    S_N вычисляется по точным коэффициентам на мелкой сетке (0, T/2).
    """
    T = 1.0
    a, b = analytic_coeffs("square", N)
    # первый максимум S_N находится около t ≈ T/(2(N+1)); берём с запасом
    t = np.linspace(0, T / 2, samples)[1:]
    s = partial_sum(a, b, T, t, N)
    return float((s.max() - 1.0) / 2.0)


def estimate_period_autocorr(x: np.ndarray, sr: float, f_min: float = 50.0, f_max: float = 4000.0,
                             min_corr: float = 0.5) -> float | None:
    """Оценка периода основного тона по нормированной автокорреляции.

    Возвращает период в секундах или None, если максимум автокорреляции
    в диапазоне лагов [1/f_max, 1/f_min] меньше min_corr (нет явной периодичности).
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    if np.allclose(x, 0):
        return None
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    r = np.fft.irfft(np.abs(X) ** 2, nfft)[:n]
    # несмещённая нормировка, чтобы большие лаги не штрафовались
    r = r / (r[0] * (n - np.arange(n)) / n)
    lo = max(1, int(sr / f_max))
    hi = min(n - 1, int(sr / f_min))
    if hi <= lo + 2:
        return None
    seg = r[lo:hi]
    # первый заметный пик, а не глобальный максимум (иначе берём кратный период)
    peaks, _ = _find_peaks(seg)
    if peaks.size == 0:
        return None
    best = peaks[np.argmax(seg[peaks])]
    good = peaks[seg[peaks] >= 0.9 * seg[best]]
    k = good[0]
    if seg[k] < min_corr:
        return None
    # параболическая интерполяция пика
    lag = lo + k
    if 1 <= k < seg.size - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            lag = lag + 0.5 * (y0 - y2) / denom
    return float(lag / sr)


# ---------------------------------------------------------------------------
# ДПФ
# ---------------------------------------------------------------------------

NAIVE_DFT_MAX_N = 4096


def dft_naive(x: np.ndarray) -> np.ndarray:
    """X[k] = Σ_n x[n] e^{−2πikn/N} прямым суммированием, O(N²)."""
    x = np.asarray(x)
    N = x.size
    if N > NAIVE_DFT_MAX_N:
        raise ValueError("наивное ДПФ только для маленьких N")
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)
    return W @ x


def idft_naive(X: np.ndarray) -> np.ndarray:
    """x[n] = (1/N) Σ_k X[k] e^{2πikn/N}."""
    X = np.asarray(X)
    N = X.size
    n = np.arange(N)
    W = np.exp(2j * np.pi * np.outer(n, n) / N)
    return (W @ X) / N


def make_window(name: str, N: int) -> np.ndarray:
    """Окно длины N: 'rect' или 'hann' (периодическое, DFT-even)."""
    if name == "rect":
        return np.ones(N)
    if name == "hann":
        return get_window("hann", N, fftbins=True)
    raise ValueError(f"неизвестное окно: {name}")


@dataclass
class Spectrum:
    freqs: np.ndarray      # f_k = k·fs/N, k = 0..N/2
    X: np.ndarray          # rfft(x·w)
    amplitude: np.ndarray  # оценка амплитуды синусоиды на частоте f_k
    N: int
    fs: float
    window: str
    coherent_gain: float   # Σw / N

    @property
    def df(self) -> float:
        return self.fs / self.N


def amplitude_spectrum(x: np.ndarray, fs: float, window: str = "hann") -> Spectrum:
    """Одностороннее амплитудное ДПФ-спектр.

    Для x[n] = A cos(2π f n/fs + φ) с f = k·fs/N (ровно на бине) даёт
    amplitude[k] = A: |X[k]| = A·Σw/2, поэтому делим на Σw/2.
    Бины k = 0 и k = N/2 не удваиваются.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    w = make_window(window, N)
    X = np.fft.rfft(x * w)
    s = w.sum()
    amp = 2 * np.abs(X) / s
    amp[0] /= 2
    if N % 2 == 0:
        amp[-1] /= 2
    return Spectrum(np.fft.rfftfreq(N, 1 / fs), X, amp, N, fs, window, s / N)


def to_db(amp: np.ndarray, ref: float | None = None, floor_db: float = -120.0) -> np.ndarray:
    """20·log10(amp/ref), ref по умолчанию — максимум."""
    amp = np.asarray(amp, dtype=float)
    ref = ref if ref is not None else (amp.max() if amp.size and amp.max() > 0 else 1.0)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(amp, 1e-300) / ref)
    return np.maximum(db, floor_db)


def find_spectral_peaks(spec: Spectrum, max_peaks: int = 8, min_prominence_db: float = 12.0,
                        min_level_db: float = -50.0, min_distance_hz: float = 20.0) -> list[dict]:
    """Пики амплитудного спектра в дБ: выступ над окрестностью ≥ min_prominence_db,
    уровень ≥ min_level_db относительно максимума, разнесены на min_distance_hz."""
    db = to_db(spec.amplitude)
    dist = max(1, int(round(min_distance_hz / spec.df)))
    idx, props = _find_peaks(db, prominence=min_prominence_db, height=min_level_db, distance=dist)
    order = np.argsort(spec.amplitude[idx])[::-1][:max_peaks]
    out = []
    for i in idx[order]:
        out.append({"k": int(i), "f": float(spec.freqs[i]), "amp": float(spec.amplitude[i]),
                    "db": float(db[i])})
    return sorted(out, key=lambda p: p["f"])


def parseval(x: np.ndarray) -> dict:
    """Дискретное равенство Парсеваля: Σ|x[n]|² = (1/N) Σ|X[k]|²."""
    x = np.asarray(x)
    X = np.fft.fft(x)
    e_t = float(np.sum(np.abs(x) ** 2))
    e_f = float(np.sum(np.abs(X) ** 2) / x.size)
    rel = abs(e_t - e_f) / e_t if e_t > 0 else 0.0
    return {"time": e_t, "freq": e_f, "rel_error": rel}


def energy_below(spec_x: np.ndarray, fs: float, fraction: float = 0.99) -> float:
    """Частота, ниже которой лежит заданная доля энергии сигнала (по |X|²)."""
    x = np.asarray(spec_x, dtype=float)
    p = np.abs(np.fft.rfft(x)) ** 2
    # одностороннее представление: бины 0 < k < N/2 встречаются дважды
    if x.size % 2 == 0:
        p[1:-1] *= 2
    else:
        p[1:] *= 2
    c = np.cumsum(p)
    if c[-1] == 0:
        return 0.0
    k = int(np.searchsorted(c, fraction * c[-1]))
    return float(np.fft.rfftfreq(x.size, 1 / fs)[min(k, p.size - 1)])


def energy_fraction_above(x: np.ndarray, fs: float, f_cut: float) -> float:
    """Доля энергии сигнала на частотах f > f_cut (по одностороннему |X|²)."""
    x = np.asarray(x, dtype=float)
    p = np.abs(np.fft.rfft(x)) ** 2
    if x.size % 2 == 0:
        p[1:-1] *= 2
    else:
        p[1:] *= 2
    tot = p.sum()
    if tot == 0:
        return 0.0
    f = np.fft.rfftfreq(x.size, 1 / fs)
    return float(p[f > f_cut].sum() / tot)


# --- Редактирование спектра и обратное ДПФ ---------------------------------

EDIT_TYPES = ("keep_band", "remove_band", "lowpass", "highpass")


def spectral_edit_mask(freqs: np.ndarray, ops: list[dict]) -> np.ndarray:
    """Маска 0/1 для бинов k = 0..N/2. Операции применяются по порядку.

    keep_band [f_lo, f_hi] — оставить только полосу; remove_band — вырезать;
    lowpass f_c — оставить f ≤ f_c; highpass f_c — оставить f ≥ f_c.
    """
    m = np.ones_like(freqs, dtype=float)
    for op in ops:
        t = op["type"]
        if t == "keep_band":
            m *= ((freqs >= op["f_lo"]) & (freqs <= op["f_hi"]))
        elif t == "remove_band":
            m *= ~((freqs >= op["f_lo"]) & (freqs <= op["f_hi"]))
        elif t == "lowpass":
            m *= freqs <= op["f_c"]
        elif t == "highpass":
            m *= freqs >= op["f_c"]
        else:
            raise ValueError(f"неизвестная операция: {t}")
    return m


def apply_spectral_edit(x: np.ndarray, fs: float, ops: list[dict]) -> tuple[np.ndarray, float]:
    """Обнуляет бины ДПФ всего сигнала (прямоугольное окно) и возвращает
    обратное ДПФ и долю сохранённой энергии.

    Используется rfft/irfft: для вещественного x X[N−k] = conj X[k], поэтому
    обнуление бина k автоматически обнуляет и N−k, а результат вещественный.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    mask = spectral_edit_mask(freqs, ops)
    y = np.fft.irfft(X * mask, n=N)
    e = float(np.sum(x ** 2))
    kept = float(np.sum(y ** 2)) / e if e > 0 else 1.0
    return y, kept
