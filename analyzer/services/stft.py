"""Кратковременное преобразование Фурье, спектрограммы и мел-шкала. Без Django.

    X(m, k) = Σ_{n=0}^{N-1} x[n + mH] w[n] e^{-2πikn/N_fft}     (фаза от начала окна)
    S(m, k) = |X(m, k)|²,   S_dB = 10·log10(S + ε)
    m(f)    = 2595·log10(1 + f/700)                               (мел-шкала, HTK)

В SPEC формула записана с фазой от начала записи: Σ_n x[n] w[n − mH] e^{−2πikn/N}.
Подстановка n → n + mH даёт множитель e^{−2πik·mH/N}, по модулю равный 1,
поэтому |X|² и спектрограмма у обеих записей одинаковые.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

WINDOWS = ("hann", "hamming", "blackman", "rect")
EPS = 1e-10
#: Граница принципа неопределённости Габора для σ_t·σ_f (σ — среднеквадратичные ширины по |w|² и |W|²).
GABOR_BOUND = 1 / (4 * np.pi)


def make_window(name: str, N: int) -> np.ndarray:
    if name == "rect":
        return np.ones(N)
    if name not in WINDOWS:
        raise ValueError(f"неизвестное окно: {name}")
    return get_window(name, N, fftbins=True)


@dataclass
class STFT:
    X: np.ndarray        # (кадры M, частоты K) комплексные
    times: np.ndarray    # центр кадра, с
    starts: np.ndarray   # первый отсчёт кадра
    freqs: np.ndarray    # Гц
    sr: float
    n_win: int
    hop: int
    n_fft: int
    window: str
    w: np.ndarray

    @property
    def power(self) -> np.ndarray:
        return np.abs(self.X) ** 2


def n_frames(n_samples: int, n_win: int, hop: int) -> int:
    """Число кадров, покрывающих весь сигнал (последний дополняется нулями)."""
    if n_samples <= n_win:
        return 1
    return 1 + int(np.ceil((n_samples - n_win) / hop))


def stft(x: np.ndarray, sr: float, n_win: int, hop: int, window: str = "hann",
         n_fft: int | None = None) -> STFT:
    x = np.asarray(x, dtype=float)
    n_fft = n_fft or n_win
    if n_fft < n_win:
        raise ValueError("n_fft должно быть не меньше длины окна")
    if not (1 <= hop):
        raise ValueError("шаг должен быть положительным")
    M = n_frames(x.size, n_win, hop)
    need = (M - 1) * hop + n_win
    xp = np.pad(x, (0, max(0, need - x.size)))
    idx = np.arange(M)[:, None] * hop + np.arange(n_win)[None, :]
    w = make_window(window, n_win)
    frames = xp[idx] * w[None, :]
    X = np.fft.rfft(frames, n=n_fft, axis=1)
    starts = np.arange(M) * hop
    return STFT(X, (starts + n_win / 2) / sr, starts, np.fft.rfftfreq(n_fft, 1 / sr), sr, n_win, hop, n_fft, window, w)


def to_db(S: np.ndarray, eps: float = EPS) -> np.ndarray:
    """10·log10(S + ε)."""
    return 10 * np.log10(np.asarray(S) + eps)


# --- мел-шкала -------------------------------------------------------------

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr: float, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None,
                   norm: str | None = None) -> np.ndarray:
    """Треугольные фильтры (n_mels × (n_fft/2+1)), центры равномерно по шкале m(f).

    Фильтр j поднимается линейно от f_{j} до f_{j+1} и опускается до f_{j+2},
    где f_0..f_{n_mels+1} — точки, равноотстоящие по мелам. norm='slaney' делит
    фильтр на половину ширины (f_{j+2} − f_j)/2, чтобы площадь была одинаковой.
    """
    fmax = sr / 2 if fmax is None else fmax
    fft_f = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, fft_f.size))
    for j in range(n_mels):
        lo, c, hi = edges[j], edges[j + 1], edges[j + 2]
        up = (fft_f - lo) / (c - lo)
        down = (hi - fft_f) / (hi - c)
        fb[j] = np.maximum(0.0, np.minimum(up, down))
    if norm == "slaney":
        fb *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    elif norm is not None:
        raise ValueError(norm)
    return fb


def mel_centers(sr: float, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = sr / 2 if fmax is None else fmax
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))[1:-1]


# --- разрешение и принцип неопределённости ---------------------------------

def window_spreads(w: np.ndarray, sr: float, oversample: int = 64) -> dict:
    """Среднеквадратичные ширины окна во времени и по частоте.

    σ_t² = Σ (t − t̄)² w² / Σ w²,  σ_f² = ∫ f² |W(f)|² df / ∫ |W(f)|² df,
    |W|² считается по ДПФ с дополнением нулями (oversample·N точек) на [−fs/2, fs/2).
    Для непрерывных сигналов σ_t·σ_f ≥ 1/(4π), равенство — у гауссова окна.
    """
    w = np.asarray(w, dtype=float)
    N = w.size
    t = np.arange(N) / sr
    e = w ** 2
    tm = np.sum(t * e) / e.sum()
    st = float(np.sqrt(np.sum((t - tm) ** 2 * e) / e.sum()))
    L = oversample * N
    P = np.abs(np.fft.fftshift(np.fft.fft(w, L))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(L, 1 / sr))
    sf = float(np.sqrt(np.sum(f ** 2 * P) / P.sum()))
    return {"sigma_t": st, "sigma_f": sf, "product": st * sf, "bound": GABOR_BOUND}


def gaussian_window(N: int, rel_sigma: float = 0.125) -> np.ndarray:
    n = np.arange(N) - (N - 1) / 2
    return np.exp(-0.5 * (n / (rel_sigma * N)) ** 2)


def spectral_flatness(p: np.ndarray, eps: float = EPS) -> float:
    """Геометрическое среднее / арифметическое среднее спектра мощности (1 — белый шум, → 0 — тон)."""
    p = np.asarray(p, dtype=float) + eps
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))
