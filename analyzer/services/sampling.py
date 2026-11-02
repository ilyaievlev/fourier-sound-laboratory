"""Дискретизация и восстановление сигнала (шаг 1). Без Django."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def _ratio(fs_out: float, fs_in: float) -> Fraction:
    return Fraction(int(round(fs_out)), int(round(fs_in))).limit_denominator(1000)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Ресемплинг с антиалиасинговым фильтром (полифазный КИХ, окно Кайзера).

    Перед прореживанием подавляются частоты выше min(fs_in, fs_out)/2.
    """
    if int(round(fs_in)) == int(round(fs_out)):
        return np.asarray(x, dtype=float).copy()
    r = _ratio(fs_out, fs_in)
    return resample_poly(np.asarray(x, dtype=float), r.numerator, r.denominator)


def sample_without_filter(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """x[n] = f(nΔt), Δt = 1/fs_out, без фильтра: частоты выше fs_out/2 «заворачиваются».

    Если fs_in/fs_out целое — это точное прореживание x[::M].
    Иначе значение в момент nΔt берётся линейной интерполяцией между
    соседними отсчётами исходной записи (своё приближение f(t) между отсчётами).
    """
    x = np.asarray(x, dtype=float)
    ratio = fs_in / fs_out
    if abs(ratio - round(ratio)) < 1e-9:
        return x[:: int(round(ratio))].copy()
    duration = x.size / fs_in
    t_new = np.arange(int(np.floor(duration * fs_out))) / fs_out
    t_old = np.arange(x.size) / fs_in
    return np.interp(t_new, t_old, x)


def reconstruct(y: np.ndarray, fs: float, fs_play: float) -> np.ndarray:
    """Восстановление непрерывного сигнала по отсчётам для прослушивания.

    Приближение идеального восстановления (формула Котельникова: сумма sinc)
    полифазным фильтром нижних частот с частотой среза fs/2.
    """
    return resample(y, fs, fs_play)


def alias_frequency(f: float, fs: float) -> float:
    """Частота, в которую переходит синусоида f после дискретизации с fs:
    |f − fs·round(f/fs)| ∈ [0, fs/2]."""
    return abs(f - fs * round(f / fs))
