"""Вероятности, энтропия и искажения сигнала для шага 6. Без Django и без torch."""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    """σ(z) = 1 / (1 + e^{−z}), численно устойчиво при больших |z|."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z, tau: float = 1.0):
    """p_i = exp(z_i/τ) / Σ_j exp(z_j/τ); из z вычитается максимум (результат тот же, без переполнения)."""
    z = np.asarray(z, dtype=float) / tau
    e = np.exp(z - z.max())
    return e / e.sum()


def entropy(p) -> float:
    """H(p) = −Σ p_i ln p_i (в натах), 0·ln 0 = 0."""
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def binary_entropy(p) -> np.ndarray:
    """H(p) = −p ln p − (1−p) ln(1−p) для каждой независимой метки."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def add_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Белый гауссов шум с заданным SNR = 10·log10(P_сигнала / P_шума).

    Мощность шума подбирается точно по выборке (не по математическому
    ожиданию), поэтому фактический SNR совпадает с заданным.
    """
    x = np.asarray(x, dtype=float)
    p_sig = float(np.mean(x ** 2))
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(x.size)
    n -= n.mean()
    p_target = p_sig / 10 ** (snr_db / 10)
    n *= np.sqrt(p_target / np.mean(n ** 2)) if p_sig > 0 else 0.0
    y = x + n
    return y, {"p_signal": p_sig, "p_noise": float(np.mean(n ** 2)), "snr_db": snr_db}


def top_k(p: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.asarray(p))[::-1][:k]
