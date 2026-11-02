"""Чтение, декодирование, нормализация и проверка качества записи. Без Django."""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import sampling

#: Записи с частотой выше этой приводятся к ней (больше для задачи не нужно).
MAX_SR = 48_000
#: Длительность рабочего сигнала: длиннее — обрезаем, короче — дополняем тишиной.
MAX_DURATION_S = 10.0
MIN_DURATION_S = 1.0
#: Частоты моделей (используются на шаге 6).
MODEL_SR = {"ast": 16_000, "clap": 48_000}

# Пороги проверки качества (в интерфейсе показываются вместе с числами).
SILENCE_RMS_DBFS = -50.0
CLIP_LEVEL = 0.999
CLIP_FRACTION = 1e-3
LOW_CONTRAST_DB = 10.0
FRAME_S = 0.02


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(path: Path) -> tuple[np.ndarray, int]:
    """Любой формат (WAV/MP3/OGG/WEBM/…) → (каналы × отсчёты float32, sr) через ffmpeg."""
    path = Path(path)
    # частоту и число каналов узнаём, декодируя во float WAV без ресемплинга
    out = path.with_suffix(".decoded.wav")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
           "-vn", "-c:a", "pcm_f32le", "-f", "wav", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise ValueError("не удалось декодировать файл: " + res.stderr.strip()[-300:])
    try:
        data, sr = sf.read(out, dtype="float32", always_2d=True)
    finally:
        out.unlink(missing_ok=True)
    return data.T, int(sr)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Моно как среднее арифметическое каналов."""
    data = np.atleast_2d(data)
    return data.mean(axis=0).astype(np.float64)


@dataclass
class Prepared:
    x: np.ndarray
    sr: int
    original_sr: int
    original_channels: int
    original_duration: float
    trimmed: bool
    padded: bool


def prepare(data: np.ndarray, sr: int) -> Prepared:
    """Моно, ограничение частоты MAX_SR, обрезка/дополнение по длительности."""
    data = np.atleast_2d(data)
    channels = data.shape[0]
    x = to_mono(data)
    duration = x.size / sr
    work_sr = sr
    if sr > MAX_SR:
        x = sampling.resample(x, sr, MAX_SR)
        work_sr = MAX_SR
    max_n = int(MAX_DURATION_S * work_sr)
    min_n = int(MIN_DURATION_S * work_sr)
    trimmed = x.size > max_n
    padded = x.size < min_n
    if trimmed:
        x = x[:max_n]
    if padded:
        x = np.concatenate([x, np.zeros(min_n - x.size)])
    return Prepared(x, work_sr, sr, channels, duration, trimmed, padded)


def for_model(x: np.ndarray, sr: int, model: str) -> np.ndarray:
    """Копия сигнала с частотой, которую ожидает модель (AST 16 кГц, CLAP 48 кГц)."""
    return sampling.resample(x, sr, MODEL_SR[model])


def dbfs(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-12)))


def quality_report(x: np.ndarray, sr: int) -> dict:
    """Проверки: тишина, клиппинг, контраст кадров, постоянная составляющая.

    Контраст кадров (кадры по 20 мс): 10·log10(средняя энергия 10% самых
    громких кадров / средняя энергия 10% самых тихих). Это оценка SNR только
    если в записи есть паузы, где звучит один фон. Для непрерывного звука
    (насос, сирена) контраст мал и без шума, поэтому флаг называется
    «возможно шумно», а не «низкий SNR».
    """
    x = np.asarray(x, dtype=float)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    clip_frac = float(np.mean(np.abs(x) >= CLIP_LEVEL)) if x.size else 0.0
    flen = max(1, int(FRAME_S * sr))
    nfr = x.size // flen
    contrast = None
    if nfr >= 10:
        e = np.sort(np.mean(x[: nfr * flen].reshape(nfr, flen) ** 2, axis=1))
        k = max(1, nfr // 10)
        quiet, loud = e[:k].mean(), e[-k:].mean()
        contrast = float(10 * np.log10(loud / quiet)) if quiet > 0 else None
    rep = {
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "dc_offset": float(np.mean(x)) if x.size else 0.0,
        "clip_fraction": clip_frac,
        "frame_contrast_db": contrast,
        "is_silent": dbfs(rms) < SILENCE_RMS_DBFS,
        "is_clipped": clip_frac > CLIP_FRACTION,
        "maybe_noisy": contrast is not None and contrast < LOW_CONTRAST_DB,
        "thresholds": {"silence_rms_dbfs": SILENCE_RMS_DBFS, "clip_level": CLIP_LEVEL,
                       "clip_fraction": CLIP_FRACTION, "low_contrast_db": LOW_CONTRAST_DB},
    }
    return rep


def loudest_time(x: np.ndarray, sr: int, win_s: float = 0.05) -> float:
    """Начало самого громкого окна win_s — разумный участок по умолчанию для шагов 2–3."""
    w = max(1, int(win_s * sr))
    if x.size <= w:
        return 0.0
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=float) ** 2)])
    e = c[w:] - c[:-w]
    return float(np.argmax(e) / sr)


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """16-битный PCM WAV; при пиковом уровне > 1 сигнал масштабируется, чтобы не было клиппинга."""
    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    sf.write(str(path), x, int(sr), subtype="PCM_16")


def prepared_info(p: Prepared) -> dict:
    d = asdict(p)
    d.pop("x")
    d["duration"] = p.x.size / p.sr
    return d
