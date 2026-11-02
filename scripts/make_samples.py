"""Генерирует синтетические примеры для галереи в samples/ (детерминированно, seed=2026).

Это замена до появления настоящих записей: каждый звук построен по простой
физической модели (описана в manifest.json). Настоящие записи можно положить
в samples/ и добавить в manifest.json с "synthetic": false.

    python scripts/make_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

SR = 48_000
DUR = 4.0
ROOT = Path(__file__).resolve().parent.parent / "samples"
rng = np.random.default_rng(2026)
t = np.arange(int(SR * DUR)) / SR


def norm(x, peak=0.8):
    return peak * x / np.max(np.abs(x))


def phase_of(freq):
    """Фаза по мгновенной частоте: φ(t) = 2π ∫ f dt."""
    return 2 * np.pi * np.cumsum(freq) / SR


def resonator(x, f, bw):
    """Двухполюсный резонатор (форманта) с центром f и полосой bw."""
    r = np.exp(-np.pi * bw / SR)
    a = [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r]
    return lfilter([1 - r], a, x)


def envelope(segments, attack=0.03, release=0.06):
    env = np.zeros_like(t)
    for s, e in segments:
        m = (t >= s) & (t < e)
        tt = t[m] - s
        env[m] = np.minimum(1, tt / attack) * np.minimum(1, (e - t[m]) / release)
    return env


def voice():
    # три гласные «а», «о», «и» с паузами; источник — импульсы голосовой щели
    vowels = [((0.25, 1.15), (730, 1090, 2440, 3400)),
              ((1.45, 2.35), (570, 840, 2410, 3400)),
              ((2.65, 3.6), (300, 2290, 3010, 3700))]
    f0 = 130 + 15 * np.sin(2 * np.pi * 0.4 * t) + 1.5 * rng.standard_normal(t.size).cumsum() / 300
    ph = np.mod(phase_of(f0) / (2 * np.pi), 1.0)
    src = np.where(ph < 0.6, np.sin(np.pi * ph / 0.6) ** 2, 0.0)  # модель Розенберга
    src = np.diff(src, prepend=0.0)
    out = np.zeros_like(t)
    for (s, e), formants in vowels:
        y = sum(resonator(src, f, bw) * g for f, bw, g in zip(formants, (90, 110, 170, 250), (1, 0.7, 0.35, 0.2)))
        out += y * envelope([(s, e)])
    out += 3e-4 * rng.standard_normal(t.size)
    return norm(out)


def whistle():
    f = 1850 + 250 * np.sin(2 * np.pi * 0.35 * t) + 25 * np.sin(2 * np.pi * 5.5 * t)
    x = np.sin(phase_of(f)) + 0.04 * np.sin(2 * phase_of(f))
    breath = sosfilt(butter(4, [1500, 2500], "bandpass", fs=SR, output="sos"), rng.standard_normal(t.size))
    env = envelope([(0.2, 1.7), (2.1, 3.7)], 0.05, 0.1)
    return norm(env * (x + 0.08 * breath) + 2e-4 * rng.standard_normal(t.size))


def clap():
    x = np.zeros_like(t)
    sos = butter(4, [900, 5000], "bandpass", fs=SR, output="sos")
    for onset in (0.3, 1.1, 1.9, 2.4, 2.9):
        n0 = int(onset * SR)
        n = np.arange(int(0.25 * SR))
        burst = rng.standard_normal(n.size) * (np.exp(-n / (0.006 * SR)) + 0.06 * np.exp(-n / (0.07 * SR)))
        x[n0:n0 + n.size] += sosfilt(sos, burst)
    return norm(x + 1e-4 * rng.standard_normal(t.size))


def siren():
    f = 1050 + 400 * np.sin(2 * np.pi * 0.5 * t)
    ph = phase_of(f)
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    return norm(x * envelope([(0.05, 3.95)], 0.1, 0.2) + 0.01 * rng.standard_normal(t.size))


def music():
    notes = [220.0, 261.63, 329.63, 440.0, 329.63, 261.63, 220.0, 196.0]
    x = np.zeros_like(t)
    step = DUR / len(notes)
    for i, f in enumerate(notes):
        s = i * step
        m = t >= s
        tt = t[m] - s
        tone = sum((0.6 ** (h - 1)) * np.exp(-tt * (1.5 + 0.8 * h)) * np.sin(2 * np.pi * h * f * tt)
                   for h in range(1, 9))
        x[m] += tone * np.minimum(1, tt / 0.004)
    return norm(x + 1e-4 * rng.standard_normal(t.size))


def pump(anomaly: bool):
    f_rot = 24.5  # 1470 об/мин
    x = np.zeros_like(t)
    for h in range(1, 9):
        amp = 0.5 / h
        if anomaly and h == 2:
            amp *= 3.0  # рост 2-й гармоники вращения (как при расцентровке)
        x += amp * np.sin(2 * np.pi * h * f_rot * t + rng.uniform(0, 2 * np.pi))
    x += 0.8 * np.sin(2 * np.pi * 6 * f_rot * t)          # лопаточная частота
    x += 0.3 * np.sin(2 * np.pi * 100 * t)                  # электромагнитный гул 2·50 Гц
    flow = sosfilt(butter(2, 1500, "lowpass", fs=SR, output="sos"), rng.standard_normal(t.size))
    x += 0.6 * flow * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))
    if anomaly:
        # удары с частотой дефекта подшипника ≈ 4,2·f_rot, возбуждающие резонанс 3,1 кГц
        f_def = 4.2 * f_rot
        imp = np.zeros_like(t)
        for k in range(int(DUR * f_def)):
            n0 = int((k / f_def + 0.0005 * rng.standard_normal()) * SR)
            if 0 <= n0 < t.size:
                imp[n0] = rng.uniform(0.7, 1.3)
        n = np.arange(int(0.01 * SR))
        ring = np.exp(-n / (0.0015 * SR)) * np.sin(2 * np.pi * 3100 * n / SR)
        x += 2.5 * np.convolve(imp, ring)[: t.size]
    return norm(x, 0.6)


SAMPLES = [
    ("voice", "Голос", voice, "Синтез гласных «а», «о», «и»: импульсы голосовой щели (модель Розенберга, f₀ ≈ 115–145 Гц) через резонаторы формант."),
    ("whistle", "Свист", whistle, "Почти чистый тон 1,6–2,1 кГц с вибрато 5,5 Гц и шумом дыхания."),
    ("clap", "Хлопки", clap, "Пять коротких шумовых импульсов в полосе 0,9–5 кГц с затуханием ≈ 6 мс."),
    ("siren", "Сирена", siren, "Частотная модуляция 650–1450 Гц с периодом 2 с, с 2-й и 3-й гармониками."),
    ("music", "Музыка", music, "Арпеджио ля минор: затухающие ноты с 8 гармониками каждая."),
    ("pump_normal", "Насос: норма", lambda: pump(False), "Модель насоса: вращение 24,5 Гц и гармоники, лопаточная частота 147 Гц, гул 100 Гц, шум потока."),
    ("pump_anomaly", "Насос: аномалия", lambda: pump(True), "Та же модель + удары ≈ 103 раза в секунду, возбуждающие резонанс 3,1 кГц, и усиленная 2-я гармоника вращения."),
]


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = []
    for sid, title, fn, desc in SAMPLES:
        x = fn()
        sf.write(ROOT / f"{sid}.wav", x, SR, subtype="PCM_16")
        manifest.append({"id": sid, "title": title, "file": f"{sid}.wav", "description": desc,
                         "synthetic": True})
        print(sid, f"{x.size / SR:.1f} с")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
