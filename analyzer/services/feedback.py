"""Rule-based feedback tied to measured audio quantities, without Django or models."""
import numpy as np

from .audio import quality_report
from .fourier import amplitude_spectrum


def describe(wave, sr):
    quality = quality_report(wave, sr)
    spectrum = amplitude_spectrum(wave, sr, 'hann')
    power = np.asarray(spectrum.amplitude[1:]) ** 2
    flatness = float(np.exp(np.mean(np.log(power + 1e-20))) / (np.mean(power) + 1e-20))
    peak = float(spectrum.freqs[1 + int(np.argmax(power))]) if power.size else 0.
    observations = []
    if quality['is_silent']:
        observations.append(f"Слишком тихо: RMS {quality['rms_dbfs']:.1f} dBFS. Поднесите источник ближе или увеличьте уровень записи.")
    if quality['is_clipped']:
        observations.append(f"Клиппинг: {100 * quality['clip_fraction']:.2f}% отсчётов на пределе. Уменьшите усиление.")
    if not quality['is_silent']:
        observations.append(f"Спектр {'с выраженными тонами' if flatness < .1 else 'с распределённой по частотам энергией'}: сильнейшая частота {peak:.0f} Гц, спектральная плоскость {flatness:.3f}.")
    contrast = quality['frame_contrast_db']
    if contrast is not None:
        observations.append(f"Контраст кадров {contrast:.1f} дБ; он не является измерением SNR без известного фонового участка.")
    return dict(quality=quality, spectral_flatness=flatness, peak_hz=peak, observations=observations)
