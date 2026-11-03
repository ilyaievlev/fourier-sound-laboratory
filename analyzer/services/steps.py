"""Сборка данных для шагов интерфейса из чистой математики. Без Django.

Каждая функция возвращает словарь, пригодный для JSON: массивы для графиков
(в base64 float32) и числа, которые подставляются в формулы на странице.
"""
from __future__ import annotations

import base64
import time

import numpy as np

from . import audio, fourier as F, sampling

FS_OPTIONS = [2000, 4000, 8000, 16000, 44100]
N_MAX_UI = 200
ANALYTIC_M = 16384
SOURCES = ("rec", "square", "saw", "abs")


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _f(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# Общее описание записи
# ---------------------------------------------------------------------------

def overview(x: np.ndarray, sr: int, meta: dict) -> dict:
    return {
        "sr": sr,
        "n": int(x.size),
        "duration": x.size / sr,
        "wave": b64(x),
        "quality": audio.quality_report(x, sr),
        "suggested_t0": audio.loudest_time(x, sr),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Шаг 1. Дискретизация
# ---------------------------------------------------------------------------

def fs_options(sr: int) -> list[dict]:
    return [{"fs": f, "available": f <= sr} for f in FS_OPTIONS]


def resample_for_step1(x: np.ndarray, sr: int, fs: int, antialias: bool) -> np.ndarray:
    if fs >= sr:
        return np.asarray(x, dtype=float)
    return sampling.resample(x, sr, fs) if antialias else sampling.sample_without_filter(x, sr, fs)


def step1(x: np.ndarray, sr: int, fs: int, antialias: bool) -> dict:
    fs = int(min(fs, sr))
    y = resample_for_step1(x, sr, fs, antialias)
    f99 = F.energy_below(x, sr, 0.99)
    spec = F.amplitude_spectrum(x, sr, "hann")
    f_peak = float(spec.freqs[int(np.argmax(spec.amplitude[1:])) + 1])
    above = F.energy_fraction_above(x, sr, fs / 2)
    t_loud = audio.loudest_time(x, sr)
    n0 = int(round(t_loud * fs))
    n0 = min(n0, y.size - 1)
    return {
        "fs": fs,
        "sr": sr,
        "antialias": antialias,
        "dt": 1 / fs,
        "nyquist": fs / 2,
        "f99": f99,
        "kotelnikov_ok": fs > 2 * f99,
        "energy_above_nyquist": above,
        "f_peak": f_peak,
        "f_peak_alias": sampling.alias_frequency(f_peak, fs) if f_peak > fs / 2 else None,
        "example": {"n": n0, "t": n0 / fs, "value": float(y[n0])},
        "n_samples": int(y.size),
        "samples": b64(y),
        "options": fs_options(sr),
    }


def step1_audio(x: np.ndarray, sr: int, fs: int, antialias: bool) -> np.ndarray:
    """То, что услышит слушатель: отсчёты при fs, восстановленные обратно до sr."""
    fs = int(min(fs, sr))
    y = resample_for_step1(x, sr, fs, antialias)
    z = sampling.reconstruct(y, fs, sr)
    return z[: x.size]


# ---------------------------------------------------------------------------
# Шаги 2–3. Участок и ряд Фурье
# ---------------------------------------------------------------------------

def _segment(x: np.ndarray, sr: int, t0: float, T_ms: float, src: str):
    """Возвращает (f на сетке из M+1 точек [0, T], T, M, i0, шаг сетки, точные коэфф.)."""
    if src == "rec":
        M = max(8, int(round(T_ms / 1000 * sr)))
        M = min(M, x.size - 1)
        i0 = int(round(t0 * sr))
        i0 = max(0, min(i0, x.size - M - 1))
        seg = np.asarray(x[i0:i0 + M + 1], dtype=float)
        return seg, M / sr, M, i0, None
    if src not in F.ANALYTIC_WAVES:
        raise ValueError(f"неизвестный источник {src}")
    T = T_ms / 1000
    M = ANALYTIC_M
    t = np.arange(M + 1) * T / M
    seg = F.ANALYTIC_WAVES[src](t, T)
    return seg, T, M, None, src


def series(x: np.ndarray, sr: int, t0: float, T_ms: float, N: int, src: str) -> dict:
    seg, T, M, i0, kind = _segment(x, sr, t0, T_ms, src)
    n_cap = min(N_MAX_UI, F.max_harmonic(M))
    N = int(max(1, min(N, n_cap)))
    t = np.arange(M) * T / M
    f = seg[:M]
    a, b = F.coeffs_fft(f, n_cap)
    S = F.partial_sum(a, b, T, t, N)
    err = f - S
    rmse = F.rmse_curve(f, a, b, T, t)
    energy = 2 * float(np.mean(f ** 2))  # (2/T)∫f² dt
    bessel = F.bessel_sums(a, b)
    H = min(N, 6)
    harmonics = [b64(F.harmonic(a, b, n, T, t)) for n in range(1, H + 1)]

    # периодическое продолжение на три периода [−T, 2T)
    t_ext = np.arange(-M, 2 * M) * T / M
    f_ext = np.tile(f, 3)
    S_ext = F.partial_sum(a, b, T, t_ext, N)
    real_ext = None
    if src == "rec":
        lo, hi = i0 - M, i0 + 2 * M
        if lo >= 0 and hi <= x.size:
            real_ext = b64(x[lo:hi])

    res = {
        "src": src, "T": T, "M": M, "omega": 2 * np.pi / T, "N": N, "n_cap": n_cap,
        "dt": T / M,
        "t": b64(t), "f": b64(f), "S": b64(S), "err": b64(err),
        "rmse": b64(rmse), "rmse_N": _f(rmse[N]),
        "harmonics": harmonics,
        "amps": b64(np.hypot(a, b)),
        "coeffs": [{"n": n, "a": float(a[n]), "b": float(b[n])} for n in range(0, min(n_cap, 10) + 1)],
        "energy": energy, "bessel_N": float(bessel[N]), "bessel_cap": float(bessel[-1]),
        "ext": {"t": b64(t_ext), "f": b64(f_ext), "S": b64(S_ext), "real": real_ext},
        "boundary": {"f0": float(f[0]), "fT": float(f[-1]), "jump": float(f[0] - f[-1])},
    }
    if src == "rec":
        res["t0"] = i0 / sr
        res["sr"] = sr
        res["boundary"]["typical_step"] = float(np.median(np.abs(np.diff(f))))
        win = x[i0:i0 + int(0.06 * sr)]
        P = F.estimate_period_autocorr(win, sr)
        res["period"] = None
        if P:
            ks = np.arange(1, 200)
            cand = ks * P
            ok = cand[(cand >= 0.02) & (cand <= 0.05)]
            best = ok[np.argmin(np.abs(ok - T))] if ok.size else cand[np.argmin(np.abs(cand - T))]
            res["period"] = {"P": P, "f0": 1 / P, "suggest_T": float(best), "k": int(round(best / P))}
    else:
        ae, be = F.analytic_coeffs(kind, n_cap)
        res["exact"] = [{"n": n, "a": float(ae[n]), "b": float(be[n])} for n in range(0, min(n_cap, 10) + 1)]
        res["coef_max_err"] = float(max(np.max(np.abs(a - ae)), np.max(np.abs(b - be))))
        if kind in ("square", "saw"):
            # скачок величины 2; для меандра у t = 0, для пилы у t = T/2
            # максимум «рожка» уже шага сетки, поэтому ищем его на мелкой сетке у разрыва
            t_jump = 0.0 if kind == "square" else T / 2
            side = 1.0 if kind == "square" else -1.0  # где предел равен +1
            t_fine = t_jump + side * np.linspace(0, 4 * T / max(N, 1), 8001)[1:]
            m = float(F.partial_sum(a, b, T, t_fine, N).max())
            res["gibbs"] = {"jump": 2.0, "max_S": m, "overshoot_fraction": (m - 1.0) / 2.0,
                            "theory": F.GIBBS_FRACTION}
    return res


def coefficient(x: np.ndarray, sr: int, t0: float, T_ms: float, n: int, kind: str,
                m: int, src: str) -> dict:
    """Шаг 3: интеграл a_n или b_n как площадь под f·базис."""
    if kind not in ("cos", "sin"):
        raise ValueError("kind должен быть cos или sin")
    seg, T, M, i0, wave = _segment(x, sr, t0, T_ms, src)
    n_cap = min(N_MAX_UI, F.max_harmonic(M))
    n = int(max(0, min(n, n_cap)))
    m = int(max(0, min(m, n_cap)))
    t = np.arange(M + 1) * T / M
    w = 2 * np.pi / T
    basis_fn = np.cos if kind == "cos" else np.sin
    basis = basis_fn(n * w * t)
    prod = seg * basis
    integral = float(np.trapezoid(prod, t))
    pos = float(np.trapezoid(np.maximum(prod, 0), t))
    neg = float(np.trapezoid(np.minimum(prod, 0), t))
    coef_trap = 2 / T * integral
    a_fft, b_fft = F.coeffs_fft(seg[:M], n_cap)
    coef_fft = float(a_fft[n] if kind == "cos" else b_fft[n])
    a_tr, b_tr = F.coeffs_trapezoid(seg, T, n)
    c_trap = complex(a_tr[n] - 1j * b_tr[n]) / 2 if n > 0 else complex(a_tr[0] / 2)
    c_fft = complex(F.complex_coeffs_fft(seg[:M], n_cap)[n])

    # ортогональность: ∫ φ_n φ_m dt для выбранного типа и смешанная пара
    bm = basis_fn(m * w * t)
    ortho_same = float(np.trapezoid(basis * bm, t))
    other = np.sin if kind == "cos" else np.cos
    ortho_cross = float(np.trapezoid(basis * other(m * w * t), t))
    if kind == "cos":
        expected_same = (T if n == m == 0 else T / 2) if n == m else 0.0
    else:
        expected_same = (T / 2 if n == m and n != 0 else 0.0)

    res = {
        "src": src, "kind": kind, "n": n, "m": m, "n_cap": n_cap, "T": T, "M": M, "omega": w,
        "t": b64(t), "f": b64(seg), "basis": b64(basis), "prod": b64(prod), "basis_m": b64(bm),
        "integral": integral, "area_pos": pos, "area_neg": neg,
        "coef_trap": coef_trap, "coef_fft": coef_fft, "diff": coef_trap - coef_fft,
        "c_trap": {"re": c_trap.real, "im": c_trap.imag, "abs": abs(c_trap)},
        "c_fft": {"re": c_fft.real, "im": c_fft.imag, "abs": abs(c_fft)},
        "a_n": float(a_tr[n]), "b_n": float(b_tr[n]),
        "ortho": {"same": ortho_same, "expected_same": expected_same, "cross": ortho_cross},
    }
    if src == "rec":
        res["t0"] = i0 / sr
        res["boundary"] = {"f0": float(seg[0]), "fT": float(seg[-1]), "f_last_period": float(seg[M - 1])}
    else:
        ae, be = F.analytic_coeffs(wave, n_cap)
        res["exact"] = float(ae[n] if kind == "cos" else be[n])
    return res


# ---------------------------------------------------------------------------
# Шаг 4. Спектр
# ---------------------------------------------------------------------------

def _naive_check(x: np.ndarray, N: int = 1024) -> dict:
    seg = np.asarray(x[:N], dtype=float)
    if seg.size < N:
        seg = np.pad(seg, (0, N - seg.size))
    t1 = time.perf_counter()
    Xn = F.dft_naive(seg)
    t2 = time.perf_counter()
    Xf = np.fft.fft(seg)
    t3 = time.perf_counter()
    return {"N": N, "max_abs_err": float(np.max(np.abs(Xn - Xf))),
            "max_abs_X": float(np.max(np.abs(Xf))),
            "t_naive": t2 - t1, "t_fft": t3 - t2}


def spectrum_payload(y: np.ndarray, sr: int, window: str) -> dict:
    s = F.amplitude_spectrum(y, sr, window)
    peaks = F.find_spectral_peaks(s)
    top = int(np.argmax(s.amplitude[1:])) + 1
    return {
        "N": s.N, "fs": sr, "df": s.df, "window": window, "coherent_gain": s.coherent_gain,
        "amp": b64(s.amplitude), "peaks": peaks,
        "top": {"k": top, "f": float(s.freqs[top]), "absX": float(np.abs(s.X[top])),
                "amp": float(s.amplitude[top]), "sum_w": s.coherent_gain * s.N},
    }


def step4(x: np.ndarray, sr: int, window: str) -> dict:
    res = spectrum_payload(x, sr, window)
    p = F.parseval(x)
    N = x.size
    res.update({
        "parseval": p,
        "naive": _naive_check(x),
        "ops_naive": float(N) ** 2,
        "ops_fft": float(N * np.log2(N)),
        "nyquist": sr / 2,
    })
    return res


def validate_ops(ops, sr: int) -> list[dict]:
    if not isinstance(ops, list) or len(ops) > 20:
        raise ValueError("ops: ожидается список до 20 операций")
    out = []
    ny = sr / 2
    for op in ops:
        t = op.get("type")
        if t in ("keep_band", "remove_band"):
            lo, hi = float(op["f_lo"]), float(op["f_hi"])
            if not (0 <= lo < hi <= ny + 1e-9):
                raise ValueError("полоса должна лежать в [0, fs/2] и f_lo < f_hi")
            out.append({"type": t, "f_lo": lo, "f_hi": hi})
        elif t in ("lowpass", "highpass"):
            fc = float(op["f_c"])
            if not (0 < fc < ny):
                raise ValueError("частота среза должна лежать в (0, fs/2)")
            out.append({"type": t, "f_c": fc})
        else:
            raise ValueError(f"неизвестная операция {t}")
    return out


def step4_edit(x: np.ndarray, sr: int, ops: list[dict], window: str) -> tuple[dict, np.ndarray]:
    y, kept = F.apply_spectral_edit(x, sr, ops)
    res = spectrum_payload(y, sr, window)
    res["kept_energy"] = kept
    res["ops"] = ops
    res["wave"] = b64(y)
    return res, y


# ---------------------------------------------------------------------------
# Шаг 5. Спектрограмма
# ---------------------------------------------------------------------------

from . import stft as ST  # noqa: E402

WIN_LENGTHS = (256, 512, 1024, 2048, 4096)
OVERLAPS = (0.0, 0.5, 0.75, 0.875)
MAX_FRAMES = 16000
MAX_CELLS = 2_500_000
#: Параметры входа AST (preprocessor_config.json и ASTFeatureExtractor, revision f826b80d).
AST_PARAMS = {"sr": 16000, "n_win": 400, "hop": 160, "n_fft": 512, "n_mels": 128, "max_frames": 1024,
              "patch": 16, "stride": 10}


def step5(x: np.ndarray, sr: int, n_win: int, overlap: float, window: str, n_mels: int,
          fmax: float, t_sel: float, mode: str) -> dict:
    if mode == "ast":
        p = AST_PARAMS
        y = sampling.resample(x, sr, p["sr"])
        sr_u, n_win, hop, n_fft, window, n_mels = p["sr"], p["n_win"], p["hop"], p["n_fft"], "hann", p["n_mels"]
        overlap = 1 - hop / n_win
    else:
        if n_win not in WIN_LENGTHS or overlap not in OVERLAPS or window not in ST.WINDOWS:
            raise ValueError("параметры окна вне списка")
        if not 8 <= n_mels <= 256:
            raise ValueError("число мел-полос от 8 до 256")
        y, sr_u = x, sr
        hop = max(1, int(round(n_win * (1 - overlap))))
        n_fft = n_win
    fmax = min(fmax if fmax > 0 else sr_u / 2, sr_u / 2)
    M = ST.n_frames(y.size, n_win, hop)
    r = ST.stft(y, sr_u, n_win, hop, window, n_fft)
    K = int(np.searchsorted(r.freqs, fmax, side="right"))
    if M > MAX_FRAMES or M * max(K, n_mels) > MAX_CELLS:
        raise ValueError(f"слишком много кадров ({M}) для показа: уменьшите перекрытие или увеличьте окно")
    S = r.power
    S_db = ST.to_db(S)
    fb = ST.mel_filterbank(sr_u, n_fft, n_mels, 0.0, sr_u / 2)
    mel_S = S @ fb.T
    mel_db = ST.to_db(mel_S)

    m = int(np.clip(np.argmin(np.abs(r.times - t_sel)), 0, M - 1))
    k = int(np.argmax(S[m, 1:])) + 1
    f_k = float(r.freqs[k])
    centers = ST.mel_centers(sr_u, n_mels)
    j = int(np.argmax(fb[:, k]))
    sp = ST.window_spreads(r.w, sr_u)
    res = {
        "mode": mode, "sr": sr_u, "n_win": n_win, "hop": hop, "n_fft": n_fft, "overlap": overlap, "window": window,
        "n_mels": n_mels, "M": M, "K": K, "fmax": fmax,
        "win_ms": 1000 * n_win / sr_u, "hop_ms": 1000 * hop / sr_u, "df": sr_u / n_fft,
        "times": b64(r.times), "freqs": b64(r.freqs[:K]), "mel_centers": b64(centers),
        "spec_db": b64(S_db[:, :K]), "mel_db": b64(mel_db), "db_max": float(S_db.max()),
        "eps": ST.EPS,
        "example": {"m": m, "t": float(r.times[m]), "start": int(r.starts[m]), "k": k, "f": f_k,
                    "S": float(S[m, k]), "S_db": float(S_db[m, k]), "mel_of_f": float(ST.hz_to_mel(f_k)),
                    "band": j, "band_center": float(centers[j]), "band_weight": float(fb[j, k])},
        "gabor": sp,
        "flatness": ST.spectral_flatness(S[m, 1:]),
    }
    if mode == "ast":
        p = AST_PARAMS
        n_t = (p["max_frames"] - p["patch"]) // p["stride"] + 1
        n_f = (p["n_mels"] - p["patch"]) // p["stride"] + 1
        res["ast"] = {**p, "frames_used": min(M, p["max_frames"]), "n_patches_t": n_t, "n_patches_f": n_f,
                      "n_patches": n_t * n_f, "padded": max(0, p["max_frames"] - M), "truncated": M > p["max_frames"]}
    return res


# ---------------------------------------------------------------------------
# Шаг 6. Классификация
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
from collections import OrderedDict  # noqa: E402

from . import probs as PR  # noqa: E402
from .labels_ru import AUDIOSET_RU, CLAP_OTHER, CLAP_TEMPLATE  # noqa: E402

_cls_cache: OrderedDict = OrderedDict()
VARIANTS = ("orig", "noise", "cut", "edit")


def make_variant(x: np.ndarray, sr: int, v: dict, edited: np.ndarray | None) -> tuple[np.ndarray, dict]:
    kind = v.get("type", "orig")
    if kind == "orig":
        return x, {"type": "orig"}
    if kind == "noise":
        snr = float(v["snr"])
        if not -10 <= snr <= 60:
            raise ValueError("SNR от −10 до 60 дБ")
        y, info = PR.add_noise(x, snr, seed=0)
        return y, {"type": "noise", **info}
    if kind == "cut":
        ops = validate_ops([{"type": "remove_band", "f_lo": v["f_lo"], "f_hi": v["f_hi"]}], sr)
        y, kept = F.apply_spectral_edit(x, sr, ops)
        return y, {"type": "cut", "f_lo": ops[0]["f_lo"], "f_hi": ops[0]["f_hi"], "kept_energy": kept}
    if kind == "edit":
        if edited is None:
            raise ValueError("на шаге 4 нет изменений")
        return edited, {"type": "edit"}
    raise ValueError("неизвестный вариант")


def _classify(y: np.ndarray, sr: int, prompts: list[str]) -> dict:
    from . import models as MD

    key = hashlib.sha1(np.ascontiguousarray(y, dtype=np.float32).tobytes() + repr((sr, prompts, MD.project_status()['available'])).encode()).hexdigest()
    if key in _cls_cache:
        _cls_cache.move_to_end(key)
        return _cls_cache[key]
    project = MD.project_infer(y, sr)
    a = MD.ast_infer(sampling.resample(y, sr, 16000))
    c = MD.clap_infer(sampling.resample(y, sr, 48000), prompts)
    res = {"ast": a, "clap": c, "project": project}
    _cls_cache[key] = res
    if len(_cls_cache) > 24:
        _cls_cache.popitem(last=False)
    return res


def _ast_summary(a: dict, k: int = 5) -> dict:
    z = a["logits"]
    p = PR.sigmoid(z)
    idx = PR.top_k(p, k)
    top = [{"label": a["labels"][i], "ru": AUDIOSET_RU.get(a["labels"][i]), "z": float(z[i]), "p": float(p[i]),
            "h": float(PR.binary_entropy(p[i]))} for i in idx]
    sm = PR.softmax(z)
    return {"top": top, "sum_sigmoid": float(p.sum()), "n_classes": int(z.size), "n_above_half": int((p > 0.5).sum()),
            "softmax_top": float(sm[idx[0]]), "infer_s": a["infer_s"], "device": a["device"]}


def _clap_summary(c: dict, labels: list[dict], threshold: float) -> dict:
    p = PR.softmax(c["logits"])
    i = int(np.argmax(p))
    H = PR.entropy(p)
    is_other = labels[i]["ru"] == CLAP_OTHER[0] or p[i] < threshold
    return {
        "labels": [{**lab, "z": float(c["logits"][j]), "cos": float(c["cos"][j]), "p": float(p[j])} for j, lab in enumerate(labels)],
        "scale": c["scale"], "tau": 1 / c["scale"], "entropy": H, "entropy_max": float(np.log(len(labels))),
        "best": i, "threshold": threshold, "verdict": CLAP_OTHER[0] if is_other else labels[i]["ru"],
        "uncertain": bool(p[i] < threshold), "infer_s": c["infer_s"], "device": c["device"],
        "repeat_padded": c["repeat_padded"],
    }


def validate_labels(labels) -> list[dict]:
    if not isinstance(labels, list) or not 1 <= len(labels) <= 15:
        raise ValueError("от 1 до 15 меток")
    out = []
    for lab in labels:
        ru, en = str(lab.get("ru", "")).strip()[:40], str(lab.get("en", "")).strip()[:60]
        if not en:
            raise ValueError("у каждой метки должен быть текст для модели (по-английски)")
        if ru != CLAP_OTHER[0]:
            out.append({"ru": ru or en, "en": en})
    out.append({"ru": CLAP_OTHER[0], "en": CLAP_OTHER[1]})  # «другое» всегда есть
    return out


def step6(x: np.ndarray, sr: int, labels: list[dict], threshold: float, variant: dict,
          edited: np.ndarray | None) -> dict:
    if not 0 < threshold < 1:
        raise ValueError("порог от 0 до 1")
    prompts = [CLAP_TEMPLATE.format(lab["en"]) for lab in labels]
    base = _classify(x, sr, prompts)
    res = {
        "orig": {"ast": _ast_summary(base["ast"]), "clap": _clap_summary(base["clap"], labels, threshold)},
        "prompts": prompts, "template": CLAP_TEMPLATE,
    }
    res['orig']['project'] = base['project']
    a = base["ast"]
    n16 = int(round(x.size / sr * 16000))
    res["input"] = {"values": b64(a["input"]), "frames": a["input"].shape[0], "mels": a["input"].shape[1],
                    "real_frames": int(min(a["input"].shape[0], max(0, (n16 - 400) // 160 + 1))),
                    "mean": a["mean"], "std": a["std"],
                    "patches": {"f": (128 - 16) // 10 + 1, "t": (1024 - 16) // 10 + 1}}
    if variant.get("type", "orig") != "orig":
        y, info = make_variant(x, sr, variant, edited)
        v = _classify(y, sr, prompts)
        res["variant"] = {"info": info, "ast": _ast_summary(v["ast"]), "clap": _clap_summary(v["clap"], labels, threshold)}
        res['variant']['project'] = v['project']
        # те же 5 классов AudioSet, что в исходной записи: как изменились их вероятности
        lab_idx = {l: i for i, l in enumerate(v["ast"]["labels"])}
        pv = PR.sigmoid(v["ast"]["logits"])
        res["variant"]["ast_same"] = [float(pv[lab_idx[t["label"]]]) for t in res["orig"]["ast"]["top"]]
    return res
