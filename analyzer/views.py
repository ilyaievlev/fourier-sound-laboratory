"""HTTP-слой: только разбор параметров и вызов analyzer.services.*"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
from functools import wraps
from pathlib import Path

import numpy as np
import soundfile as sf
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .services import audio, steps
from .services.store import Store, new_id, valid_id

ALLOWED_EXT = {".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac", ".m4a"}


def _store() -> Store:
    return Store(settings.SESSIONS_ROOT)


def _sid(request) -> str:
    sid = request.session.get("sid")
    if not valid_id(sid or ""):
        sid = new_id()
        request.session["sid"] = sid
    return sid


def _err(msg: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"error": msg}, status=status)


def with_rec(view):
    """Загружает запись текущей сессии и передаёт (x, sr, meta) во view."""

    @wraps(view)
    def inner(request, rid, *a, **kw):
        try:
            x, sr, meta = _store().load(_sid(request), rid)
        except KeyError:
            return _err("Запись не найдена. Загрузите звук заново.", 404)
        try:
            return view(request, rid, x, sr, meta, *a, **kw)
        except (ValueError, KeyError, TypeError) as e:
            return _err(f"Неверные параметры: {e}")

    return inner


def _wav_response(x: np.ndarray, sr: int, name: str) -> HttpResponse:
    buf = io.BytesIO()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    y = x / peak if peak > 1 else x
    sf.write(buf, np.asarray(y, dtype=np.float64), sr, format="WAV", subtype="PCM_16")
    r = HttpResponse(buf.getvalue(), content_type="audio/wav")
    r["Content-Disposition"] = f'inline; filename="{name}"'
    r["Cache-Control"] = "no-store"
    return r


def load_manifest() -> list[dict]:
    p = Path(settings.SAMPLES_DIR) / "manifest.json"
    if not p.exists():
        return []
    items = json.loads(p.read_text(encoding="utf-8"))
    return [it for it in items if (Path(settings.SAMPLES_DIR) / it["file"]).exists()]


# ---------------------------------------------------------------------------

@require_GET
def index(request):
    _sid(request)
    _store().cleanup()
    return render(request, "analyzer/index.html", {"samples": load_manifest()})


def _ingest(request, data: np.ndarray, sr: int, meta: dict) -> JsonResponse:
    p = audio.prepare(data, sr)
    meta = {**meta, **audio.prepared_info(p)}
    rid = _store().create(_sid(request), p.x, p.sr, meta)
    return JsonResponse({"rid": rid, **steps.overview(p.x, p.sr, meta)})


@require_POST
def api_upload(request):
    f = request.FILES.get("file")
    if f is None:
        return _err("Файл не передан.")
    if f.size > settings.FSL_MAX_UPLOAD_BYTES:
        return _err("Файл слишком большой (больше 25 МБ).")
    ext = Path(f.name).suffix.lower() or ".webm"
    if ext not in ALLOWED_EXT:
        return _err(f"Формат {ext} не поддерживается. Нужен WAV, MP3, OGG или WEBM.")
    source = request.POST.get("source", "upload")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"input{ext}"
        with open(tmp, "wb") as out:
            for chunk in f.chunks():
                out.write(chunk)
        try:
            data, sr = audio.decode(tmp)
        except ValueError as e:
            return _err(str(e))
    if data.size == 0:
        return _err("В файле нет звука.")
    name = "Запись с микрофона" if source == "mic" else f.name
    return _ingest(request, data, sr, {"source": source, "name": name})


@require_POST
def api_sample(request):
    sid = request.POST.get("id", "")
    item = next((it for it in load_manifest() if it["id"] == sid), None)
    if item is None:
        return _err("Нет такого примера.", 404)
    data, sr = audio.decode(Path(settings.SAMPLES_DIR) / item["file"])
    return _ingest(request, data, sr, {"source": "sample", "name": item["title"], "sample_id": item["id"],
                                       "synthetic": item.get("synthetic", False)})


@require_GET
@with_rec
def api_overview(request, rid, x, sr, meta):
    return JsonResponse({"rid": rid, **steps.overview(x, sr, meta)})


@require_GET
@with_rec
def rec_audio(request, rid, x, sr, meta):
    return _wav_response(x, sr, "record.wav")


def _fs_aa(request, sr):
    fs = int(request.GET.get("fs", sr))
    if fs not in steps.FS_OPTIONS and fs != sr:
        raise ValueError("fs вне списка")
    return fs, request.GET.get("aa", "1") == "1"


@require_GET
@with_rec
def api_step1(request, rid, x, sr, meta):
    fs, aa = _fs_aa(request, sr)
    return JsonResponse(steps.step1(x, sr, fs, aa))


@require_GET
@with_rec
def step1_audio(request, rid, x, sr, meta):
    fs, aa = _fs_aa(request, sr)
    return _wav_response(steps.step1_audio(x, sr, fs, aa), sr, f"fs{fs}.wav")


def _seg_params(request):
    src = request.GET.get("src", "rec")
    if src not in steps.SOURCES:
        raise ValueError("src")
    T_ms = float(request.GET.get("T", 30))
    if not 2 <= T_ms <= 200:
        raise ValueError("T должна быть от 2 до 200 мс")
    return float(request.GET.get("t0", 0)), T_ms, src


@require_GET
@with_rec
def api_series(request, rid, x, sr, meta):
    t0, T_ms, src = _seg_params(request)
    N = int(request.GET.get("N", 10))
    return JsonResponse(steps.series(x, sr, t0, T_ms, N, src))


@require_GET
@with_rec
def api_coef(request, rid, x, sr, meta):
    t0, T_ms, src = _seg_params(request)
    return JsonResponse(steps.coefficient(x, sr, t0, T_ms, int(request.GET.get("n", 1)),
                                          request.GET.get("kind", "cos"), int(request.GET.get("m", 2)), src))


def _window(request):
    w = request.GET.get("window", "hann")
    if w not in ("hann", "rect"):
        raise ValueError("window")
    return w


@require_GET
@with_rec
def api_spectrum(request, rid, x, sr, meta):
    return JsonResponse(steps.step4(x, sr, _window(request)))


@require_POST
@with_rec
def api_edit(request, rid, x, sr, meta):
    body = json.loads(request.body or b"{}")
    ops = steps.validate_ops(body.get("ops", []), sr)
    window = body.get("window", "hann")
    if window not in ("hann", "rect"):
        raise ValueError("window")
    res, y = steps.step4_edit(x, sr, ops, window)
    d = _store().rec_dir(_sid(request), rid)
    audio.write_wav(d / "edited.wav", y, sr)
    (d / "edit.json").write_text(json.dumps({"ops": ops}, ensure_ascii=False), encoding="utf-8")
    v = hashlib.sha1(json.dumps(ops, sort_keys=True).encode()).hexdigest()[:10]
    res["audio_url"] = f"/api/rec/{rid}/edit.wav?v={v}"
    return JsonResponse(res)


@require_GET
@with_rec
def edit_audio(request, rid, x, sr, meta):
    p = _store().rec_dir(_sid(request), rid) / "edited.wav"
    if not p.exists():
        return _err("Изменённой версии нет.", 404)
    y, sr2 = sf.read(str(p), dtype="float64")
    return _wav_response(y, sr2, "edited.wav")


@require_GET
@with_rec
def band_audio(request, rid, x, sr, meta):
    """Только полоса [f_lo, f_hi] — «послушать эту компоненту». Ничего не сохраняет."""
    ops = steps.validate_ops([{"type": "keep_band", "f_lo": request.GET["f_lo"], "f_hi": request.GET["f_hi"]}], sr)
    y, _ = steps.F.apply_spectral_edit(x, sr, ops)
    return _wav_response(y, sr, "band.wav")


@require_GET
@with_rec
def api_spectrogram(request, rid, x, sr, meta):
    g = request.GET
    mode = g.get("mode", "custom")
    if mode not in ("custom", "ast"):
        raise ValueError("mode")
    return JsonResponse(steps.step5(x, sr, int(g.get("win", 1024)), float(g.get("overlap", 0.75)),
                                    g.get("window", "hann"), int(g.get("mels", 64)), float(g.get("fmax", 8000)),
                                    float(g.get("t", 0)), mode))


@require_GET
def api_models_status(request):
    from .services import models as MD
    return JsonResponse(MD.status())


@require_POST
@with_rec
def api_classify(request, rid, x, sr, meta):
    body = json.loads(request.body or b"{}")
    labels = steps.validate_labels(body.get("labels", []))
    variant = body.get("variant", {"type": "orig"})
    edited = None
    if variant.get("type") == "edit":
        p = _store().rec_dir(_sid(request), rid) / "edited.wav"
        if p.exists():
            edited, _ = sf.read(str(p), dtype="float64")
    try:
        res = steps.step6(x, sr, labels, float(body.get("threshold", 0.5)), variant, edited)
    except ImportError:
        return _err("Не установлены PyTorch/transformers: выполните uv sync --extra ml.", 503)
    except OSError as e:
        return _err(f"Модели не загружены: {e}. Выполните uv run python scripts/download_models.py.", 503)
    res["labels"] = labels
    return JsonResponse(res)


@require_GET
@with_rec
def variant_audio(request, rid, x, sr, meta):
    """Искажённый вариант для шага 6 (шум или вырезанная полоса) — послушать, что получает модель."""
    v = {k: request.GET[k] for k in ("type", "snr", "f_lo", "f_hi") if k in request.GET}
    edited = None
    if v.get("type") == "edit":
        p = _store().rec_dir(_sid(request), rid) / "edited.wav"
        if p.exists():
            edited, _ = sf.read(str(p), dtype="float64")
    y, _ = steps.make_variant(x, sr, v, edited)
    return _wav_response(y, sr, "variant.wav")
