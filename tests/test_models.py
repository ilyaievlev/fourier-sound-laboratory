"""Тесты шага 6 с настоящими моделями (только CPU, чтобы не мешать обучению на GPU).

Пропускаются, если весов нет в кэше HF или задано FSL_SKIP_MODELS=1.
"""
import os

import numpy as np
import pytest

os.environ["FSL_DEVICE"] = "cpu"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

torch = pytest.importorskip("torch")
from analyzer.services import models as MD  # noqa: E402
from analyzer.services import steps  # noqa: E402
from analyzer.services.labels_ru import CLAP_DEFAULT  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("FSL_SKIP_MODELS") == "1" or not (MD.weights_cached("ast") and MD.weights_cached("clap")),
    reason="нет весов в кэше (scripts/download_models.py) или FSL_SKIP_MODELS=1")

SR = 48000


def tone(f, dur=3.0):
    t = np.arange(int(dur * SR)) / SR
    return 0.5 * np.sin(2 * np.pi * f * t)


def test_device_respects_env():
    assert MD.pick_device() == "cpu"


def test_ast_shapes_and_sigmoid():
    a = MD.ast_infer(tone(1000, 2.0)[::3])  # 48 → 16 кГц (тон 1 кГц, алиасинга нет)
    assert a["logits"].shape == (527,)
    assert a["input"].shape == (1024, 128)
    assert a["mean"] == pytest.approx(-4.2677393) and a["std"] == pytest.approx(4.5689974)


def test_ast_recognizes_pure_tone():
    """Проверка связки «признаки → модель → метки»: чистый тон должен попасть в класс Sine wave."""
    labels = steps.validate_labels([{"ru": r, "en": e} for r, e in CLAP_DEFAULT])
    r = steps.step6(tone(1000), SR, labels, 0.5, {"type": "orig"}, None)
    top = [t["label"] for t in r["orig"]["ast"]["top"]]
    assert "Sine wave" in top, top


def test_clap_probabilities_and_other_label():
    labels = steps.validate_labels([{"ru": "свист", "en": "a person whistling"}, {"ru": "собака", "en": "a dog barking"}])
    assert labels[-1]["ru"] == "другое"
    r = steps.step6(tone(1800), SR, labels, 0.5, {"type": "noise", "snr": 0}, None)
    c = r["orig"]["clap"]
    p = np.array([l["p"] for l in c["labels"]])
    assert p.sum() == pytest.approx(1.0, abs=1e-5)
    assert 0 <= c["entropy"] <= c["entropy_max"] + 1e-9
    assert c["scale"] == pytest.approx(1 / c["tau"])
    for l in c["labels"]:
        assert l["z"] == pytest.approx(c["scale"] * l["cos"], rel=1e-5)
    assert "variant" in r and len(r["variant"]["ast_same"]) == 5


def test_validate_labels():
    with pytest.raises(ValueError):
        steps.validate_labels([{"ru": "x", "en": ""}])
    out = steps.validate_labels([{"ru": "другое", "en": "x"}, {"ru": "", "en": "rain"}])
    assert out == [{"ru": "rain", "en": "rain"}, {"ru": "другое", "en": "some other sound"}]


def test_classify_api(tmp_path):
    import json

    import django
    from django.conf import settings
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()
    settings.ALLOWED_HOSTS = [*settings.ALLOWED_HOSTS, "testserver"]
    settings.SESSIONS_ROOT = tmp_path
    from django.test import Client
    c = Client()
    rid = c.post("/api/sample", {"id": "whistle"}).json()["rid"]
    body = {"labels": [{"ru": "свист", "en": "a person whistling"}], "threshold": 0.5, "variant": {"type": "cut", "f_lo": 1000, "f_hi": 3000}}
    r = c.post(f"/api/rec/{rid}/classify", json.dumps(body), content_type="application/json")
    assert r.status_code == 200, r.content
    d = r.json()
    assert d["variant"]["info"]["kept_energy"] < 0.05
    assert c.get(f"/api/rec/{rid}/variant.wav", {"type": "noise", "snr": 5}).status_code == 200
    assert c.get("/api/models/status").json()["ast"]["revision"].startswith("f826b80d")
