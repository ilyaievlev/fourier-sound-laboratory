"""Интеграционные тесты HTTP-слоя: все шаги 1–4 на примерах из галереи."""
import json
import os

import django
import numpy as np
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from django.conf import settings
    django.setup()
    settings.SESSIONS_ROOT = tmp_path_factory.mktemp("sessions")
    settings.ALLOWED_HOSTS = [*settings.ALLOWED_HOSTS, "testserver"]
    from django.test import Client
    return Client()


def load_sample(client, sid="voice"):
    r = client.post("/api/sample", {"id": sid})
    assert r.status_code == 200, r.content
    return r.json()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.content.decode()
    assert "Как ИИ слышит мир" in html and "Галерея примеров" in html


@pytest.mark.parametrize("sid", ["voice", "whistle", "clap", "siren", "music", "pump_normal", "pump_anomaly"])
def test_all_steps_on_gallery(client, sid):
    j = load_sample(client, sid)
    rid, t0 = j["rid"], j["suggested_t0"]
    assert j["sr"] == 48000 and abs(j["duration"] - 4.0) < 1e-9
    for fs in (2000, 8000, 44100):
        for aa in (0, 1):
            r = client.get(f"/api/rec/{rid}/step1", {"fs": fs, "aa": aa})
            assert r.status_code == 200, r.content
            assert r.json()["nyquist"] == fs / 2
    w = client.get(f"/api/rec/{rid}/step1.wav", {"fs": 4000, "aa": 0})
    assert w.status_code == 200 and w["Content-Type"] == "audio/wav"
    for src in ("rec", "square", "saw", "abs"):
        r = client.get(f"/api/rec/{rid}/series", {"t0": t0, "T": 30, "N": 200, "src": src})
        assert r.status_code == 200, r.content
        d = r.json()
        assert d["N"] == d["n_cap"] and d["bessel_N"] <= d["energy"] * (1 + 1e-9)
        r = client.get(f"/api/rec/{rid}/coef", {"t0": t0, "T": 30, "n": 3, "kind": "sin", "m": 3, "src": src})
        assert r.status_code == 200, r.content
    for win in ("hann", "rect"):
        r = client.get(f"/api/rec/{rid}/spectrum", {"window": win})
        assert r.status_code == 200
        assert r.json()["parseval"]["rel_error"] < 1e-10


def test_gibbs_in_api(client):
    rid = load_sample(client)["rid"]
    for src in ("square", "saw"):
        d = client.get(f"/api/rec/{rid}/series", {"T": 20, "N": 199, "src": src}).json()
        # у пилы f наклонная около разрыва: max S_N − 1 отстаёт от предела на ~1/(2N)
        tol = 5e-4 if src == "square" else 1 / 199
        assert abs(d["gibbs"]["overshoot_fraction"] - 0.08949) < tol, (src, d["gibbs"])
    assert d["coef_max_err"] < 1e-4


def test_edit_and_band(client):
    rid = load_sample(client, "whistle")["rid"]
    body = {"ops": [{"type": "lowpass", "f_c": 1000}], "window": "hann"}
    r = client.post(f"/api/rec/{rid}/edit", json.dumps(body), content_type="application/json")
    assert r.status_code == 200, r.content
    assert r.json()["kept_energy"] < 0.05  # свист выше 1 кГц почти весь удалён
    assert client.get(r.json()["audio_url"]).status_code == 200
    assert client.get(f"/api/rec/{rid}/band.wav", {"f_lo": 1800, "f_hi": 1900}).status_code == 200
    bad = client.post(f"/api/rec/{rid}/edit", json.dumps({"ops": [{"type": "lowpass", "f_c": 1e6}]}),
                      content_type="application/json")
    assert bad.status_code == 400


def test_upload_wav_and_errors(client, tmp_path):
    import soundfile as sf
    sr = 44100
    t = np.arange(int(2 * sr)) / sr
    p = tmp_path / "a.wav"
    sf.write(p, np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1) * 0.5, sr)
    with open(p, "rb") as f:
        r = client.post("/api/upload", {"file": f, "source": "upload"})
    assert r.status_code == 200, r.content
    j = r.json()
    assert j["sr"] == 44100 and j["meta"]["original_channels"] == 2
    bad = tmp_path / "x.txt"
    bad.write_text("hello")
    with open(bad, "rb") as f:
        assert client.post("/api/upload", {"file": f}).status_code == 400
    fake = tmp_path / "x.webm"
    fake.write_bytes(b"not audio at all")
    with open(fake, "rb") as f:
        assert client.post("/api/upload", {"file": f}).status_code == 400


def test_foreign_session_cannot_read(client):
    from django.test import Client
    rid = load_sample(client)["rid"]
    other = Client()
    other.get("/")
    assert other.get(f"/api/rec/{rid}/step1").status_code == 404
    assert client.get("/api/rec/..%2f..%2fetc/step1").status_code == 404


@pytest.mark.parametrize("params", [
    {"mode": "custom", "win": 1024, "overlap": 0.75, "window": "hann", "mels": 64, "fmax": 8000},
    {"mode": "custom", "win": 256, "overlap": 0.875, "window": "rect", "mels": 128, "fmax": 0},
    {"mode": "custom", "win": 4096, "overlap": 0.5, "window": "blackman", "mels": 32, "fmax": 4000},
    {"mode": "ast"},
])
def test_spectrogram_api(client, params):
    rid = load_sample(client, "siren")["rid"]
    r = client.get(f"/api/rec/{rid}/spectrogram", {**params, "t": 1.0})
    assert r.status_code == 200, r.content
    d = r.json()
    assert d["gabor"]["product"] >= d["gabor"]["bound"] * (1 - 1e-3)
    if params["mode"] == "ast":
        assert d["sr"] == 16000 and d["hop"] == 160 and d["ast"]["n_patches"] == 1212
    assert client.get(f"/api/rec/{rid}/spectrogram", {"win": 1000}).status_code == 400
