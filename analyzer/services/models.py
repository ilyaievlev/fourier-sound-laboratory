"""Загрузка и инференс AST и CLAP. Модели грузятся один раз на процесс и кэшируются.

Без Django: можно вызывать из тестов и скриптов.

Устройство: FSL_DEVICE (cuda / mps / cpu), иначе cuda → mps → cpu.
Пока на той же видеокарте идёт обучение, ставьте FSL_DEVICE=cpu.
Число потоков CPU: FSL_TORCH_THREADS (по умолчанию 4, чтобы не отнимать все ядра у обучения).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

MODELS = {
    "ast": {
        "repo": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "revision": "f826b80d28226b62986cc218e5cec390b1096902",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"],
        "license": "BSD-3-Clause",
        "sr": 16000,
    },
    "clap": {
        "repo": "laion/clap-htsat-unfused",
        "revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin", "merges.txt", "vocab.json",
                  "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "license": "Apache-2.0",
        "sr": 48000,
    },
}

CLAP_MAX_S = 10.0

_lock = threading.Lock()
_cache: dict[str, object] = {}


def pick_device() -> str:
    import torch

    want = os.environ.get("FSL_DEVICE", "").strip().lower()
    if want in ("cpu", "cuda", "mps"):
        if want == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if want == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return "cpu"
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def weights_cached(key: str) -> bool:
    """Есть ли веса модели в локальном кэше HF (без обращения к сети)."""
    from huggingface_hub import try_to_load_from_cache

    spec = MODELS[key]
    weight = next(f for f in spec["files"] if f.endswith((".safetensors", ".bin")))
    p = try_to_load_from_cache(spec["repo"], weight, revision=spec["revision"])
    return isinstance(p, str)


def _from_pretrained(cls, key: str):
    spec = MODELS[key]
    try:
        return cls.from_pretrained(spec["repo"], revision=spec["revision"], local_files_only=True)
    except OSError:
        # первая загрузка: качаем ровно ту же ревизию
        return cls.from_pretrained(spec["repo"], revision=spec["revision"])


def _load(key: str):
    import torch

    if key in _cache:
        return _cache[key]
    with _lock:
        if key in _cache:
            return _cache[key]
        torch.set_num_threads(int(os.environ.get("FSL_TORCH_THREADS", "4")))
        device = pick_device()
        t = time.perf_counter()
        if key == "ast":
            from transformers import ASTFeatureExtractor, ASTForAudioClassification

            proc = _from_pretrained(ASTFeatureExtractor, key)
            model = _from_pretrained(ASTForAudioClassification, key)
        elif key == "clap":
            from transformers import ClapModel, ClapProcessor

            proc = _from_pretrained(ClapProcessor, key)
            model = _from_pretrained(ClapModel, key)
        else:
            raise KeyError(key)
        model.eval().to(device)
        _cache[key] = {"proc": proc, "model": model, "device": device, "load_s": time.perf_counter() - t}
        return _cache[key]


def status() -> dict:
    return {k: {"loaded": k in _cache, "cached": weights_cached(k),
                "device": _cache[k]["device"] if k in _cache else None,
                "load_s": _cache[k]["load_s"] if k in _cache else None,
                "repo": MODELS[k]["repo"], "revision": MODELS[k]["revision"], "license": MODELS[k]["license"]}
            for k in MODELS}


def ast_infer(x16: np.ndarray) -> dict:
    """Логиты AST (527) и тот самый вход модели — нормированная лог-мел-спектрограмма 1024×128."""
    import torch

    m = _load("ast")
    proc, model = m["proc"], m["model"]
    feats = proc(np.asarray(x16, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
    with _lock, torch.inference_mode():
        t = time.perf_counter()
        logits = model(feats["input_values"].to(m["device"])).logits[0].float().cpu().numpy()
        dt = time.perf_counter() - t
    labels = [model.config.id2label[i] for i in range(logits.size)]
    return {"logits": logits, "labels": labels, "input": feats["input_values"][0].numpy(), "infer_s": dt,
            "mean": float(proc.mean), "std": float(proc.std), "device": m["device"]}


def clap_infer(x48: np.ndarray, prompts: list[str]) -> dict:
    """Логиты CLAP: z_i = e^{s} · cos(аудио, текст_i), s = logit_scale_a."""
    import torch

    m = _load("clap")
    proc, model = m["proc"], m["model"]
    x48 = np.asarray(x48, dtype=np.float32)[: int(CLAP_MAX_S * 48000)]
    inp = proc(text=prompts, audio=[x48], sampling_rate=48000, return_tensors="pt", padding=True)
    inp = {k: v.to(m["device"]) for k, v in inp.items()}
    with _lock, torch.inference_mode():
        t = time.perf_counter()
        out = model(**inp)
        dt = time.perf_counter() - t
        scale = float(model.logit_scale_a.exp().item())
    logits = out.logits_per_audio[0].float().cpu().numpy()
    return {"logits": logits, "scale": scale, "cos": logits / scale, "infer_s": dt, "device": m["device"],
            "repeat_padded": x48.size < int(CLAP_MAX_S * 48000)}
