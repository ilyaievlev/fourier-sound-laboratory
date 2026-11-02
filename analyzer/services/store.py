"""Хранение записей во временной папке по ID сессии. Без Django."""
from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

_ID = re.compile(r"^[0-9a-f]{16,32}$")
MAX_AGE_S = 24 * 3600


def new_id(nbytes: int = 8) -> str:
    return secrets.token_hex(nbytes)


def valid_id(s: str) -> bool:
    return bool(s) and bool(_ID.match(s))


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def rec_dir(self, sid: str, rid: str) -> Path:
        if not (valid_id(sid) and valid_id(rid)):
            raise KeyError("неверный идентификатор")
        return self.root / sid / rid

    def create(self, sid: str, x: np.ndarray, sr: int, meta: dict) -> str:
        rid = new_id()
        d = self.rec_dir(sid, rid)
        d.mkdir(parents=True, exist_ok=False)
        sf.write(str(d / "work.wav"), np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return rid

    def load(self, sid: str, rid: str) -> tuple[np.ndarray, int, dict]:
        d = self.rec_dir(sid, rid)
        p = d / "work.wav"
        if not p.exists():
            raise KeyError("запись не найдена")
        x, sr = _read_cached(str(p), p.stat().st_mtime_ns)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return x, sr, meta

    def cleanup(self, max_age_s: float = MAX_AGE_S) -> int:
        """Удаляет папки сессий старше max_age_s. Возвращает число удалённых."""
        if not self.root.exists():
            return 0
        now, n = time.time(), 0
        for d in self.root.iterdir():
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n


@lru_cache(maxsize=16)
def _read_cached(path: str, _mtime: int) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64")
    x.setflags(write=False)
    return x, int(sr)
