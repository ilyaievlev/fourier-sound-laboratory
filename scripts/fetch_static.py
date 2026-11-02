"""Скачивает Plotly.js и KaTeX из npm-реестра в analyzer/static/vendor.

Версии закреплены, целостность проверяется по sha512 из реестра npm.
После этого фронтенд работает без интернета.

    python scripts/fetch_static.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "analyzer" / "static" / "vendor"

PACKAGES = {
    # имя пакета: (версия, {путь в архиве -> путь в vendor})
    "plotly.js-dist-min": ("4.1.2", {"package/plotly.min.js": "plotly/plotly.min.js",
                                     "package/LICENSE": "plotly/LICENSE"}),
    "katex": ("0.18.11", {"package/dist/katex.min.js": "katex/katex.min.js",
                          "package/dist/katex.min.css": "katex/katex.min.css",
                          "package/dist/fonts/": "katex/fonts/",
                          "package/LICENSE": "katex/LICENSE"}),
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> None:
    for name, (version, mapping) in PACKAGES.items():
        meta = json.loads(fetch(f"https://registry.npmjs.org/{name}/{version}"))
        tar_bytes = fetch(meta["dist"]["tarball"])
        algo, b64 = meta["dist"]["integrity"].split("-", 1)
        assert algo == "sha512"
        digest = base64.b64encode(hashlib.sha512(tar_bytes).digest()).decode()
        if digest != b64:
            raise SystemExit(f"{name}: контрольная сумма не совпала")
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                for src, dst in mapping.items():
                    if member.name == src or (src.endswith("/") and member.name.startswith(src)):
                        rel = dst + member.name[len(src):] if src.endswith("/") else dst
                        out = VENDOR / rel
                        # защита от путей вида ../ в архиве
                        if VENDOR.resolve() not in out.resolve().parents:
                            raise SystemExit(f"подозрительный путь: {member.name}")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as f, open(out, "wb") as o:
                            shutil.copyfileobj(f, o)
        print(f"{name}@{version}: ok")


if __name__ == "__main__":
    main()
