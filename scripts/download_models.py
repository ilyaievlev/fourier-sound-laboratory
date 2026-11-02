"""Скачивает готовые модели шага 6 в кэш Hugging Face (один раз, нужен интернет).

После этого приложение работает офлайн: модели грузятся с local_files_only.

    uv run python scripts/download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download  # noqa: E402

from analyzer.services.models import MODELS  # noqa: E402


def main() -> None:
    for key, spec in MODELS.items():
        path = snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=spec["files"])
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 2 ** 20
        print(f"{key}: {spec['repo']}@{spec['revision'][:8]} → {path} ({size:.0f} МБ)")


if __name__ == "__main__":
    main()
