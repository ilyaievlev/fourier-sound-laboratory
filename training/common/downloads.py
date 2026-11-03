"""Bounded public downloads, checksums, and extraction into a fixed data root."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile


def request(url, **headers):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'fourier-sound-lab/0.1 (research dataset preparation)', **headers}), timeout=90)


def download(url, path, limit, checksum=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pending = path.with_suffix(path.suffix + '.partial')
        with request(url) as response, pending.open('wb') as out:
            if int(response.headers.get('Content-Length', 0)) > limit:
                raise ValueError(f'Download exceeds limit: {url}')
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise ValueError('Download budget exceeded')
                out.write(chunk)
                if total % (100 * 1024**2) == 0:
                    print(f'{path.name}: {total // 1024**2} MiB', flush=True)
        pending.replace(path)
    if checksum:
        algorithm, expected = checksum.split(':', 1)
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, algorithm).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {path}')
    return path


def extract(archive, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if not (root / item.filename).resolve().is_relative_to(root):
                raise ValueError('Unsafe ZIP path')
        source.extractall(root)


def record_info(record, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'zenodo_{record}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    with request(f'https://zenodo.org/api/records/{record}') as response:
        raw = response.read(8 * 1024**2)
    data = json.loads(raw)
    path.write_bytes(raw)
    return data
