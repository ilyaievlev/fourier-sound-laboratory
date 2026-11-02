"""Download a pinned official ESC-50 archive, capped at 1 GiB."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = '33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
URL = f'https://codeload.github.com/karolpiczak/ESC-50/zip/{REVISION}'
LIMIT = 1024 ** 3


def main():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    archive = data / f'ESC-50-{REVISION}.zip'
    partial = archive.with_suffix('.zip.partial')
    if not archive.exists():
        digest, total = hashlib.sha256(), 0
        with urllib.request.urlopen(URL, timeout=90) as response, partial.open('wb') as handle:
            length = int(response.headers.get('Content-Length', 0))
            if length > LIMIT:
                raise ValueError('Archive exceeds download limit')
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > LIMIT:
                    raise ValueError('Archive exceeds download limit')
                handle.write(chunk)
                digest.update(chunk)
                if total % (50 * 1024 * 1024) == 0:
                    print(f'Downloaded {total // 1024**2} MiB', flush=True)
        partial.rename(archive)
    destination = data / 'ESC-50-master'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    with zipfile.ZipFile(archive) as source:
        if sum(item.file_size for item in source.infolist()) > 2 * LIMIT:
            raise ValueError('Expanded archive exceeds limit')
        for item in source.infolist():
            target = (data / item.filename).resolve()
            if not target.is_relative_to(data.resolve()):
                raise ValueError('Unsafe archive path')
        source.extractall(data)
    (data / f'ESC-50-{REVISION}').rename(destination)
    with archive.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    (data / 'esc50_source.json').write_text(json.dumps(dict(
        url=URL, revision=REVISION, archive_sha256=sha,
        archive_bytes=archive.stat().st_size,
        license_url='https://github.com/karolpiczak/ESC-50/blob/master/LICENSE',
    ), indent=2) + '\n', encoding='utf-8')
    print(f'Extracted to {destination}', flush=True)


if __name__ == '__main__':
    main()
