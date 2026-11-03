"""Atomic epoch checkpoints using safetensors and JSON (no pickle)."""
import hashlib
import json
from pathlib import Path
import random
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def rng_state():
    numpy = np.random.get_state()
    return dict(python=random.getstate(), numpy=(numpy[0], numpy[1].tolist(), *numpy[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_checkpoint(directory, model, optimizer, scheduler, scaler, progress):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    pointer = directory / 'latest.json'
    previous = json.loads(pointer.read_text())['blob'] if pointer.exists() else None
    tensors = {}
    def pack(value):
        if isinstance(value, torch.Tensor):
            key = f'tensor_{len(tensors)}'
            tensors[key] = value.detach().cpu().contiguous()
            return {'tensor': key}
        if isinstance(value, dict):
            return {'dict': [[pack(key), pack(item)] for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'tuple' if isinstance(value, tuple) else 'list': [pack(item) for item in value]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return {'value': value}
        raise TypeError(f'Unsupported checkpoint value: {type(value)}')
    tree = pack(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     rng=rng_state(), progress=progress))
    blob = f'epoch_{uuid.uuid4().hex}.safetensors'
    pending = directory / (blob + '.tmp')
    save_file(tensors, str(pending))
    pending.replace(directory / blob)
    with (directory / blob).open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    # The pointer changes only after the complete immutable tensor file exists.
    atomic_json(pointer, dict(schema=1, blob=blob, sha256=digest, tree=tree))
    if previous and Path(previous).name == previous and previous != blob:
        (directory / previous).unlink(missing_ok=True)


def load_checkpoint(directory):
    directory = Path(directory)
    pointer = json.loads((directory / 'latest.json').read_text(encoding='utf-8'))
    if pointer['schema'] != 1 or Path(pointer['blob']).name != pointer['blob']:
        raise ValueError('Invalid checkpoint pointer')
    blob = directory / pointer['blob']
    with blob.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != pointer['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
    tensors = load_file(str(blob), device='cpu')
    def unpack(value):
        if 'tensor' in value:
            return tensors[value['tensor']]
        if 'dict' in value:
            return {unpack(key): unpack(item) for key, item in value['dict']}
        if 'tuple' in value:
            return tuple(unpack(item) for item in value['tuple'])
        if 'list' in value:
            return [unpack(item) for item in value['list']]
        return value['value']
    return unpack(pointer['tree'])
