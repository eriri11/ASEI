"""Small I/O, reproducibility and checkpoint helpers."""
import hashlib
import json
import random
import uuid
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def unpack(bits, shape):
    """Restore a packed boolean evidence mask without its byte padding."""
    return np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)


def load_checkpoint(path, model, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    metadata = checkpoint.get('model_metadata', {})
    expected = model.checkpoint_metadata
    for name in ['model_normalization', 'model_norm_group_counts']:
        if name in metadata and metadata[name] != expected.get(name):
            raise ValueError(f'Checkpoint {name}={metadata[name]} differs from model {expected.get(name)}')
    state = dict(state)
    state.setdefault('skip_log_gates', model.skip_log_gates.detach().clone())
    model.load_state_dict(state, strict=True)
    return checkpoint


def save_checkpoint(path, model, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    torch.save(dict(model_state_dict=model.state_dict(), model_metadata=model.checkpoint_metadata, **extra), temporary)
    temporary.replace(path)


def make_device(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f'Unavailable device: {name}')
    return device
