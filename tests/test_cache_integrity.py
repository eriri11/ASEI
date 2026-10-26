"""Current GLFP/ECSS cache identity, integrity, and reuse checks."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from test_core import FakeSAM, TinyModel
from asei import adaptation, ecss, medsam
from asei.config import PROJECT_ROOT, load_config
from asei.utils import read_json, save_checkpoint, sha_file, write_json


@pytest.mark.parametrize('change', [None, 'labels', 'responses', 'sets', 'policy', 'rounds', 'single_round'])
def test_bank_reuse_requires_matching_identity_and_integrity(tmp_path, monkeypatch, change):
    """Compatible banks avoid new inference; corruption and policy edits do not."""
    images = tmp_path / 'data/mr/train'
    images.mkdir(parents=True)
    np.savez(images / 'image.npz', image=np.linspace(0, 1, 256*256, dtype=np.float32).reshape(1, 256, 256))
    initial = TinyModel()
    with torch.no_grad():
        initial.head.weight.zero_()
        initial.head.bias.fill_(-8.)
        initial.head.bias[2] = 8.
    checkpoint = tmp_path / 'initial.pt'
    save_checkpoint(checkpoint, initial)
    checkpoint_hash = sha_file(checkpoint)
    sam_path = tmp_path / 'sam.pth'
    sam_path.write_bytes(b'fake checkpoint; no real SAM inference')
    config = load_config(PROJECT_ROOT / 'configs/ct_to_mr.yaml')
    config['runtime']['device'] = 'cpu'
    config['data'].update(root=str(tmp_path / 'data'), batch_size=1, num_workers=0)
    config['source_checkpoint'] = str(checkpoint)
    config['output']['root'] = str(tmp_path / 'output')
    config['adaptation'].update(medsam_checkpoint=str(sam_path),
        embedding_cache_root=str(tmp_path / 'features'))
    config['adaptation']['warmup']['enabled'] = False

    sam = FakeSAM()
    calls = dict(sam=0, sets=0)
    original_generate_sets = ecss.generate_sets

    def fake_sam(*args, **kwargs):
        calls['sam'] += 1
        return sam

    def counted_sets(*args, **kwargs):
        calls['sets'] += 1
        return original_generate_sets(*args, **kwargs)

    def fake_training(student, *args, **kwargs):
        return deepcopy(student), 1

    monkeypatch.setattr(adaptation, 'build_model', lambda cfg: TinyModel())
    monkeypatch.setattr(adaptation, 'train_mt', fake_training)
    monkeypatch.setattr(medsam, 'MedSAM', fake_sam)
    monkeypatch.setattr(ecss, 'generate_sets', counted_sets)
    original_config = deepcopy(config)
    adaptation.adapt_target(config)
    assert calls == dict(sam=1, sets=1)
    preparation_path = tmp_path / 'output/preparation.json'
    prepared = read_json(preparation_path)
    prompt_bank = Path(prepared['labels']).parent
    set_bank = Path(prepared['sets'])
    assert prompt_bank.parent.name == 'glfp'
    assert set_bank.parent.name == 'ecss'
    original_bank_hashes = {p: sha_file(p) for directory in [prompt_bank, set_bank]
                            for p in directory.rglob('*.npz')}

    if change in ['labels', 'responses', 'sets']:
        directory = {'labels': prompt_bank / 'round2',
                     'responses': prompt_bank / 'responses', 'sets': set_bank}[change]
        marker = read_json(directory / 'bank.json')
        marker['sha256'] = '0'*64
        write_json(directory / 'bank.json', marker)
    if change == 'policy':
        config['adaptation']['glfp']['thresholds'][1] = .96
    if change in ['rounds', 'single_round']:
        config['adaptation']['glfp']['num_rounds'] = 3 if change == 'rounds' else 1

    adaptation.adapt_target(config)
    after = read_json(preparation_path)
    policy_changed = change in ['policy', 'rounds', 'single_round']
    assert calls['sam'] == (2 if change in ['labels', 'responses'] or policy_changed else 1)
    assert calls['sets'] == (2 if change == 'sets' or policy_changed else 1)
    assert sam.encodes == 1  # Image embeddings survive bank invalidation.
    # Corruption regenerates the same content-addressed bank; policy changes
    # select a new identity. The inference call counters above distinguish reuse.
    assert (Path(after['labels']).parent == prompt_bank) == (not policy_changed)
    assert Path(after['labels']).name == ('round3' if change == 'rounds' else 'round1' if change == 'single_round' else 'round2')
    # Corrupted ECSS banks regenerate under the same content-addressed identity.
    assert (Path(after['sets']) == set_bank) == (not policy_changed)
    assert adaptation.valid_bank(set_bank, 1)
    assert all(sha_file(path) == digest for path, digest in original_bank_hashes.items())
    assert sha_file(checkpoint) == checkpoint_hash
    if change in [None, 'labels', 'responses', 'sets']:
        assert config == original_config  # Cache handling must not mutate settings.
    saved = read_json(tmp_path / 'output/summary.json')
    assert saved['framework'] == 'ASEI'
    assert saved['modules'] == ['GLFP', 'ECSS']
    saved_config = load_config(tmp_path / 'output/config.yaml')
    assert saved_config == config
