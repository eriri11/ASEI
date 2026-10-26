"""Source holdout correctness and checkpoint selection on synthetic data."""
from pathlib import Path

import numpy as np
import pytest
import torch

from test_core import TinyModel
from asei import source
from asei.config import PROJECT_ROOT, load_config
from asei.utils import read_json


@pytest.fixture
def source_config(tmp_path):
    config = load_config(PROJECT_ROOT / 'configs/source_ct.yaml')
    config['runtime']['device'] = 'cpu'
    config['data'].update(root=str(tmp_path / 'data'), batch_size=4, num_workers=0)
    config['output']['root'] = str(tmp_path / 'run')
    config['source_training']['epochs'] = 1
    directory = tmp_path / 'data/ct/train'
    directory.mkdir(parents=True)
    for index in range(10):
        image = np.linspace(0, 1, 256*256, dtype=np.float32).reshape(1, 256, 256)
        label = np.full((256, 256), index % 5, dtype=np.int64)
        np.savez(directory / f'sample_{index:02d}.npz', image=image, label=label)
    return config


def test_holdout_is_disjoint_repeatable_and_unaugmented(source_config):
    state = torch.get_rng_state().clone()
    training, validation, record = source.build_source_partitions(source_config)
    assert torch.equal(state, torch.get_rng_state())
    assert (len(training), len(validation)) == (8, 2)
    assert not set(record['training_files']) & set(record['validation_files'])
    assert len(set(record['training_files']) | set(record['validation_files'])) == 10
    assert isinstance(training.dataset.transform, source.SourceAugmentation)
    assert validation.dataset.transform is None
    assert record == source.build_source_partitions(source_config)[2]
    source_config['source_training']['split_seed'] += 1
    assert record != source.build_source_partitions(source_config)[2]


@pytest.mark.parametrize('fraction', [0, 1, -0.2, float('nan')])
def test_invalid_holdout_fraction_is_rejected(source_config, fraction):
    source_config['source_training']['validation_fraction'] = fraction
    with pytest.raises(ValueError, match='validation_fraction'):
        source.build_source_partitions(source_config)


def test_test_set_cannot_be_used_for_checkpoint_selection(source_config):
    source_config['source_training']['validation_split'] = 'test'
    with pytest.raises(ValueError, match='train_holdout or val'):
        source.build_source_partitions(source_config)


def test_source_training_saves_partition_and_heldout_metrics(source_config, monkeypatch):
    requested = []
    build_dataset = source.build_dataset

    def checked_dataset(config, domain, split, *args):
        requested.append(split)
        assert split != 'test'
        return build_dataset(config, domain, split, *args)

    monkeypatch.setattr(source, 'build_dataset', checked_dataset)
    monkeypatch.setattr(source, 'build_model', lambda config: TinyModel())
    best = source.train_source(source_config)
    assert best.is_file()
    output = Path(source_config['output']['root'])
    record = read_json(output / 'source_split.json')
    assert len(record['training_files']) == 8 and len(record['validation_files']) == 2
    history = read_json(output / 'history.json')
    assert history[0]['validation']['num_samples'] == 2
    assert history[0]['validation']['metric_protocol'] == 'dataset_pixel_dice'
    assert requested == ['train', 'train']


def test_provided_validation_directory_is_supported(source_config):
    source_config['source_training']['validation_split'] = 'val'
    root = Path(source_config['data']['root'])
    (root / 'ct/val').mkdir()
    np.savez(root / 'ct/val/heldout.npz', image=np.zeros((1, 256, 256), np.float32),
             label=np.zeros((256, 256), np.int64))
    training, validation, record = source.build_source_partitions(source_config)
    assert (len(training), len(validation)) == (10, 1)
    assert record['strategy'] == 'provided_validation_split'
    assert validation.transform is None
