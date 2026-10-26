"""Round count, feedback geometry, and exact evidence/cache reconstruction."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_core import FakeCache, FakeSAM, SeedModel
from asei import glfp
from asei.adaptation import GLFP_DEFAULTS, validate_options
from asei.ecss import ECSS_DEFAULTS, construct_sets, linked_components
from asei.utils import load_npz, unpack


class LocalSeedModel(SeedModel):
    """Put the synthetic organ in MYO, whose hard labels use local responses."""
    def forward(self, image):
        return super().forward(image)[:, [0, 2, 1, 3, 4]]


def generate(output, rounds, model=None, duplicates=1, feedback=True):
    options = deepcopy(GLFP_DEFAULTS)
    options['num_rounds'] = rounds
    options['local']['maximum_boxes'] = 1
    options['feedback'].update(maximum_extra_boxes=1, enabled=feedback)
    image = torch.linspace(0, 1, 256*256).reshape(1, 256, 256)
    loader = torch.utils.data.DataLoader(
        [dict(image=image, index=i) for i in range(duplicates)], batch_size=duplicates)
    sam = FakeSAM()
    final, responses = glfp.generate_labels(model or LocalSeedModel(), loader, options,
        sam, FakeCache(), output, torch.device('cpu'))
    return final, responses, options, sam


@pytest.mark.parametrize('rounds', [1, 2, 3])
def test_total_round_count_and_global_embedding_reuse(tmp_path, rounds):
    final, responses, _, sam = generate(tmp_path, rounds)
    raw = load_npz(responses/'00000.npz')
    assert final.name == f'round{rounds}'
    assert int(raw['num_rounds']) == int(raw['completed_rounds']) == rounds
    np.testing.assert_array_equal(raw['extra_rounds'], np.arange(2, rounds+1))
    assert len(raw['global_boxes']) == 3
    assert sam.encodes == 1
    assert sam.prompts == 3 + rounds  # Three global boxes and one local box per round.
    if rounds == 1:
        assert len(raw['extra_boxes']) == 0
        assert not unpack(raw['extra_anchors'], (4, 256, 256)).any()


def test_historical_disable_switch_overrides_round_count(tmp_path):
    final, responses, _, sam = generate(tmp_path, 4, feedback=False)
    raw = load_npz(responses/'00000.npz')
    assert final.name == 'round1' and int(raw['completed_rounds']) == 1
    assert sam.prompts == 4 and len(raw['extra_boxes']) == 0


@pytest.fixture
def three_round_bank(tmp_path):
    return tmp_path, generate(tmp_path, 3, duplicates=2)


def test_feedback_uses_previous_fusion_and_all_previous_rois(three_round_bank):
    root, (_, responses, options, _) = three_round_bank
    raw = load_npz(responses/'00000.npz')
    cores = unpack(raw['cores'], (4, 256, 256))
    anchors = unpack(raw['feedback_anchors'], (2, 4, 256, 256))
    covered = glfp.roi_union(raw['local_boxes'], (256, 256))
    for index, round_id in enumerate(raw['feedback_rounds']):
        previous = load_npz(root/f'round{round_id-1}'/'00000.npz')
        for c in range(4):
            expected = linked_components(
                (previous['pseudo'] == c+1) & (previous['weight'] > 0), cores[c])
            np.testing.assert_array_equal(anchors[index, c], expected)
        for x0, y0, x1, y1 in raw['extra_boxes'][raw['extra_rounds'] == round_id]:
            novel = (~covered[y0:y1, x0:x1]).mean()
            assert novel >= options['feedback']['minimum_new_query_fraction']
            covered[y0:y1, x0:x1] = True
    assert not np.array_equal(anchors[0], anchors[1])


def test_round_history_reconstructs_identical_hard_labels_and_ecss(three_round_bank):
    _, (final, responses, options, _) = three_round_bank
    raw = load_npz(responses/'00000.npz')
    target = load_npz(final/'00000.npz')
    direct = construct_sets(target['pseudo'], target['weight'], raw)
    cores = unpack(raw['cores'], (4, 256, 256))
    queries = glfp.cached_local_queries(raw, cores)
    assert len(queries) == 3
    p, w, d, evidence = glfp.fuse(raw['mean_probability'], cores,
        unpack(raw['regions'], cores.shape), raw['bg_weight'],
        unpack(raw['stable_background'], (256, 256)), queries, options,
        return_evidence=True, global_classes=raw['global_classes'])
    np.testing.assert_array_equal(p, target['pseudo'])
    np.testing.assert_array_equal(w, target['weight'])
    np.testing.assert_array_equal(d, direct['conflict'])
    for key in ('global_positive', 'global_negative', 'local_positive', 'local_negative'):
        np.testing.assert_array_equal(evidence[key], direct[key])
        del raw[key]
    reconstructed = construct_sets(target['pseudo'], target['weight'], raw, glfp_options=options)
    for key in ('set_bits', 'set_weight', 'extension'):
        np.testing.assert_array_equal(direct[key], reconstructed[key])


def test_duplicate_images_reuse_every_saved_round(three_round_bank):
    root, (_, responses, _, sam) = three_round_bank
    assert sam.encodes == 1 and sam.prompts == 6
    for directory in [responses, *(root/f'round{r}' for r in (1, 2, 3))]:
        assert (directory/'00000.npz').read_bytes() == (directory/'00001.npz').read_bytes()


def test_stops_when_no_new_boxes_are_available(tmp_path, monkeypatch):
    original = glfp.feedback_plan
    calls = []

    def no_feedback(*args, **kwargs):
        calls.append(1)
        anchors, _, _, _ = original(*args, **kwargs)
        return anchors, np.empty((0, 4), np.int32), np.empty(0, np.uint8), []

    monkeypatch.setattr(glfp, 'feedback_plan', no_feedback)
    final, responses, _, sam = generate(tmp_path, 5)
    raw = load_npz(responses/'00000.npz')
    assert final.name == 'round5' and int(raw['completed_rounds']) == 1
    assert len(calls) == 1 and sam.prompts == 4
    original = load_npz(tmp_path/'round1'/'00000.npz')
    for key, value in load_npz(final/'00000.npz').items():
        np.testing.assert_array_equal(value, original[key])


@pytest.mark.parametrize('rounds', [0, -1, True, 2.0, '3', None])
def test_invalid_round_counts_fail_before_generation(tmp_path, rounds):
    options = dict(GLFP_DEFAULTS, num_rounds=rounds)
    with pytest.raises(ValueError, match='num_rounds'):
        validate_options(options, ECSS_DEFAULTS)
    with pytest.raises(ValueError, match='num_rounds'):
        generate(tmp_path, rounds)
    assert not list(tmp_path.iterdir())
