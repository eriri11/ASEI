"""Synthetic interface tests: no real adaptation or MedSAM inference is run."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.config import load_config, PROJECT_ROOT
from asei.data import SliceDataset, VolumeDataset, build_loader
from asei.models import UNet2D, EMATeacher
from asei.utils import load_checkpoint, save_checkpoint, load_npz
from asei.adaptation import (GLFP_DEFAULTS, TargetAugmentation,
    LabelBank, train_mt, valid_bank, seal_bank, data_digest)
from asei.glfp import (plan, feedback_plan, fuse, fuse_candidates, generate_labels,
    patches, roi_union, CandidateBuilder)
from asei.ecss import generate_sets, ECSSBank
from asei._losses import aligned_cached_targets, aligned_unknown_mask, masked_probability_mse
from asei.metrics import evaluate, dice_3d, voxel_assd
from asei.medsam import EmbeddingCache, MedSAM

torch.set_num_threads(2)


class TinyModel(torch.nn.Module):
    checkpoint_metadata = {'model_normalization': 'batchnorm'}

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(1, 5, 1)
        self.skip_log_gates = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    def forward(self, image):
        return self.head(image)


class FakeCache:
    image_key = staticmethod(EmbeddingCache.image_key)

    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value.cpu().numpy()


class FakeSAM:
    def __init__(self):
        self.encodes = 0
        self.prompts = 0

    def encode_batch(self, images):
        self.encodes += len(images)
        return torch.ones(len(images), 256, 64, 64)

    def probabilities(self, embedding, boxes, shape, batch_size=24):
        self.prompts += len(boxes)
        result = np.zeros((len(boxes), *shape), np.float32)
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            result[index, int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = 1.
        return result


class SeedModel(torch.nn.Module):
    def forward(self, image):
        logits = image.new_full((len(image), 5, 256, 256), -8.)
        logits[:, 0] = 8.
        logits[:, 0, 60:120, 60:120] = -8.
        logits[:, 2, 60:120, 60:120] = 8.
        return logits


class CoreTests(unittest.TestCase):
    def test_configs_and_project_paths(self):
        for name in ['source_ct', 'source_mr', 'ct_to_mr', 'mr_to_ct']:
            config = load_config(PROJECT_ROOT / f'configs/{name}.yaml')
            self.assertTrue(Path(config['data']['root']).is_absolute())
        ct = load_config(PROJECT_ROOT / 'configs/ct_to_mr.yaml')
        mr = load_config(PROJECT_ROOT / 'configs/mr_to_ct.yaml')
        self.assertEqual(ct['adaptation']['training']['max_steps'], 1000)
        self.assertEqual(ct['adaptation']['training']['learning_rate'], 1e-5)
        self.assertEqual(mr['adaptation']['training']['max_steps'], 1000)
        self.assertEqual(mr['adaptation']['training']['learning_rate'], 1e-4)
        self.assertIsNone(ct['adaptation']['training']['epochs'])
        self.assertIsNone(mr['adaptation']['training']['epochs'])
        self.assertEqual(ct['adaptation']['glfp']['thresholds'], [.95] * 4)
        self.assertEqual(ct['adaptation']['ecss']['loss_weight'], 1.)
        self.assertEqual(mr['adaptation']['ecss']['loss_weight'], 1.)
        self.assertEqual(ct['adaptation']['ecss']['radius'], 12)

    def test_unlabeled_loader_and_label_free_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'mr/train'
            directory.mkdir(parents=True)
            image = np.arange(3*32*32, dtype=np.float32).reshape(3, 32, 32)
            file = directory / 'case_slice_0003.npz'
            np.savez(file, image=image)
            dataset = SliceDataset(tmp, 'mr', with_labels=False)
            sample = dataset[0]
            self.assertNotIn('label', sample)
            self.assertEqual((sample['case_id'], sample['slice_id']), ('case', '3'))
            self.assertEqual(tuple(sample['image'].shape), (1, 32, 32))
            before = data_digest(dataset)
            np.savez(file, image=image, label=np.full_like(image, 255))
            self.assertEqual(data_digest(dataset), before)
            with self.assertRaises(ValueError):
                SliceDataset(tmp, 'mr', with_labels=True)[0]

    def test_checkpoint_and_ema_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for norm in ['batchnorm', 'groupnorm']:
                model = UNet2D(5, base_channels=4, normalization=norm)
                path = Path(tmp) / 'model.pt'
                save_checkpoint(path, model)
                copy = UNet2D(5, base_channels=4, normalization=norm)
                load_checkpoint(path, copy)
                self.assertEqual(set(model.state_dict()), set(copy.state_dict()))
            with self.assertRaises(ValueError):
                load_checkpoint(path, UNet2D(5, base_channels=4))
        student = UNet2D(5, base_channels=4)
        teacher = EMATeacher(student, .5)
        with torch.no_grad():
            student.input_block.block[1].running_mean.fill_(2)
            student.input_block.block[1].num_batches_tracked.fill_(7)
        teacher.update(student)
        self.assertTrue(torch.equal(teacher.teacher.input_block.block[1].running_mean, torch.ones(4)))
        self.assertEqual(int(teacher.teacher.input_block.block[1].num_batches_tracked), 7)

    def test_conflicts_ignore_and_myo_geometric_prior(self):
        masks = np.zeros((4, 32, 32), bool)
        masks[0, 2:8, 2:8] = True
        masks[2, 4:10, 4:10] = True
        probability = masks.astype(np.float32)
        pseudo, weight = fuse_candidates(masks, probability, np.ones((32, 32), np.float32))
        self.assertTrue((pseudo[4:8, 4:8] == 255).all())
        self.assertTrue((weight[4:8, 4:8] == 0).all())
        regions = np.zeros_like(masks)
        regions[0, 10:20, 10:20] = True
        core = regions.copy()
        candidates = CandidateBuilder(np.ones_like(probability), core, regions, 3).masks([.9]*4)
        self.assertFalse(candidates[0, 0, 0])
        self.assertTrue(candidates[0, 12, 12])

    def test_two_round_generation_and_duplicate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, sam = FakeCache(), FakeSAM()
            images = torch.linspace(0, 1, 256*256).reshape(1, 256, 256)
            samples = [dict(image=images, case_id='case', slice_id=str(i), index=i) for i in range(2)]
            loader = torch.utils.data.DataLoader(samples, batch_size=2)
            final, raw = generate_labels(SeedModel(), loader, deepcopy(GLFP_DEFAULTS), sam, cache, tmp, torch.device('cpu'))
            self.assertEqual(sam.encodes, 1)
            a, b = load_npz(final / '00000.npz'), load_npz(final / '00001.npz')
            self.assertTrue(np.array_equal(a['pseudo'], b['pseudo']))
            self.assertTrue((a['pseudo'] == 2).any())
            self.assertTrue((a['pseudo'] == 0).any())
            record = load_npz(raw / '00000.npz')
            self.assertLessEqual(len(record['local_boxes']), 6)
            conflict = np.unpackbits(record['global_local_conflict'])[:256*256].reshape(256,256).astype(bool)
            self.assertFalse((conflict & (a['weight'] > 0)).any())
            for box in record['extra_boxes']:
                self.assertLessEqual(max(box[2]-box[0], box[3]-box[1]), 24)
            seal_bank(final, 2)
            self.assertTrue(valid_bank(final, 2))
            np.savez(final / '00001.npz', pseudo=np.zeros((256, 256)), weight=np.ones((256, 256)))
            self.assertFalse(valid_bank(final, 2))

    def test_global_local_conflict_mask_records_both_directions(self):
        cores = np.zeros((4, 16, 16), bool)
        cores[0, 4:6, 4:6] = True
        regions = np.zeros_like(cores)
        regions[0, 2:12, 2:12] = True
        stable_bg = np.zeros((16, 16), bool)
        stable_bg[1, 5] = True
        boxes = np.array([[0, 0, 12, 12]])
        classes = np.array([0])
        mean = np.zeros((4, 16, 16), np.float32)
        mean[0] = 1.
        query = [(boxes, classes, np.zeros((1, 12, 12), np.float32), cores)]
        p, w, conflict = fuse(mean, cores, regions, np.ones((16, 16)), stable_bg,
                              query, deepcopy(GLFP_DEFAULTS), return_conflict=True)
        self.assertTrue(conflict[1, 5])  # Global positive, supported local negative.
        self.assertEqual(p[1, 5], 255)
        self.assertEqual(w[1, 5], 0)
        mean[:] = 0
        query = [(boxes, classes, np.ones((1, 12, 12), np.float32), cores)]
        p, w, conflict = fuse(mean, cores, regions, np.ones((16, 16)), stable_bg,
                              query, deepcopy(GLFP_DEFAULTS), return_conflict=True)
        self.assertTrue(conflict[4, 4])  # Local positive, global negative.
        self.assertFalse((w[conflict] > 0).any())
        self.assertTrue((p[conflict] == 255).all())

    def test_unknown_consistency_and_visible_supervision(self):
        p = torch.tensor([[[1, 255], [0, 2]]])
        w = torch.tensor([[[1., 0.], [0., 1.]]])
        visible = torch.tensor([[[True, True], [True, False]]])
        labels, weights = aligned_cached_targets(p, w, visible, 5)
        self.assertEqual(labels.tolist(), [[[1, 255], [255, 255]]])
        mask = aligned_unknown_mask(p, w, visible)
        self.assertEqual(mask.tolist(), [[[False, True], [True, False]]])
        student = torch.zeros(1, 5, 2, 2, requires_grad=True)
        teacher = torch.ones_like(student)
        loss = masked_probability_mse(student, teacher, mask)
        self.assertEqual(float(loss), 1.)
        loss.backward()
        self.assertEqual(float(student.grad[0, :, 1, 1].abs().sum()), 0.)

    def test_synthetic_stage2_updates_and_teacher_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'mr/train').mkdir(parents=True)
            (root / 'labels').mkdir()
            image = np.random.default_rng(42).random((3, 32, 32), dtype=np.float32)
            np.savez(root / 'mr/train/case_slice_0.npz', image=image)
            p = np.full((32, 32), 255, np.uint8)
            p[5:20, 5:20] = 2
            w = (p != 255).astype(np.float16)
            np.savez(root / 'labels/00000.npz', pseudo=p, weight=w)
            config = dict(experiment=dict(seed=42,target_domain='mr'),
                data=dict(root=str(root),batch_size=1,num_workers=0,center_slice=1),runtime=dict(device='cpu'),
                model=dict(base_channels=4,normalization='batchnorm',norm_groups=8))
            student = TinyModel()
            before = student.head.weight.detach().clone()
            teacher, count = train_mt(student, config, dict(epochs=1,max_steps=1,learning_rate=.001),
                root / 'stage2', torch.device('cpu'), LabelBank(root / 'labels', 1))
            self.assertEqual(count, 1)
            self.assertFalse(torch.equal(before, student.head.weight))
            self.assertFalse(student.skip_log_gates.requires_grad)
            self.assertIsNone(student.skip_log_gates.grad)
            self.assertTrue((root / 'stage2/teacher_final.pt').is_file())
            self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))

    def test_pipeline_reuses_labels_for_stage2_tuning(self):
        from asei.adaptation import adapt_target
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/mr/train').mkdir(parents=True)
            image = np.linspace(0, 1, 3*256*256, dtype=np.float32).reshape(3, 256, 256)
            np.savez(root / 'data/mr/train/case_slice_0.npz', image=image)
            initial = TinyModel()
            with torch.no_grad():
                initial.head.weight.zero_()
                initial.head.bias.fill_(-8.)
                initial.head.bias[2] = 8.
            save_checkpoint(root / 'initial.pt', initial)
            (root / 'sam.pth').write_bytes(b'mock checkpoint, no MedSAM model loaded')
            config = load_config(PROJECT_ROOT / 'configs/ct_to_mr.yaml')
            config['runtime']['device'] = 'cpu'
            config['data'].update(root=str(root / 'data'), batch_size=1, num_workers=0)
            config['source_checkpoint'] = str(root / 'initial.pt')
            config['output']['root'] = str(root / 'output')
            config['adaptation'].update(medsam_checkpoint=str(root / 'sam.pth'), embedding_cache_root=str(root / 'features'))
            config['adaptation']['warmup']['enabled'] = False
            sam = FakeSAM()
            budgets = []
            set_banks = []

            def fake_training(student, cfg, settings, output, device, bank=None, teacher=None, warmup=False, **kwargs):
                budgets.append(settings['max_steps'])
                set_banks.append(kwargs.get('ecss_bank'))
                return deepcopy(student), 1

            with patch('asei.adaptation.build_model', side_effect=lambda cfg:TinyModel()), \
                 patch('asei.adaptation.train_mt', side_effect=fake_training), \
                 patch('asei.medsam.MedSAM', return_value=sam) as constructor, \
                 patch('asei.ecss.generate_sets', wraps=generate_sets) as sets:
                adapt_target(config)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(sam.encodes, 1)
                first = (root / 'output/preparation.json').read_text()
                config['adaptation']['training'].update(learning_rate=2e-5, max_steps=200)
                config['adaptation']['ecss']['loss_weight'] = 3.
                adapt_target(config)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(sets.call_count, 1)
                self.assertEqual(first, (root / 'output/preparation.json').read_text())
                config['adaptation']['glfp']['thresholds'][1] = .96
                adapt_target(config)
                self.assertEqual(constructor.call_count, 2)
                # Shared-evidence policy is part of the ECSS identity.
                self.assertEqual(sets.call_count, 2)
                self.assertEqual(sam.encodes, 1)  # Reuse image embedding after prompt-policy change.
                config['adaptation']['ecss']['radius'] = 8
                adapt_target(config)
                self.assertEqual(sets.call_count, 3)
                self.assertEqual(constructor.call_count, 2)
                config['adaptation']['ecss']['enabled'] = False
                adapt_target(config)
                self.assertEqual(sets.call_count, 3)
                self.assertEqual(constructor.call_count, 2)
                self.assertEqual(budgets, [1000, 200, 200, 200, 200])
                self.assertTrue(all(isinstance(bank, ECSSBank) for bank in set_banks[:-1]))
                self.assertIsNone(set_banks[-1])
                self.assertTrue((root / 'output/teacher_final.pt').is_file())

    def test_volume_reader_and_assd(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifests').mkdir()
            image = np.arange(8*8*2, dtype=np.float32).reshape(8, 8, 2)
            label = np.zeros((8, 8, 2), np.int16)
            label[1:4, 2:5] = 3
            nib.save(nib.Nifti1Image(image, np.eye(4)), root / 'image.nii.gz')
            nib.save(nib.Nifti1Image(label, np.eye(4)), root / 'label.nii.gz')
            (root / 'manifests/test_cases.csv').write_text('domain,split,case_id,image_path,label_path,shape\nct,test,1,image.nii.gz,label.nii.gz,8x8x2\n')
            dataset = VolumeDataset(root, 'ct')
            self.assertEqual(len(dataset), 2)
            self.assertTrue(np.array_equal(dataset[0]['label'].numpy(), np.flip(label[:, :, 0], (0, 1))))
        mask = label == 3
        self.assertEqual(dice_3d(mask, mask), 1.)
        self.assertEqual(voxel_assd(mask, mask), 0.)
        self.assertIsNone(voxel_assd(mask, np.zeros_like(mask)))


if __name__ == '__main__':
    unittest.main()
