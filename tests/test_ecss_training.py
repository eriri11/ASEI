"""Checks for preserving hard supervision, categorical alignment and weak mass."""
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asei.ecss import align_sets,cap_mass,set_loss,ECSSBank,ECSS_DEFAULTS,construct_sets
from asei.ecss import validate_options, generate_sets
from asei.utils import load_npz
from asei._spatial import SpatialContextAugmentation
from asei._losses import aligned_cached_targets,aligned_unknown_mask


@pytest.mark.parametrize('epochs,expected_steps', [(1, 1), (None, 3)])
def test_step_budget_repeats_target_loader_only_without_epoch_limit(tmp_path, monkeypatch, epochs, expected_steps):
    from asei import adaptation

    class Tiny(torch.nn.Module):
        checkpoint_metadata = {'model_normalization': 'batchnorm'}

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv2d(1, 5, 1)
            self.skip_log_gates = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

        def forward(self, image):
            return self.head(image)

    monkeypatch.setattr(adaptation, 'build_model', lambda cfg: Tiny())
    images = tmp_path / 'mr/train'
    images.mkdir(parents=True)
    np.savez(images / '0.npz', image=np.arange(16, dtype=np.float32).reshape(1, 4, 4))
    config = dict(experiment=dict(seed=42, target_domain='mr'), runtime=dict(device='cpu'),
                  data=dict(root=str(tmp_path), batch_size=1, num_workers=0, center_slice=0))
    settings = dict(epochs=epochs, max_steps=3, learning_rate=1e-3)
    _, steps = adaptation.train_mt(Tiny(), config, settings, tmp_path / 'training', torch.device('cpu'))
    assert steps == expected_steps
    records = [json.loads(line) for line in (tmp_path / 'training/losses.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, expected_steps + 1))
    checkpoint = torch.load(tmp_path / 'training/teacher_final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['steps'] == expected_steps


def test_known_and_corrupted_pixels_never_receive_ecss():
    bits=torch.full((1,2,2),3);weights=torch.ones((1,2,2))*.1
    unknown=torch.tensor([[[True,False],[True,True]]]);visible=torch.tensor([[[True,True],[False,True]]])
    _,weight,valid=align_sets(bits,weights,unknown,visible)
    assert valid.tolist()==[[[True,False],[False,True]]]
    assert float(weight[~valid].sum())==0


def test_mixed_sets_are_excluded_instead_of_becoming_a_single_class():
    class Mix:
        def warp_like(self,x,phase,sign):return (x+x.flip(-1))/2
        def valid_like(self,x,phase,sign):return torch.ones(x.shape[-2:],dtype=torch.bool)
    shape=(1,1,2);yes=torch.ones(shape,dtype=torch.bool)
    _,weight,valid=align_sets(torch.tensor([[[3,5]]]),torch.ones(shape)*.1,yes,yes,Mix(),0,1)
    assert not bool(valid.any()) and float(weight.sum())==0
    _,_,valid=align_sets(torch.tensor([[[3,3]]]),torch.ones(shape)*.1,yes,yes,Mix(),0,1)
    assert bool(valid.all())


def test_singleton_background_and_full_set_are_not_new_supervision():
    bits=torch.tensor([[[1,31,3,11]]]);yes=torch.ones_like(bits,dtype=torch.bool)
    _,_,valid=align_sets(bits,torch.ones_like(bits,dtype=torch.float32),yes,yes)
    assert valid.tolist()==[[[False,False,True,True]]]


def test_mass_budget_is_per_image_and_zero_foreground_means_zero_added_mass():
    labels=torch.tensor([[[1,1]],[[0,255]]]);old=torch.tensor([[[1.,1.]],[[1.,0.]]])
    capped,mass=cap_mass(torch.ones((2,1,2)),labels,old,.1)
    torch.testing.assert_close(mass,torch.tensor([2.,0.]))
    torch.testing.assert_close(capped.sum((1,2)),torch.tensor([.2,0.]))


def test_weak_weight_does_not_cancel_in_loss_normalization():
    logits=torch.zeros((1,5,1,1),requires_grad=True);bits=torch.full((1,1,1),3)
    a=set_loss(logits,bits,torch.full((1,1,1),.01),torch.tensor([1.]))
    b=set_loss(logits,bits,torch.full((1,1,1),.02),torch.tensor([1.]))
    torch.testing.assert_close(b,2*a)
    a.backward();assert logits.grad[0,0,0,0]<0 and logits.grad[0,1,0,0]<0
    assert logits.grad[0,2,0,0]>0


def test_empty_bank_has_finite_zero_loss_and_gradient():
    logits=torch.randn((1,5,2,2),requires_grad=True)
    loss=set_loss(logits,torch.zeros((1,2,2),dtype=torch.long),torch.zeros((1,2,2)),torch.tensor([0.]))
    loss.backward();assert float(loss)==0 and torch.isfinite(logits.grad).all() and float(logits.grad.abs().sum())==0


def test_spatial_auxiliary_does_not_modify_original_hard_targets():
    p=torch.full((2,128,128),255,dtype=torch.long);p[:,:,:50]=1
    w=(p!=255).float();visible=torch.ones_like(p,dtype=torch.bool)
    bits=torch.full_like(p,3);added=torch.zeros_like(w);added[:,:,50:65]=.1
    spatial=SpatialContextAugmentation({});_,_,_,phase,sign=spatial(torch.zeros((2,1,128,128)),torch.zeros((2,5,128,128)))
    labels,weights=aligned_cached_targets(p,w,visible,5,spatial,phase,sign)
    old_labels,old_weights=labels.clone(),weights.clone()
    unknown=aligned_unknown_mask(p,w,visible,spatial,phase,sign);old_unknown=unknown.clone()
    _,_,valid=align_sets(bits,added,unknown,visible,spatial,phase,sign)
    assert torch.equal(labels,old_labels) and torch.equal(weights,old_weights) and torch.equal(unknown,old_unknown)
    assert not bool((valid&(labels!=255)).any())


def test_training_ecss_and_zero_coefficient_preserve_first_hard_loss(tmp_path,monkeypatch):
    from asei import adaptation
    class Tiny(torch.nn.Module):
        checkpoint_metadata={'model_normalization':'batchnorm'}
        def __init__(self):
            super().__init__();self.head=torch.nn.Conv2d(1,5,1)
            self.skip_log_gates=torch.nn.Parameter(torch.zeros(4),requires_grad=False)
        def forward(self,x):return self.head(x)
    monkeypatch.setattr(adaptation,'build_model',lambda cfg:Tiny())
    images=tmp_path/'mr/train';images.mkdir(parents=True)
    label_dir=tmp_path/'labels';label_dir.mkdir();set_dir=tmp_path/'sets';set_dir.mkdir()
    p=np.tile(np.array([1,1,255,0],np.uint8),(4,1));w=(p!=255).astype(np.float16)
    for i in range(2):
        np.savez_compressed(images/f'{i}.npz',image=np.arange(16,dtype=np.float32).reshape(1,4,4),label=np.full((1,4,4),99))
        np.savez_compressed(label_dir/f'{i:05d}.npz',pseudo=p,weight=w)
        np.savez_compressed(set_dir/f'{i:05d}.npz',set_bits=np.full((4,4),3,np.uint8),set_weight=(p==255).astype(np.float32)*.1)
    config=dict(experiment=dict(seed=42,target_domain='mr'),runtime=dict(device='cpu'),
                data=dict(root=str(tmp_path),batch_size=2,num_workers=0,center_slice=0))
    settings=dict(epochs=1,max_steps=1,learning_rate=1e-3)
    initial=Tiny().state_dict();logs={}
    for mode in ['baseline','zero_coefficient','set']:
        student=Tiny();student.load_state_dict(initial)
        bank=None if mode=='baseline' else ECSSBank(set_dir,2)
        _,steps=adaptation.train_mt(student,config,settings,tmp_path/mode,torch.device('cpu'),
            bank=adaptation.LabelBank(label_dir,2),ecss_bank=bank,ecss_options=dict(ECSS_DEFAULTS,loss_weight=0. if mode=='zero_coefficient' else 1.))
        assert steps==1;logs[mode]=json.loads((tmp_path/mode/'losses.jsonl').read_text())
    assert logs['baseline']['ce']==logs['zero_coefficient']['ce']==logs['set']['ce']
    assert logs['baseline']['dice']==logs['zero_coefficient']['dice']==logs['set']['dice']
    assert logs['zero_coefficient']['ecss_pixels']==logs['set']['ecss_pixels']==8
    assert logs['baseline']['consistency_pixels']==8
    assert logs['zero_coefficient']['consistency_pixels']==logs['set']['consistency_pixels']==0
    assert logs['zero_coefficient']['ecss_loss_weight']==0 and logs['set']['ecss_ce']>0
    assert logs['zero_coefficient']['loss']<logs['set']['loss']
