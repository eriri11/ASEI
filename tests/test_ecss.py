"""Shared GLFP evidence, geometric candidates and immutable hard supervision."""
from pathlib import Path
import sys,json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asei.ecss import construct_sets,ECSS_DEFAULTS,generate_sets,validate_options
from asei.glfp import fuse
from asei.adaptation import GLFP_DEFAULTS


def sample(mean=.3, queried=(0,1,2,3), packed=False):
    shape=(20,20);cores=np.zeros((4,*shape),bool);cores[0,9:11,9:11]=True
    regions=cores.copy();stable=np.zeros(shape,bool)
    probability=np.zeros((4,*shape),np.float32);probability[0]=mean
    q=np.full((1,6,6),.7,np.float32);q[:,2:4,2:4]=1.
    boxes=np.array([[7,7,13,13]]);classes=np.array([0])
    p,w,d,e=fuse(probability,cores,regions,np.zeros(shape,np.float32),stable,
        [(boxes,classes,q,cores)],GLFP_DEFAULTS,return_evidence=True,global_classes=np.array(queried))
    raw=dict(cores=np.packbits(cores),regions=np.packbits(regions),stable_background=np.packbits(stable),
        bg_weight=np.zeros(shape,np.float32),mean_probability=probability,global_classes=np.array(queried),
        local_boxes=boxes,local_classes=classes,local_probabilities=q,
        extra_boxes=np.empty((0,4),int),extra_classes=np.empty(0,int),extra_probabilities=np.empty((0,6,6)),
        extra_anchors=np.packbits(np.zeros_like(cores)),global_local_conflict=np.packbits(d),image_sha256=np.asarray('synthetic'))
    if packed:raw.update({k:np.packbits(e[k]) for k in ('global_positive','global_negative','local_positive','local_negative')})
    return p,w,raw,e


def construct(p,w,raw):return construct_sets(p,w,raw,glfp_options=GLFP_DEFAULTS)


def test_four_masks_identical_to_hard_fusion_and_packed_cache():
    p,w,raw,e=sample();result=construct(p,w,raw)
    for key in ('global_positive','global_negative','local_positive','local_negative'):
        np.testing.assert_array_equal(result[key],e[key]);raw[key]=np.packbits(e[key])
    packed=construct_sets(p,w,raw)
    for key in ('set_bits','set_weight'):np.testing.assert_array_equal(packed[key],result[key])
    assert not result['local_positive'][0,8,9]  # .7 is NOT GLFP's accepted strict evidence.


def test_geometric_candidate_does_not_need_positive_at_own_pixel_and_missing_is_unknown():
    p,w,r,e=sample(queried=(0,1,3));x=construct(p,w,r)
    assert x['extension'][8,9]
    assert x['set_bits'][8,9]==(1|2|8)  # BG, inconclusive MYO, unobserved LV.
    assert not (x['global_positive']|x['local_positive'])[:,8,9].any()
    assert not x['global_negative'][2].any() and not x['local_negative'][:,0,0].any()


def test_hard_labels_weights_and_mass_are_preserved():
    p,w,r,e=sample();before_p=p.copy();before_w=w.copy();x=construct(p,w,r)
    np.testing.assert_array_equal(x['set_bits'][w>0],np.left_shift(1,p[w>0]))
    assert not (x['extension']&(w>0)).any()
    np.testing.assert_array_equal(p,before_p);np.testing.assert_array_equal(w,before_w)
    assert x['set_weight'].sum(dtype=np.float64)<=.1*float(w.sum())+1e-6
    assert (x['set_bits'][x['extension']]&1).all()


def test_original_conflict_abstains_and_bg_singletons_are_skipped():
    p,w,r,e=sample(mean=.05);x=construct(p,w,r)
    assert x['conflict'][9,9] and not x['extension'][9,9]
    assert not x['extension'].any()  # no reliable reference and all negatives outside it.


def test_full_set_stays_unknown():
    p,w,r,e=sample(queried=(0,));x=construct(p,w,r)
    assert x['geometric'][8,9] and not x['extension'].any()


def test_local_negatives_cancelled_by_strict_accepted_positives_only():
    cores=np.zeros((4,8,8),bool);cores[0,2,2]=True;regions=cores.copy();stable=np.ones((8,8),bool)
    boxes=np.array([[0,0,8,8],[0,0,8,8]]);classes=np.array([0,0]);q=np.ones((2,8,8),np.float32)*.05
    q[0,2,2]=1.;q[1]=.7;q[1,2,2]=1.
    _,_,_,e=fuse(np.ones((4,8,8),np.float32)*.3,cores,regions,np.zeros((8,8)),stable,
        [(boxes,classes,q,cores)],GLFP_DEFAULTS,return_evidence=True)
    assert e['local_negative'][0,2,3]  # weak .7 cannot cancel the other query's supported negative.
    assert not e['local_negative'][0,2,2] and e['local_positive'][0,2,2]


def test_native_generation_matches_constructor_without_touching_hard_bank(tmp_path):
    p,w,raw,e=sample();labels,responses,sets=[tmp_path/k for k in ('labels','responses','sets')]
    labels.mkdir();responses.mkdir();np.savez_compressed(labels/'00000.npz',pseudo=p,weight=w)
    np.savez_compressed(responses/'00000.npz',**raw);before=(labels/'00000.npz').read_bytes()
    generate_sets(labels,responses,sets,1,ECSS_DEFAULTS,GLFP_DEFAULTS);expected=construct(p,w,raw)
    with np.load(sets/'00000.npz') as saved:
        for key in ('set_bits','set_weight','observed','original_foreground_mass'):
            np.testing.assert_array_equal(saved[key],expected[key])
    assert (labels/'00000.npz').read_bytes()==before
    metadata=json.loads((sets/'generation.json').read_text())
    assert not metadata['target_labels_used'] and metadata['encoder_calls']==metadata['decoder_calls']==0


def test_historical_cache_requires_matching_policy():
    p,w,r,e=sample()
    with pytest.raises(ValueError,match='glfp_options'):construct_sets(p,w,r)
    wrong={**GLFP_DEFAULTS,'thresholds':[.5]*4}
    with pytest.raises(ValueError,match='reconstruct'):construct_sets(p,w,r,glfp_options=wrong)


@pytest.mark.parametrize('key,value',[('radius',0),('radius',1.5),('positive_threshold',.5),
    ('negative_threshold',.1),('pixel_weight',0),('maximum_mass_fraction',0),
    ('loss_weight',float('nan')),('enabled','yes')])
def test_invalid_or_independent_evidence_options_fail_explicitly(key,value):
    with pytest.raises(ValueError):validate_options(dict(ECSS_DEFAULTS,**{key:value}))
