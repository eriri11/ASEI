"""ECSS: evidence-constrained set supervision; GLFP targets stay fixed.

The identical four GLFP masks constrain geometric unknown neighbors.
Only observed, unopposed negatives exclude classes; BG stays possible.
"""
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

from .utils import load_npz, unpack, write_json


ECSS_DEFAULTS = dict(enabled=True, radius=12,
                     pixel_weight=.1, maximum_mass_fraction=.1, loss_weight=1.)


def validate_options(options):
    if set(options) != set(ECSS_DEFAULTS):
        raise ValueError('ECSS options must match ECSS_DEFAULTS')
    if type(options['enabled']) is not bool:
        raise ValueError('ecss.enabled must be boolean')
    if type(options['radius']) is not int or options['radius'] <= 0:
        raise ValueError('ecss.radius must be a positive integer')
    for key in ('pixel_weight', 'maximum_mass_fraction'):
        if not 0 < options[key] <= 1:
            raise ValueError(f'ecss.{key} must be in (0,1]')
    if not np.isfinite(options['loss_weight']) or options['loss_weight'] < 0:
        raise ValueError('ecss.loss_weight must be finite and nonnegative')


def linked_components(mask, anchor):
    """Keep whole four-connected components touching a same-class reference."""
    component, _ = ndimage.label(mask)
    ids = np.unique(component[anchor])
    ids = ids[ids != 0]
    return np.isin(component, ids) if len(ids) else np.zeros_like(mask, dtype=bool)


def construct_sets(pseudo, weight, raw, options=None, glfp_options=None):
    """Construct sets from exactly the four masks used by GLFP hard fusion.

    Canonical current caches store those masks directly. Historical response
    caches reconstruct them through GLFP's *same* extractor and verify the
    original hard targets/conflicts; no independent evidence thresholds exist.
    """
    from .glfp import fuse, cached_local_queries
    settings={**ECSS_DEFAULTS, **(options or {})}
    validate_options(settings)
    if pseudo.ndim!=2 or weight.shape!=pseudo.shape:
        raise ValueError('Expected aligned GLFP labels and weights')
    if not np.isin(pseudo,[0,1,2,3,4,255]).all() or not np.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Invalid GLFP hard targets')
    shape=pseudo.shape;cores=unpack(raw['cores'],(4,*shape))
    unknown=(pseudo==255)|(weight<=0)
    conflict=unpack(raw['global_local_conflict'],shape)
    if (conflict&~unknown).any():raise ValueError('GLFP conflicts must remain unknown')
    global_seen=np.array([(raw['global_classes']==c).any() for c in range(4)])
    local_seen=np.zeros_like(cores)
    for prefix in ('local','extra'):
        boxes,classes,patches=[raw[prefix+'_'+k] for k in ('boxes','classes','probabilities')]
        if not len(boxes)==len(classes)==len(patches):raise ValueError('Misaligned local queries')
        for box,c,q in zip(boxes,classes,patches):
            x0,y0,x1,y1=map(int,box)
            if not 0<=int(c)<4 or not (0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]):
                raise ValueError('Invalid query geometry')
            local_seen[c,y0:y1,x0:x1]=True
    keys=('global_positive','global_negative','local_positive','local_negative')
    if all(k in raw for k in keys):
        evidence={k:unpack(raw[k],(4,*shape)) for k in keys}
        if (evidence['global_negative']&~global_seen[:,None,None]).any():
            raise ValueError('Unobserved global query cannot provide negative evidence')
        if any((evidence[k]&~local_seen).any() for k in ('local_positive','local_negative')):
            raise ValueError('Local evidence outside query ROI')
    else:
        if glfp_options is None:
            raise ValueError('Historical responses require the original glfp_options')
        stable=unpack(raw['stable_background'],shape)
        regions=unpack(raw['regions'],(4,*shape))
        queries=cached_local_queries(raw,cores)
        p,w,d,evidence=fuse(raw['mean_probability'],cores,regions,raw['bg_weight'],stable,
            queries,glfp_options,return_evidence=True,global_classes=raw['global_classes'])
        if not all(np.array_equal(a,b) for a,b in ((p,pseudo),(w,weight),(d,conflict))):
            raise ValueError('GLFP evidence policy does not reconstruct original hard bank')
    gp,gn,lp,ln=[evidence[k] for k in keys]
    reference=np.stack([linked_components((pseudo==c+1)&(weight>0),cores[c]) for c in range(4)])
    near=np.stack([unknown&(ndimage.distance_transform_edt(~r)<=settings['radius'])
                   if r.any() else np.zeros(shape,bool) for r in reference])
    geometric=near.any(0)&~conflict
    rejection=(gn|ln)&~(gp|lp)
    allowed=np.full(shape,31,np.uint8)
    for c in range(4):allowed[rejection[c]]&=np.uint8(31^(1<<(c+1)))
    size=np.asarray([int(v).bit_count() for v in range(32)],np.uint8)[allowed]
    extension=geometric&(size>=2)&(size<=4)
    bits=np.zeros(shape,np.uint8)
    bits[~unknown]=np.left_shift(1,pseudo[~unknown]).astype(np.uint8)
    bits[extension]=allowed[extension]
    foreground=(pseudo>0)&(pseudo<5)&(weight>0)
    original_mass=float(weight[foreground].astype(np.float64).sum())
    set_weight=extension.astype(np.float32)*settings['pixel_weight']
    set_weight*=min(1.,settings['maximum_mass_fraction']*original_mass/
                    max(1e-8,float(set_weight.sum(dtype=np.float64))))
    return dict(set_bits=bits,set_weight=set_weight,original_foreground_mass=np.asarray(original_mass),
        reference=reference,near=near,geometric=geometric,extension=extension,recruitment=geometric,
        global_positive=gp,global_negative=gn,local_positive=lp,local_negative=ln,
        global_observed=global_seen,local_observed=local_seen,rejection=rejection,
        conflict=conflict,unknown=unknown,observed=global_seen|local_seen.reshape(4,-1).any(1))


def generate_sets(labels, responses, output, count, options, glfp_options=None):
    """Build shared-evidence ECSS without image/GT reads or MedSAM calls."""
    validate_options(options)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stats=dict(images=0,set_pixels=0,original_conflict_pixels=0,set_mass=0.)
    for index in range(count):
        name=f'{index:05d}.npz';original=load_npz(Path(labels)/name)
        raw=load_npz(Path(responses)/name)
        result=construct_sets(original['pseudo'],original['weight'],raw,options,glfp_options)
        np.savez_compressed(output/name,**{k:result[k] for k in
            ('set_bits','set_weight','observed','original_foreground_mass')},image_sha256=raw['image_sha256'])
        stats['images']+=1;stats['set_pixels']+=int(result['extension'].sum())
        stats['original_conflict_pixels']+=int(result['conflict'].sum())
        stats['set_mass']+=float(result['set_weight'].sum(dtype=np.float64))
        if (index+1)%512==0 or index+1==count:
            print(f'[ECSS shared GLFP evidence] {index+1}/{count} slices',flush=True)
    write_json(output/'generation.json',dict(complete=True,module='ECSS',statistics=stats,
        evidence_mode='exact_shared_GLFP_masks',target_labels_used=False,
        original_hard_targets_unchanged=True,encoder_calls=0,decoder_calls=0))
    return output


class ECSSBank:
    def __init__(self, directory, count):
        self.directory, self.count = Path(directory), count

    def batch(self, indices, device):
        bits, weights = [], []
        for index in indices.tolist():
            if not 0 <= index < self.count:
                raise ValueError('ECSS bank index out of range')
            value = load_npz(self.directory/f'{index:05d}.npz')
            bits.append(value['set_bits'])
            weights.append(value['set_weight'].astype(np.float32))
        return (torch.from_numpy(np.stack(bits)).long().to(device),
                torch.from_numpy(np.stack(weights)).to(device))


def align_sets(bits, weights, original_unknown, visible, spatial=None, phase=None, sign=None):
    """Use the MT spatial grid; exclude pixels interpolating different class sets."""
    membership = torch.stack([((bits >> c) & 1).float() for c in range(5)], 1)
    channels = torch.cat([membership*weights[:, None], weights[:, None], visible[:, None].float()], 1)
    valid = torch.ones_like(original_unknown)
    if spatial is not None:
        channels = spatial.warp_like(channels, phase, sign)
        valid = spatial.valid_like(channels, phase, sign)[None].expand_as(original_unknown)
    mass = channels[:, 5]
    fraction = channels[:, :5]/mass[:, None].clamp_min(1e-8)
    homogeneous = ((fraction <= 1e-6) | (fraction >= 1-1e-6)).all(1)
    allowed = fraction >= .5
    size = allowed.sum(1)
    valid = valid & original_unknown & (channels[:, 6] >= 1-1e-6) & (mass > 1e-8)
    valid = valid & homogeneous & allowed[:, 0] & (size >= 2) & (size <= 4)
    aligned_bits = sum(allowed[:, c].long()*(1 << c) for c in range(5))
    return aligned_bits.detach(), (mass*valid).detach(), valid


def cap_mass(weights, original_labels, original_weights, maximum_fraction=.1):
    """Cap added supervision per image using the original aligned foreground mass."""
    if not 0 < maximum_fraction <= 1:
        raise ValueError('ECSS mass fraction must be in (0,1]')
    foreground = (original_labels > 0) & (original_labels < 5) & (original_weights > 0)
    original_mass = (original_weights.detach()*foreground).sum((1, 2))
    added_mass = weights.sum((1, 2))
    scale = (maximum_fraction*original_mass/added_mass.clamp_min(1e-8)).clamp(max=1)
    return (weights*scale[:, None, None]).detach(), original_mass.detach()


def set_loss(logits, bits, weights, original_foreground_mass):
    """-log(sum allowed probabilities), normalized by original foreground mass."""
    active = weights > 0
    membership = torch.stack([((bits >> c) & 1).bool() for c in range(5)], 1)
    membership = membership | ~active[:, None]
    pixel_loss = -torch.logsumexp(logits.log_softmax(1).masked_fill(~membership, -torch.inf), dim=1)
    return (pixel_loss*weights).sum()/original_foreground_mass.sum().clamp_min(1e-8)
