"""GLFP: stable seeds, constrained local boxes and configurable feedback rounds."""
import numpy as np
from scipy import ndimage
import torch


def local_round_count(options):
    """Total local prompting rounds, including the initial core-based queries."""
    count = options.get('num_rounds', 2)
    if type(count) is not int or count < 1:
        raise ValueError('glfp.num_rounds must be a positive integer')
    return count if options['feedback']['enabled'] else 1


def cached_local_queries(raw, cores):
    """Restore cached local queries and their per-round anchors."""
    from .utils import unpack
    queries = [(raw['local_boxes'], raw['local_classes'], raw['local_probabilities'], cores)]
    history_keys = ('extra_rounds', 'feedback_rounds', 'feedback_anchors')
    if any(key in raw for key in history_keys):
        if not all(key in raw for key in history_keys):
            raise ValueError('Incomplete local round history')
        query_rounds = np.asarray(raw['extra_rounds'])
        rounds = np.asarray(raw['feedback_rounds'])
        if (query_rounds.shape != (len(raw['extra_boxes']),)
                or rounds.ndim != 1
                or not np.issubdtype(query_rounds.dtype, np.integer)
                or not np.issubdtype(rounds.dtype, np.integer)
                or (rounds < 2).any() or (np.diff(rounds) <= 0).any()
                or not np.isin(query_rounds, rounds).all()):
            raise ValueError('Invalid local round history')
        anchors = unpack(raw['feedback_anchors'], (len(rounds), *cores.shape))
        for round_id, anchor in zip(rounds, anchors):
            selected = query_rounds == round_id
            if selected.any():
                queries.append((raw['extra_boxes'][selected], raw['extra_classes'][selected],
                                raw['extra_probabilities'][selected], anchor))
    elif len(raw['extra_boxes']):
        queries.append((raw['extra_boxes'], raw['extra_classes'], raw['extra_probabilities'],
                        unpack(raw['extra_anchors'], cores.shape)))
    return queries


def mask_box(mask: np.ndarray):
    """Return xyxy with exclusive right/bottom edges, or None if absent."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('A box requires a 2D mask')
    y, x = np.where(mask)
    return (float(x.min()), float(y.min()), float(x.max()+1), float(y.max()+1)) if len(x) else None

def expanded_box(box, factor, height, width):
    if not np.isfinite(factor) or factor < 1:
        raise ValueError('Box expansion must be finite and >=1')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Box lies outside the image')
    cx, cy = (x0+x1)/2, (y0+y1)/2
    bw, bh = (x1-x0)*factor, (y1-y0)*factor
    return (max(0., cx-bw/2), max(0., cy-bh/2), min(float(width), cx+bw/2), min(float(height), cy+bh/2))

def largest_component(mask):
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, bool)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    return labels == int(areas.argmax())

def prompt_regions(probabilities, confidence=.9, minimum_seed_pixels=16):
    prediction = probabilities.argmax(0)
    regions = np.stack([largest_component(prediction == c) for c in range(1, len(probabilities))])
    seeds = regions & (probabilities[1:] >= confidence)
    groups = []
    for c in range(1, len(probabilities)):
        if int(seeds[c - 1].sum()) < minimum_seed_pixels:
            seeds[c - 1] = False
            continue
        groups.append({'class_id': c, 'anchor': seeds[c - 1], 'box': mask_box(regions[c - 1])})
    return groups, seeds, regions

def boxes_for_groups(groups, scales, shape):
    return [expanded_box(g['box'], scale, *shape) for g in groups for scale in scales]

def average_responses(groups, decoded, shape, scales=3):
    assert len(decoded) == len(groups) * scales
    mean = np.zeros((4, *shape), np.float32)
    raw_foreground = np.zeros(shape, bool)
    for index, group in enumerate(groups):
        values = np.asarray(decoded[index * scales:(index + 1) * scales], np.float32)
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        mean[group['class_id'] - 1] = values.mean(0)
        raw_foreground |= (values >= .5).any(0)
    return mean, raw_foreground

class CandidateBuilder:
    def __init__(self, probabilities, seeds, regions, myo_radius=3):
        if probabilities.shape != seeds.shape or seeds.shape != regions.shape or probabilities.shape[0] != 4:
            raise ValueError('Expected aligned four-organ maps')
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError('Invalid mean probability')
        if type(myo_radius) is not int or myo_radius < 0:
            raise ValueError('MYO radius must be a nonnegative pixel count')
        self.probabilities = probabilities
        self.seeds = seeds.astype(bool)
        if myo_radius:
            y, x = np.ogrid[-myo_radius:myo_radius + 1, -myo_radius:myo_radius + 1]
            self.myo_allowed = ndimage.binary_dilation(regions[0], structure=x*x + y*y <= myo_radius*myo_radius)
        else:
            self.myo_allowed = regions[0].astype(bool)

    def mask(self, organ_index, threshold):
        if not .5 <= threshold <= 1:
            raise ValueError('Candidate probability must be between0.5 and1')
        anchor = self.seeds[organ_index]
        if not anchor.any():
            return np.zeros_like(anchor)
        mask = self.probabilities[organ_index] >= threshold
        if organ_index == 0:
            mask &= self.myo_allowed
        components, count = ndimage.label(mask)
        overlaps = np.bincount(components[anchor], minlength=count + 1)
        overlaps[0] = 0
        return components == int(overlaps.argmax()) if overlaps.max(initial=0) else np.zeros_like(anchor)

    def masks(self, thresholds):
        if np.isscalar(thresholds):
            thresholds = [thresholds] * 4
        if len(thresholds) != 4:
            raise ValueError('Need a threshold for each organ')
        return np.stack([self.mask(c, float(t)) for c, t in enumerate(thresholds)])

def fuse_candidates(masks, probabilities, bg_weights):
    votes = masks.sum(0)
    foreground = votes == 1
    background = (votes == 0) & (bg_weights > 0)
    labels = np.full(votes.shape, 255, np.uint8)
    weight = np.zeros(votes.shape, np.float32)
    cls = masks.argmax(0)
    labels[foreground] = cls[foreground] + 1
    selected_probability = np.take_along_axis(probabilities, cls[None], axis=0)[0]
    weight[foreground] = selected_probability[foreground]
    labels[background] = 0
    weight[background] = bg_weights[background]
    return labels, weight.astype(np.float16)

LOCAL_POLICY=dict(initial_side=12,step=4,maximum_side=24,maximum_boxes=6,minimum_seeds=16,minimum_target_fraction=.6,maximum_foreign_core_fraction=.05)

def constrained_local_boxes(probability, core, region, foreign, policy=None):
    """Grow boxes along image axes under spatial constraints."""
    settings = {**LOCAL_POLICY, **(policy or {})}
    height, width = core.shape
    side = settings['initial_side']
    if min(height, width) < side:
        raise ValueError('Image smaller than initial local box')
    if not (probability.shape == core.shape == region.shape == foreign.shape):
        raise ValueError('Unaligned prompt maps')
    covered = np.zeros_like(core)
    centers, boxes, records = [], [], []
    def integral(mask):
        return np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ri, ci, fi = map(integral, (region, core, foreign))
    def count(table, box):
        x0, y0, x1, y1 = box
        return int(table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0])
    def valid(box):
        area = (box[2]-box[0])*(box[3]-box[1])
        return (count(ci, box) >= settings['minimum_seeds']
                and count(ri, box)/area >= settings['minimum_target_fraction']
                and count(fi, box)/area <= settings['maximum_foreign_core_fraction']
                and max(box[2]-box[0], box[3]-box[1]) <= settings['maximum_side'])
    ys, xs = np.where(core)
    x0 = np.clip(xs-side//2, 0, width-side)
    y0 = np.clip(ys-side//2, 0, height-side)
    def vector_count(table):
        return table[y0+side,x0+side]-table[y0,x0+side]-table[y0+side,x0]+table[y0,x0]
    eligible = ((vector_count(ci) >= settings['minimum_seeds'])
                & (vector_count(ri) >= settings['minimum_target_fraction']*side*side)
                & (vector_count(fi) <= settings['maximum_foreign_core_fraction']*side*side))
    while len(boxes) < settings['maximum_boxes']:
        ids = np.flatnonzero(eligible & ~covered[ys, xs])
        if not len(ids):
            break
        if centers:
            distance = np.min(np.stack([(ys[ids]-cy)**2+(xs[ids]-cx)**2 for cy,cx in centers]), axis=0)
            ids = ids[distance == distance.max()]
        best = ids[np.argmax(probability[ys[ids], xs[ids]])]
        box = (int(x0[best]), int(y0[best]), int(x0[best]+side), int(y0[best]+side))
        trajectory = [list(box)]
        new_integral = integral(core & ~covered)
        while True:
            left, top, right, bottom = box
            step = settings['step']
            choices = [(max(0,left-step),top,right,bottom), (left,max(0,top-step),right,bottom),
                       (left,top,min(width,right+step),bottom), (left,top,right,min(height,bottom+step))]
            choices = [candidate for candidate in choices if candidate != box and valid(candidate)]
            if not choices:
                break
            def rank(candidate):
                area = (candidate[2]-candidate[0])*(candidate[3]-candidate[1])
                return count(new_integral,candidate), count(ci,candidate), count(ri,candidate)/area, area
            box = max(choices, key=rank)
            trajectory.append(list(box))
        new_seeds = count(new_integral, box)
        if new_seeds < settings['minimum_seeds']:
            eligible[best] = False
            continue
        left, top, right, bottom = box
        area = (right-left)*(bottom-top)
        records.append(dict(box=list(box), center_xy=[int(xs[best]),int(ys[best])],
                            growth_trajectory=trajectory, new_core_pixels=new_seeds,
                            target_fraction=count(ri,box)/area, foreign_core_fraction=count(fi,box)/area))
        boxes.append(box)
        centers.append((int(ys[best]),int(xs[best])))
        covered[top:bottom,left:right] = True
    return boxes, records

def roi_union(boxes,shape):
    out=np.zeros(shape,bool)
    for x0,y0,x1,y1 in boxes:out[int(y0):int(y1),int(x0):int(x1)]=True
    return out

def plan(probability,stable,options):
    groups,_,regions=prompt_regions(probability,options['seed_confidence'],options['minimum_seed_pixels'])
    cores=regions&stable[None]
    for c in range(4):
        if cores[c].sum()<options['minimum_seed_pixels']:cores[c]=False
    boxes=[];classes=[]
    for c in range(4):
        if not cores[c].any():continue
        foreign=np.any(np.delete(cores,c,axis=0),axis=0)
        selected,_=constrained_local_boxes(probability[c+1],cores[c],regions[c],foreign,options['local'])
        boxes.extend(selected);classes.extend([c]*len(selected))
    global_boxes=boxes_for_groups(groups,options['global_scales'],probability.shape[-2:])
    return groups,regions,cores,np.asarray(boxes,np.int32).reshape(-1,4),np.asarray(classes,np.uint8),global_boxes

def feedback_plan(p,w,probability,cores,regions,old_boxes,old_classes,options):
    masks=np.zeros_like(cores);boxes=[];classes=[];records=[]
    for c in range(4):
        if not cores[c].any():continue
        components,n=ndimage.label((p==c+1)&(w>0));ids=np.unique(components[cores[c]]);ids=ids[ids>0]
        masks[c]=np.isin(components,ids)&(p==c+1)&(w>0)
        foreign=np.any(np.delete(cores,c,axis=0),axis=0);previous=old_boxes[old_classes==c]
        selected,details=feedback_boxes(w,masks[c],regions[c],foreign,previous,options['local'],options['feedback'])
        if options['feedback']['match_MT_budget']:
            reference,_=feedback_boxes(probability[c+1],cores[c],regions[c],foreign,previous,options['local'],options['feedback'])
            n=min(len(selected),len(reference));selected=selected[:n];details=details[:n]
        boxes.extend(selected);classes.extend([c]*len(selected));records.extend([dict(organ=c,**item) for item in details])
    return masks,np.asarray(boxes,np.int32).reshape(-1,4),np.asarray(classes,np.uint8),records

def extract_evidence(mean,cores,regions,stable_background,queries,options,global_classes=None):
    """One shared extraction for GLFP hard fusion and ECSS class sets.

    Missing global queries are unobserved. Local negatives use original cores
    in all rounds and are cancelled only by accepted strict local positives.
    """
    local=np.zeros_like(cores);weights=np.zeros_like(mean);negative=np.zeros_like(cores);rawfg=np.zeros(cores.shape[-2:],bool)
    local_seen=np.zeros_like(cores)
    for boxes,classes,patches,anchors in queries:
        for box,c in zip(boxes,classes):
            x0,y0,x1,y1=map(int,box);local_seen[c,y0:y1,x0:x1]=True
        m,w,n,fg=local_evidence(boxes,classes,patches,anchors,regions,stable_background,options)
        # Negative support uses the initial stable cores in every round.
        if anchors is not cores:
            n=np.zeros_like(cores)
            for box,c,patch in zip(boxes,classes,patches):
                x0,y0,x1,y1=map(int,box);q=patch[:y1-y0,:x1-x0]
                foreign=np.any(np.delete(cores,c,axis=0),axis=0)
                n[c,y0:y1,x0:x1]|=(q<=options['negative_threshold'])&(stable_background|foreign)[y0:y1,x0:x1]
        local|=m;weights=np.maximum(weights,w);negative|=n;rawfg|=fg
    negative&=~local
    global_masks=CandidateBuilder(mean,cores,regions,myo_radius=options['myo_radius']).masks(options['thresholds'])
    observed=np.ones(4,bool) if global_classes is None else np.asarray([
        (np.asarray(global_classes)==c).any() for c in range(4)])
    return dict(global_positive=global_masks,
        global_negative=observed[:,None,None]&(mean<=options['negative_threshold']),
        local_positive=local,local_negative=negative,local_weights=weights,
        raw_foreground=rawfg,global_observed=observed,local_observed=local_seen)


def fuse(mean,cores,regions,bg_weight,stable_background,queries,options,return_conflict=False,
         return_evidence=False,global_classes=None):
    evidence=extract_evidence(mean,cores,regions,stable_background,queries,options,global_classes)
    local=evidence['local_positive'];weights=evidence['local_weights']
    negative=evidence['local_negative'];rawfg=evidence['raw_foreground']
    global_masks=evidence['global_positive']
    masks=global_masks.copy();probabilities=mean.copy();masks[[0,2]]=local[[0,2]];probabilities[[0,2]]=weights[[0,2]]
    conflict=(global_masks&negative).any(0)|(local[[0,2]]&(mean[[0,2]]<=options['negative_threshold'])).any(0)
    bg=np.where(rawfg|conflict,0,bg_weight).astype(np.float32)
    if not cores.any():bg[:]=0
    p,w=fuse_candidates(masks,probabilities,bg);p[conflict]=255;w[conflict]=0
    if return_evidence:return p,w,conflict,evidence
    return (p,w,conflict) if return_conflict else (p,w)

def local_evidence(boxes, classes, patches, cores, regions, stable_background, options):
    shape = cores.shape[-2:]
    masks = np.zeros_like(cores)
    weights = np.zeros((4,*shape),np.float32)
    negatives = np.zeros_like(cores)
    raw_foreground = np.zeros(shape,bool)
    radius=options['myo_radius'];y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
    myo_allowed = ndimage.binary_dilation(regions[0],structure=x*x+y*y<=radius*radius)
    for box,c,patch in zip(boxes,classes,patches):
        left,top,right,bottom = map(int,box)
        q = patch[:bottom-top,:right-left]
        anchors = cores[c,top:bottom,left:right]
        if q.shape != anchors.shape or not anchors.any():
            raise ValueError('Invalid local response geometry or anchor')
        proposed = q >= options['thresholds'][c]
        if c == 0:
            proposed &= myo_allowed[top:bottom,left:right]
        labels,count = ndimage.label(proposed)
        overlaps = np.bincount(labels[anchors],minlength=count+1); overlaps[0]=0
        accepted = labels == int(overlaps.argmax()) if overlaps.max(initial=0) else np.zeros_like(anchors)
        masks[c,top:bottom,left:right] |= accepted
        weights[c,top:bottom,left:right] = np.maximum(weights[c,top:bottom,left:right],np.where(accepted,q,0))
        foreign = np.any(np.delete(cores,c,axis=0),axis=0)
        supported_negative = (stable_background | foreign)[top:bottom,left:right]
        negatives[c,top:bottom,left:right] |= (q <= options['negative_threshold']) & supported_negative
        raw_foreground[top:bottom,left:right] |= q >= options['background_exclusion_threshold']
    # Conflicting local prompts do not become negative evidence.
    negatives &= ~masks
    return masks,weights,negatives,raw_foreground

def feedback_boxes(confidence,anchors,region,foreign,old_boxes,local,feedback):
    """Vectorized constrained growth, novel ROI instead of novel seed count."""
    h,w=anchors.shape;side=local['initial_side'];step=local['step'];minimum=local['minimum_seeds'];cap=feedback['candidate_center_cap']
    if min(h,w)<side or anchors.sum()<minimum:return [],[]
    def integral(mask):return np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
    ai,ri,fi=map(integral,(anchors,region,foreign))
    def count(table,b):return table[b[:,3],b[:,2]]-table[b[:,1],b[:,2]]-table[b[:,3],b[:,0]]+table[b[:,1],b[:,0]]
    def valid(b):
        area=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
        return ((count(ai,b)>=minimum)&(count(ri,b)/area>=local['minimum_target_fraction'])&(count(fi,b)/area<=local['maximum_foreign_core_fraction'])
            &((b[:,2]-b[:,0])<=local['maximum_side'])&((b[:,3]-b[:,1])<=local['maximum_side']))
    ys,xs=np.where(anchors)
    left=np.clip(xs-side//2,0,w-side);top=np.clip(ys-side//2,0,h-side)
    initial=np.stack([left,top,left+side,top+side],1).astype(np.int32)
    ids=np.flatnonzero(valid(initial))
    if not len(ids):return [],[]
    if len(ids)>cap:
        best=ids[np.argsort(-confidence[ys[ids],xs[ids]],kind='stable')[:feedback['high_confidence_centers']]]
        spread=ids[np.linspace(0,len(ids)-1,cap-feedback['high_confidence_centers'],dtype=int)]
        ids=np.unique(np.concatenate([best,spread]))
    initial=initial[ids];ys=ys[ids];xs=xs[ids]
    covered=roi_union(old_boxes,(h,w));selected=[];records=[]
    for _ in range(feedback['maximum_extra_boxes']):
        ni=integral(~covered);boxes=initial.copy()
        for _growth in range(2*local['maximum_side']//step+2):
            proposals=np.repeat(boxes[None],4,axis=0)
            proposals[0,:,0]=np.maximum(0,proposals[0,:,0]-step)
            proposals[1,:,1]=np.maximum(0,proposals[1,:,1]-step)
            proposals[2,:,2]=np.minimum(w,proposals[2,:,2]+step)
            proposals[3,:,3]=np.minimum(h,proposals[3,:,3]+step)
            choice=np.full(len(boxes),-1,int)
            best_values=[np.full(len(boxes),-np.inf) for _ in range(4)]
            for direction,p in enumerate(proposals):
                okay=valid(p)&np.any(p!=boxes,axis=1)
                area=(p[:,2]-p[:,0])*(p[:,3]-p[:,1])
                values=[count(ni,p),count(ai,p),count(ri,p)/area,area]
                better=np.zeros(len(boxes),bool);equal=np.ones(len(boxes),bool)
                for v,best in zip(values,best_values):better|=equal&(v>best);equal&=v==best
                update=okay&better
                for v,best in zip(values,best_values):best[update]=v[update]
                choice[update]=direction
            moving=choice>=0
            if not moving.any():break
            rows=np.flatnonzero(moving);boxes[rows]=proposals[choice[rows],rows]
        area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);novel=count(ni,boxes)
        eligible=valid(boxes)&(novel/area>=feedback['minimum_new_query_fraction'])
        if not eligible.any():break
        candidates=np.flatnonzero(eligible)
        anchor_count=count(ai,boxes);target=count(ri,boxes)/area
        best=max(candidates,key=lambda i:(int(novel[i]),int(anchor_count[i]),float(confidence[ys[i],xs[i]]),float(target[i]),int(area[i])))
        box=tuple(map(int,boxes[best]));selected.append(box)
        records.append(dict(box=list(box),center_xy=[int(xs[best]),int(ys[best])],
            anchor_pixels=int(anchor_count[best]),new_query_pixels=int(novel[best]),
            new_query_fraction=float(novel[best]/area[best]),target_fraction=float(target[best]),
            foreign_core_fraction=float(count(fi,boxes[best:best+1])[0]/area[best]),
            center_confidence=float(confidence[ys[best],xs[best]])))
        x0,y0,x1,y1=box;covered[y0:y1,x0:x1]=True
    return selected,records

def patches(boxes,probabilities,side):
    out=np.zeros((len(boxes),side,side),np.float32)
    for j,(x0,y0,x1,y1) in enumerate(boxes):out[j,:y1-y0,:x1-x0]=probabilities[j][y0:y1,x0:x1]
    return out


@torch.inference_mode()
def generate_labels(model, loader, options, sam, features, output, device,
                    encoder_batch_size=2, decoder_batch_size=24):
    """Offline prompting for up to num_rounds; the frozen teacher never updates."""
    import shutil
    from pathlib import Path
    import torch
    from .utils import write_json
    num_rounds = local_round_count(options)
    output = Path(output)
    first, final, rawdir = [output / name for name in ['round1', f'round{num_rounds}', 'responses']]
    round_paths = {1: first, num_rounds: final}
    for path in [first, final, rawdir]:
        path.mkdir(parents=True, exist_ok=True)
    model.eval()
    cursor, seen = 0, {}
    stats = dict(images=0, encoded_images=0, embedding_hits=0, decoded_boxes=0, duplicates=0,
                 local_boxes=[0]*4, feedback_boxes=[0]*4, num_rounds=num_rounds)
    for batch in loader:
        if 'label' in batch:
            raise RuntimeError('Target labels must not enter GLFP')
        host = batch['image']
        images = host.to(device)
        if images.shape[1:] != (1, 256, 256):
            raise ValueError('GLFP expects single-channel 256x256 images')
        probability = model(images).softmax(1)
        confidence, prediction = probability.max(1)
        stable = confidence >= options['seed_confidence']
        for gamma in options['weak_gammas']:
            q = model(images.clamp(0, 1).pow(gamma)).softmax(1)
            qc, ql = q.max(1)
            stable &= (ql == prediction) & (qc >= options['seed_confidence'])
        probabilities, stable = probability.cpu().numpy(), stable.cpu().numpy()
        keys = [features.image_key(image.numpy()) for image in host]
        jobs, encoded, missing, pending = {}, {}, [], set()
        for j, key in enumerate(keys):
            if key in seen or key in pending:
                continue
            pending.add(key)
            jobs[j] = plan(probabilities[j], stable[j], options)
            if jobs[j][-1] or len(jobs[j][3]):
                value = features.get(key)
                if value is None:
                    missing.append(j)
                else:
                    encoded[j] = torch.from_numpy(value).to(device)
                    stats['embedding_hits'] += 1
        for start in range(0, len(missing), encoder_batch_size):
            ids = missing[start:start+encoder_batch_size]
            values = sam.encode_batch(images[ids])
            for k, j in enumerate(ids):
                encoded[j] = values[k:k+1]
                features.put(keys[j], encoded[j])
            stats['encoded_images'] += len(ids)
        for j, key in enumerate(keys):
            name = f'{cursor:05d}.npz'
            cursor += 1
            if key in seen:
                for path in [*round_paths.values(), rawdir]:
                    original = path / f'{seen[key]:05d}.npz'
                    if original.is_file():
                        shutil.copyfile(original, path / name)
                stats['duplicates'] += 1
            else:
                groups, regions, cores, boxes, classes, global_boxes = jobs[j]
                size = (256, 256)
                all_boxes = list(dict.fromkeys(tuple(map(float, b)) for b in [*global_boxes, *boxes]))
                decoded = sam.probabilities(encoded.get(j), all_boxes, size, batch_size=decoder_batch_size)
                stats['decoded_boxes'] += len(all_boxes)
                lookup = dict(zip(all_boxes, decoded))
                mean, _ = average_responses(groups, [lookup[tuple(b)] for b in global_boxes], size,
                                             scales=len(options['global_scales']))
                rawfg = np.zeros(size, bool)
                for box in global_boxes:
                    rawfg |= lookup[tuple(box)] >= options['background_exclusion_threshold']
                stable_bg = stable[j] & (probabilities[j].argmax(0) == 0)
                bg = np.where(stable_bg & ~rawfg, probabilities[j, 0], 0).astype(np.float32) if groups and rawfg.any() else np.zeros(size, np.float32)
                side = options['local']['maximum_side']
                local_patches = patches(boxes, [lookup[tuple(map(float, b))] for b in boxes], side)
                queries = [(boxes, classes, local_patches, cores)]
                global_classes = np.asarray([g['class_id']-1 for g in groups for _ in options['global_scales']], np.uint8)
                y, w, conflict, evidence = fuse(mean, cores, regions, bg, stable_bg, queries, options,
                    return_evidence=True, global_classes=global_classes)
                np.savez_compressed(first / name, pseudo=y, weight=w)
                anchors = np.zeros_like(cores)
                previous_boxes, previous_classes = boxes, classes
                feedback_rounds, feedback_anchors, query_rounds = [], [], []
                completed_rounds = 1
                for round_id in range(2, num_rounds + 1):
                    current_anchors, new_boxes, new_classes, _ = feedback_plan(
                        y, w, probabilities[j], cores, regions, previous_boxes, previous_classes, options)
                    feedback_rounds.append(round_id)
                    feedback_anchors.append(current_anchors)
                    anchors |= current_anchors
                    # No new queries means neither evidence nor anchors can change.
                    if not len(new_boxes):
                        break
                    new_q = sam.probabilities(encoded.get(j), new_boxes.tolist(), size,
                                              batch_size=decoder_batch_size)
                    stats['decoded_boxes'] += len(new_boxes)
                    new_patches = patches(new_boxes, new_q, side)
                    queries.append((new_boxes, new_classes, new_patches, current_anchors))
                    query_rounds.extend([round_id] * len(new_boxes))
                    previous_boxes = np.concatenate([previous_boxes, new_boxes])
                    previous_classes = np.concatenate([previous_classes, new_classes])
                    y, w, conflict, evidence = fuse(mean, cores, regions, bg, stable_bg, queries, options,
                        return_evidence=True, global_classes=global_classes)
                    completed_rounds = round_id
                    round_path = output / f'round{round_id}'
                    round_path.mkdir(parents=True, exist_ok=True)
                    round_paths[round_id] = round_path
                    np.savez_compressed(round_path / name, pseudo=y, weight=w)
                extra_boxes = np.concatenate([q[0] for q in queries[1:]]) if len(queries) > 1 else np.empty((0, 4), np.int32)
                extra_classes = np.concatenate([q[1] for q in queries[1:]]) if len(queries) > 1 else np.empty(0, np.uint8)
                extra_patches = np.concatenate([q[2] for q in queries[1:]]) if len(queries) > 1 else np.empty((0, side, side), np.float32)
                anchor_history = np.stack(feedback_anchors) if feedback_anchors else np.empty((0, *cores.shape), bool)
                np.savez_compressed(final / name, pseudo=y, weight=w)
                np.savez_compressed(rawdir / name, cores=np.packbits(cores), regions=np.packbits(regions),
                    stable_background=np.packbits(stable_bg), mt_bg_probability=probabilities[j, 0],
                    image_sha256=np.asarray(key), mean_probability=mean, bg_weight=bg,
                    global_boxes=np.asarray(global_boxes, np.float32).reshape(-1, 4),
                    global_classes=np.asarray([g['class_id']-1 for g in groups for _ in options['global_scales']], np.uint8),
                    local_boxes=boxes, local_classes=classes, local_probabilities=local_patches,
                    extra_boxes=extra_boxes, extra_classes=extra_classes, extra_probabilities=extra_patches,
                    extra_anchors=np.packbits(anchors), global_local_conflict=np.packbits(conflict),
                    extra_rounds=np.asarray(query_rounds, np.int32),
                    feedback_rounds=np.asarray(feedback_rounds, np.int32),
                    feedback_anchors=np.packbits(anchor_history),
                    num_rounds=np.asarray(num_rounds), completed_rounds=np.asarray(completed_rounds),
                    **{key:np.packbits(evidence[key]) for key in (
                        "global_positive","global_negative","local_positive","local_negative")})
                seen[key] = cursor-1
                stats['local_boxes'] = (np.asarray(stats['local_boxes']) + np.bincount(classes, minlength=4)).tolist()
                stats['feedback_boxes'] = (np.asarray(stats['feedback_boxes']) + np.bincount(extra_classes, minlength=4)).tolist()
            stats['images'] += 1
        write_json(output / 'progress.json', stats)
        print(f'[GLFP] {cursor}/{len(loader.dataset)} slices; {stats["decoded_boxes"]} prompts', flush=True)
    write_json(output / 'complete.json', dict(complete=True, statistics=stats, target_labels_used=False))
    return final, rawdir
