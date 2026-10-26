"""ASEI: MT warmup -> frozen GLFP evidence -> ECSS -> MT adaptation."""
from copy import deepcopy
from pathlib import Path
import fcntl
import json
import numpy as np
import torch
from .config import merge, save_config
from .ecss import ECSS_DEFAULTS, validate_options as validate_ecss_options
from .data import build_dataset, build_loader
from .models import build_model, EMATeacher
from ._spatial import SpatialContextAugmentation
from ._losses import (aligned_cached_targets, aligned_unknown_mask,
                      masked_balanced_ce_dice, masked_probability_mse)
from .utils import (seed_everything, load_checkpoint, save_checkpoint, sha_file,
                    json_digest, state_digest, read_json, write_json, load_npz, make_device)

GLFP_DEFAULTS = dict(num_rounds=2, seed_confidence=.9, minimum_seed_pixels=16, weak_gammas=[.9, 1.1],
    global_scales=[1., 1.1, 1.2], thresholds=[.95, .95, .95, .95], myo_radius=3,
    negative_threshold=.1, background_exclusion_threshold=.5,
    local=dict(initial_side=12, step=4, maximum_side=24, maximum_boxes=6, minimum_seeds=16,
               minimum_target_fraction=.6, maximum_foreign_core_fraction=.05),
    feedback=dict(enabled=True, maximum_extra_boxes=6, minimum_new_query_fraction=.25,
                  candidate_center_cap=512, high_confidence_centers=32, match_MT_budget=True))

class TargetAugmentation:
    """Fixed weak/strong photometric views; mark noisy/erased pixels."""
    def __init__(self, strong_warmup=False):
        self.strong_warmup = strong_warmup

    def __call__(self, sample):
        weak = sample['image']
        image = (weak.float().clamp(0, 1)*255).to(torch.uint8).clone()
        height, width = image.shape[-2:]
        corrupted = torch.zeros((height, width), dtype=torch.bool)
        strong = self.strong_warmup
        if torch.rand(()) < (.5 if strong else .25):
            is_salt = torch.rand(()) < .5
            fraction = .2 if is_salt else .8
            count = int(np.ceil((.01 if strong else .001)*height*width*3*fraction))
            rows = torch.randint(0, max(1, height-1), (count,))
            columns = torch.randint(0, max(1, width-1), (count,))
            # Salt uses value 1 in the uint8 image.
            image[:, rows, columns] = 1 if is_salt else 0
            corrupted[rows, columns] = True
        if torch.rand(()) < .5:
            gamma = torch.empty(()).uniform_(*([.5, 3.5] if strong else [.9, 1.1])).item()
            image = ((image.float()/255).pow(1./gamma)*255).to(torch.uint8)
        if torch.rand(()) < (.5 if strong else 0.):
            area = torch.empty(()).uniform_(.1, .3).item()*height*width
            aspect = torch.empty(()).uniform_(.3, .6).item()
            ew = min(width, max(1, int(np.sqrt(area/aspect))))
            eh = min(height, max(1, int(np.sqrt(area*aspect))))
            left = int(torch.randint(0, width-ew+1, ()).item())
            top = int(torch.randint(0, height-eh+1, ()).item())
            image[:, top:top+eh, left:left+ew] = int((torch.rand(())*255).item())
            corrupted[top:top+eh, left:left+ew] = True
        return {**sample, 'weak_image': weak, 'strong_image': (image.float()/255).contiguous(),
                'strong_corrupted_mask': corrupted}


class LabelBank:
    def __init__(self, directory, count):
        self.directory, self.count = Path(directory), count

    def batch(self, indices, device):
        labels, weights = [], []
        for index in indices.tolist():
            if not 0 <= index < self.count:
                raise ValueError('Label bank index out of range')
            value = load_npz(self.directory / f'{index:05d}.npz')
            labels.append(value['pseudo'])
            weights.append(value['weight'].astype(np.float32))
        return torch.from_numpy(np.stack(labels)).long().to(device), torch.from_numpy(np.stack(weights)).to(device)


def data_digest(dataset):
    # Hash images, not labels or NPZ compression, to allow GT-free cache reuse.
    from .medsam import EmbeddingCache
    import hashlib
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        sample = dataset[index]
        if 'label' in sample:
            raise RuntimeError('The target-train dataset must be unlabeled')
        digest.update(str((index, sample['case_id'], sample['slice_id'])).encode())
        digest.update(EmbeddingCache.image_key(sample['image'].numpy()).encode())
        if (index+1) % 2048 == 0 or index+1 == len(dataset):
            print(f'[ASEI data] fingerprint {index+1}/{len(dataset)} slices', flush=True)
    return digest.hexdigest()


def bank_digest(path, count):
    import hashlib
    digest = hashlib.sha256()
    for index in range(count):
        digest.update(sha_file(Path(path) / f'{index:05d}.npz').encode())
    return digest.hexdigest()


def valid_bank(path, count):
    marker = Path(path) / 'bank.json'
    if not marker.is_file():
        return False
    record = read_json(marker)
    try:
        return record['count'] == count and record['sha256'] == bank_digest(path, count)
    except (FileNotFoundError, KeyError):
        return False


def seal_bank(path, count):
    write_json(Path(path) / 'bank.json', dict(count=count, sha256=bank_digest(path, count), target_labels_used=False))


def validate_options(glfp, ecss):
    from .glfp import local_round_count
    local_round_count(glfp)
    if len(glfp['thresholds']) != 4 or any(not .5 <= t <= 1 for t in glfp['thresholds']):
        raise ValueError('GLFP requires four thresholds in [.5,1] in MYO,LA,LV,AA order')
    if not .5 <= glfp['seed_confidence'] <= 1 or not glfp['weak_gammas'] or not glfp['global_scales']:
        raise ValueError('Invalid GLFP confidence or prompt scales')
    if any(v <= 0 for v in glfp['weak_gammas']) or any(v < 1 for v in glfp['global_scales']):
        raise ValueError('Weak gammas must be positive and global scales >= 1')
    validate_ecss_options(ecss)
    local = glfp['local']
    if not 0 < local['initial_side'] <= local['maximum_side'] <= 256 or local['step'] <= 0:
        raise ValueError('Invalid local box size')


def train_mt(student, config, settings, output, device, bank=None, teacher=None, warmup=False,
             ecss_bank=None, ecss_options=None):
    """A fixed-budget segment; epochs=None runs until max_steps is reached."""
    seed_everything(config['experiment']['seed'])
    # Consume model-initialization RNG draws before sampling and augmentation.
    initialization = build_model(config)
    del initialization
    target = config['experiment']['target_domain']
    transform = TargetAugmentation(strong_warmup=warmup and target == 'ct')
    dataset = build_dataset(config, target, 'train', False, transform)
    loader = build_loader(config, dataset, True)
    epochs, max_steps = settings['epochs'], settings['max_steps']
    if (epochs is not None and epochs < 1) or max_steps < 1 or settings['learning_rate'] <= 0:
        raise ValueError('MT needs positive max_steps and learning_rate, and positive epochs or None')
    if not 0 <= settings.get('ema_decay', .98) < 1:
        raise ValueError('ema_decay must be in [0,1)')
    if min(settings.get('ce_weight', 1.), settings.get('dice_weight', 1.), settings.get('consistency_weight', 1.)) < 0:
        raise ValueError('Loss weights must be nonnegative')
    if epochs is None:
        if len(loader) == 0:
            raise ValueError('Step-based MT training requires a nonempty target loader')
        epochs = (max_steps + len(loader) - 1) // len(loader)
    planned = min(epochs*len(loader), max_steps)
    student.skip_log_gates.requires_grad_(False)
    ema = EMATeacher(student, settings.get('ema_decay', .98)).to(device)
    if teacher is not None:
        ema.teacher.load_state_dict(teacher.state_dict())
    optimizer = torch.optim.Adam(student.parameters(), lr=settings['learning_rate'],
        betas=tuple(settings.get('optimizer_betas', [.9, .99])), weight_decay=settings.get('weight_decay', 0.))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=settings.get('lr_gamma', .8))
    spatial = SpatialContextAugmentation({}) if target == 'ct' else None
    student.train()
    step = 0
    sampling = []
    if ecss_bank is not None:
        if bank is None or warmup:
            raise ValueError('ECSS requires the original Stage2 GLFP label bank')
        from .ecss import align_sets, cap_mass, set_loss
        ecss_options = merge(ECSS_DEFAULTS, ecss_options or {})
        validate_ecss_options(ecss_options)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'losses.jsonl').open('w') as log:
        for epoch in range(epochs):
            for offset, batch in enumerate(loader):
                if step >= planned:
                    break
                if 'label' in batch:
                    raise RuntimeError('Target labels must not enter MT training')
                if offset < 3:
                    import hashlib
                    sampling.append(dict(epoch=epoch, indices=batch['index'].tolist(),
                        views_sha256={key: hashlib.sha256(batch[key].numpy().tobytes()).hexdigest()
                                      for key in ['weak_image', 'strong_image']}))
                    write_json(output / 'sampling.json', sampling)
                weak, strong = batch['weak_image'].to(device), batch['strong_image'].to(device)
                with torch.no_grad():
                    probability = ema(weak).softmax(1)
                    phase = sign = None
                    spatial_valid = None
                    if spatial is not None:
                        strong, probability, spatial_valid, phase, sign = spatial(strong, probability)
                    if bank is not None:
                        pseudo, weights = bank.batch(batch['index'], device)
                        visible = ~batch['strong_corrupted_mask'].to(device).bool()
                        labels, aligned_weights = aligned_cached_targets(pseudo, weights, visible, 5, spatial, phase, sign)
                        mask = aligned_unknown_mask(pseudo, weights, visible, spatial, phase, sign)
                        if ecss_bank is not None:
                            set_bits, set_weights = ecss_bank.batch(batch['index'], device)
                            set_bits, set_weights, set_valid = align_sets(
                                set_bits, set_weights, mask, visible, spatial, phase, sign)
                            set_weights, original_fg_mass = cap_mass(
                                set_weights, labels, aligned_weights, ecss_options['maximum_mass_fraction'])
                            set_valid = set_valid & (set_weights > 0)
                            original_unknown_count = int(mask.sum())
                            mask = mask & ~set_valid
                optimizer.zero_grad(set_to_none=True)
                logits = student(strong)
                if bank is None:
                    ce = dice = logits.sum()*0
                    error = (logits.softmax(1)-probability).square().mean(1)
                    if spatial_valid is None:
                        consistency = error.mean()
                    else:
                        consistency = error[:, spatial_valid].mean() if spatial_valid.any() else error.sum()*0
                    active = True
                    supervised, unknown = 0, int(error.numel())
                else:
                    ce, dice = masked_balanced_ce_dice(logits, labels, aligned_weights)
                    consistency = masked_probability_mse(logits.softmax(1), probability, mask)
                    valid = (labels != 255) & (aligned_weights > 0)
                    active = bool(valid.any() or mask.any())
                    supervised, unknown = int(valid.sum()), int(mask.sum())
                loss = settings.get('ce_weight', 1.)*ce + settings.get('dice_weight', 1.)*dice + settings.get('consistency_weight', 1.)*consistency
                if ecss_bank is not None:
                    added_set = set_loss(logits, set_bits, set_weights, original_fg_mass)
                    loss = loss + ecss_options['loss_weight']*added_set
                    active = active or bool(set_valid.any())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite MT loss')
                if active:
                    loss.backward()
                    optimizer.step()
                    ema.update(student)
                step += 1
                record = dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()), dice=float(dice.detach()),
                              consistency=float(consistency.detach()), supervised_pixels=supervised, consistency_pixels=unknown)
                if ecss_bank is not None:
                    record.update(ecss_ce=float(added_set.detach()), ecss_pixels=int(set_valid.sum()),
                        original_unknown_pixels=original_unknown_count, ecss_loss_weight=ecss_options['loss_weight'],
                        ecss_mass=float(set_weights.sum()), original_foreground_mass=float(original_fg_mass.sum()))
                log.write(json.dumps(record) + '\n')
                if step == 1 or step % 25 == 0 or step == planned:
                    log.flush()
                    print(f'[MT {"warmup" if warmup else "stage2"}] {step}/{planned}, loss={record["loss"]:.6f}', flush=True)
            scheduler.step()
            if step >= planned:
                break
    save_checkpoint(output / 'student_final.pt', student, steps=step, role='student')
    save_checkpoint(output / 'teacher_final.pt', ema.teacher, steps=step, role='teacher')
    return ema.teacher, step


def adapt_target(config):
    """Run with a short lock to avoid concurrent writes to one run directory."""
    output = Path(config['output']['root'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This output directory is already in use')
        return _adapt_target(config, output)


def _adapt_target(config, output):
    from .medsam import MedSAM, EmbeddingCache
    from .glfp import generate_labels, local_round_count
    from .ecss import ECSSBank, generate_sets
    device = make_device(config['runtime']['device'])
    seed_everything(config['experiment']['seed'])
    save_config(config, output)
    settings = config['adaptation']
    unknown = set(settings) - {'medsam_checkpoint', 'embedding_cache_root', 'encoder_batch_size',
        'decoder_batch_size', 'warmup', 'glfp', 'ecss', 'training', 'segments', 'prepare_only'}
    if unknown:
        raise ValueError(f'Unknown ASEI adaptation options: {sorted(unknown)}')
    glfp = merge(GLFP_DEFAULTS, settings.get('glfp', {}))
    ecss = merge(ECSS_DEFAULTS, settings.get('ecss', {}))
    validate_options(glfp, ecss)
    if not Path(settings['medsam_checkpoint']).is_file():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {settings['medsam_checkpoint']}")
    if settings.get('encoder_batch_size', 2) < 1 or settings.get('decoder_batch_size', 24) < 1:
        raise ValueError('MedSAM batch sizes must be positive')
    if settings.get('segments', 1) < 1:
        raise ValueError('segments must be positive')
    budgets = [settings['training']]
    if settings['warmup']['enabled']:
        budgets.append(settings['warmup'])
    for budget in budgets:
        if (budget['epochs'] is not None and budget['epochs'] < 1) or budget['max_steps'] < 1 or budget['learning_rate'] <= 0:
            raise ValueError('MT needs positive max_steps and learning_rate, and positive epochs or None')
    model = build_model(config).to(device)
    load_checkpoint(config['source_checkpoint'], model, device)
    print(f"[ASEI] device={device}, source_checkpoint={config['source_checkpoint']}", flush=True)
    dataset = build_dataset(config, config['experiment']['target_domain'], 'train', False)
    dataset_hash = data_digest(dataset)
    warmup = settings['warmup']
    cache = output / 'cache'
    warm_steps = 0
    if warmup['enabled']:
        identity = json_digest(dict(initial=state_digest(model), data=dataset_hash, warmup=warmup,
            seed=config['experiment']['seed'], batch_size=config['data']['batch_size'],
            workers=config['data']['num_workers'], target=config['experiment']['target_domain'], version=1))
        directory = cache / 'warmup' / identity
        checkpoint = directory / 'teacher_final.pt'
        marker = directory / 'complete.json'
        if marker.is_file() and checkpoint.is_file() and read_json(marker)['sha256'] == sha_file(checkpoint):
            load_checkpoint(checkpoint, model, device)
            warm_steps = read_json(marker)['steps']
            print('[ASEI] reuse MT warmup teacher', flush=True)
        else:
            teacher, warm_steps = train_mt(model, config, warmup, directory, device, warmup=True)
            model.load_state_dict(teacher.state_dict())
            write_json(marker, dict(sha256=sha_file(checkpoint), steps=warm_steps))
    frozen = deepcopy(model).eval().requires_grad_(False)
    save_checkpoint(output / 'prompt_teacher.pt', frozen, role='frozen_prompt_teacher')
    encoder_batch_size, decoder_batch_size = settings.get('encoder_batch_size', 2), settings.get('decoder_batch_size', 24)
    features = EmbeddingCache(settings['embedding_cache_root'], settings['medsam_checkpoint'], device, encoder_batch_size)
    descriptor = dict(teacher=state_digest(frozen), data=dataset_hash, options=glfp,
        feature_namespace=features.root.name, decoder_batch_size=decoder_batch_size,
        implementation=sha_file(Path(__file__).with_name('glfp.py')), version=2)
    directory = cache / 'glfp' / json_digest(descriptor)
    final_round = f'round{local_round_count(glfp)}'
    source, rawdir = directory / final_round, directory / 'responses'
    if not valid_bank(source, len(dataset)) or not valid_bank(rawdir, len(dataset)):
        sam = MedSAM(settings['medsam_checkpoint'], device)
        loader = build_loader(config, dataset, generator=torch.Generator().manual_seed(420104))
        source, rawdir = generate_labels(frozen, loader, glfp, sam, features, directory, device,
                                       encoder_batch_size, decoder_batch_size)
        seal_bank(source, len(dataset))
        seal_bank(rawdir, len(dataset))
        del sam
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        print('[ASEI] reuse GLFP pseudo labels', flush=True)
    ecss_path = None
    if ecss['enabled']:
        generation_options = {k:v for k,v in ecss.items() if k not in ['enabled', 'loss_weight']}
        descriptor = dict(source=read_json(source / 'bank.json')['sha256'],
            responses=read_json(rawdir / 'bank.json')['sha256'], options=generation_options,
            implementation=sha_file(Path(__file__).with_name('ecss.py')),
            evidence_implementation=sha_file(Path(__file__).with_name('glfp.py')),
            glfp_options=glfp,version=3)
        ecss_path = cache / 'ecss' / json_digest(descriptor)
        if not valid_bank(ecss_path, len(dataset)):
            generate_sets(source, rawdir, ecss_path, len(dataset), ecss, glfp_options=glfp)
            seal_bank(ecss_path, len(dataset))
        else:
            print('[ASEI] reuse ECSS class sets', flush=True)
    write_json(output / 'preparation.json', dict(prompt_teacher_sha256=state_digest(frozen),
        target_data_sha256=dataset_hash, glfp=glfp,
        ecss={k:v for k,v in ecss.items() if k != 'loss_weight'}, labels=str(source),
        responses=str(rawdir), sets=str(ecss_path) if ecss_path else None, target_labels_used=False))
    if settings.get('prepare_only', False):
        print('[ASEI] GLFP labels and ECSS sets prepared; Stage2 skipped', flush=True)
        return None
    bank = LabelBank(source, len(dataset))
    set_bank = ECSSBank(ecss_path, len(dataset)) if ecss_path else None
    teacher, total = None, 0
    segments = settings.get('segments', 1)
    for segment in range(segments):
        teacher, steps = train_mt(model, config, settings['training'], output / f'stage2/segment{segment+1}',
                                  device, bank, teacher, ecss_bank=set_bank, ecss_options=ecss)
        total += steps
    save_checkpoint(output / 'student_final.pt', model, steps=total, role='student')
    save_checkpoint(output / 'teacher_final.pt', teacher, steps=total, role='teacher')
    write_json(output / 'summary.json', dict(warmup_steps=warm_steps, stage2_steps=total, segments=segments,
        framework='ASEI', modules=['GLFP'] + (['ECSS'] if ecss['enabled'] else []),
        target_labels_used=False, evaluation_checkpoint='teacher_final.pt', labels=str(source),
        sets=str(ecss_path) if ecss_path else None, ecss_loss_weight=ecss['loss_weight']))
    return output / 'teacher_final.pt'
