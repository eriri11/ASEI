"""Source-only supervised training with fixed intensity augmentation."""
from pathlib import Path
import torch
from torch.utils.data import Subset
from .data import build_dataset, build_loader
from .models import build_model
from ._losses import combined_ce_dice_loss
from .metrics import evaluate, evaluate_pixel_dice
from .utils import seed_everything, save_checkpoint, write_json, make_device
from .config import save_config


class SourceAugmentation:
    def __call__(self, sample):
        image = sample['image'].float()
        if image.shape[-2:] != (256, 256):
            raise ValueError('Source training expects converted 256x256 slices')
        # Consume RNG draws for disabled augmentations to preserve reproducibility.
        for _ in range(5):
            torch.rand(())
        if torch.rand(()) < .8:
            contrast = torch.empty(()).uniform_(.5, 1.5).item()
            mean = image.mean((-2, -1), keepdim=True)
            image = (image - mean) * contrast + mean
        if torch.rand(()) < 1.:
            image = image + torch.empty(()).uniform_(-.4, .1).item()
        if torch.rand(()) < .5:
            std = torch.empty(()).uniform_(0., .08).item()
            image = image + torch.randn_like(image) * std
        image = image.clamp(0, 1)
        torch.rand(())  # Disabled salt/pepper.
        if torch.rand(()) < .7:
            gamma = torch.empty(()).uniform_(.4, 2.5).item()
            image = image.pow(1. / gamma)
        torch.rand(())  # Disabled erasing.
        return {**sample, 'image': image.contiguous()}


def build_source_partitions(config):
    """Hold out source training samples before augmentation, using a local RNG.

    The prepared TFRecords do not provide verified patient identities. This
    is a sample split; independent NIfTI test volumes are never opened here.
    """
    settings = config['source_training']
    domain = config['experiment']['source_domain']
    split = settings.get('validation_split', 'train_holdout')
    training = build_dataset(config, domain, 'train', True, SourceAugmentation())
    if split == 'train_holdout':
        fraction = settings.get('validation_fraction', .2)
        if isinstance(fraction, bool) or not 0 < fraction < 1:
            raise ValueError('source_training.validation_fraction must be between 0 and 1')
        count = len(training)
        validation_count = round(count * fraction)
        if not 0 < validation_count < count:
            raise ValueError('Source holdout requires nonempty training and validation partitions')
        validation = build_dataset(config, domain, 'train', True)
        seed = settings.get('split_seed', config['experiment']['seed'])
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
        validation_indices = sorted(order[:validation_count])
        training_indices = sorted(order[validation_count:])
        record = dict(strategy='seeded_sample_split', seed=seed,
            validation_fraction=fraction, domain=domain,
            training_files=[str(training.files[i].relative_to(training.root)) for i in training_indices],
            validation_files=[str(validation.files[i].relative_to(validation.root)) for i in validation_indices])
        return Subset(training, training_indices), Subset(validation, validation_indices), record
    if split != 'val':
        raise ValueError('source_training.validation_split must be train_holdout or val')
    validation = build_dataset(config, domain, 'val', True)
    training_files = {path.resolve() for path in training.files}
    if training_files.intersection(path.resolve() for path in validation.files):
        raise ValueError('Source training and validation files must be disjoint')
    record = dict(strategy='provided_validation_split', domain=domain,
        training_files=[str(path.relative_to(training.root)) for path in training.files],
        validation_files=[str(path.relative_to(validation.root)) for path in validation.files])
    return training, validation, record


def train_source(config):
    seed_everything(config['experiment']['seed'])
    device = make_device(config['runtime']['device'])
    settings = config['source_training']
    if settings['epochs'] < 1:
        raise ValueError('source_training.epochs must be positive')
    training, validation, partition = build_source_partitions(config)
    train_loader = build_loader(config, training, True)
    val_loader = build_loader(config, validation)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
        weight_decay=settings.get('weight_decay', 1e-4), betas=tuple(settings.get('optimizer_betas', [.9, .999])))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=settings.get('lr_gamma', 1.))
    loss_options = settings['loss']
    class_weights = loss_options.get('class_weights')
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=device)
    output = Path(config['output']['root'])
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output)
    write_json(output / 'source_split.json', partition)
    best, history = -1., []
    for epoch in range(settings['epochs']):
        model.train()
        total, count = 0., 0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = combined_ce_dice_loss(model(batch['image'].to(device)), batch['label'].to(device),
                ce_weight=loss_options['ce_weight'], dice_weight=loss_options['dice_weight'],
                ignore_background=loss_options.get('ignore_background', True), class_weights=class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite source loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch['image'])
            count += len(batch['image'])
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                print(f'[source] epoch {epoch+1}/{settings["epochs"]}, step {step}/{len(train_loader)}, loss={loss.item():.5f}', flush=True)
        if getattr(validation, 'volume_sequence_verified', False):
            metrics = evaluate(model, val_loader, 5, device, compute_assd=False)
        else:
            metrics = evaluate_pixel_dice(model, val_loader, 5, device)
        history.append(dict(epoch=epoch+1, loss=total/count, validation=metrics))
        scheduler.step()
        save_checkpoint(output / 'source_last.pt', model, epoch=epoch+1, metrics=metrics)
        if metrics['mean_dice'] > best:
            best = metrics['mean_dice']
            save_checkpoint(output / 'source_best.pt', model, epoch=epoch+1, metrics=metrics)
        write_json(output / 'history.json', history)
        print(f'[source] epoch {epoch+1}: loss={total/count:.5f}, validation Dice={metrics["mean_dice"]:.5f}', flush=True)
    return output / 'source_best.pt'
