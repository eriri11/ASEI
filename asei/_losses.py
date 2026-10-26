"""Supervised and partial-label losses."""
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    include_background: bool = False,
    smooth: float = 1.0,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Compute soft multiclass Dice loss.

    Dice is averaged over classes rather than over pixels. By default class 0
    is excluded because background dominates the cardiac segmentation masks.
    """

    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError(f"Expected logits [B,C,H,W] and target [B,H,W], got {logits.shape} and {target.shape}")
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"Logits/target shape mismatch: {logits.shape} vs {target.shape}")
    if logits.shape[1] < 2 and not include_background:
        raise ValueError("Foreground-only Dice requires at least two classes")
    if smooth <= 0:
        raise ValueError(f"smooth must be positive, got {smooth}")

    probabilities = F.softmax(logits, dim=1)
    valid = None if ignore_index is None else target != ignore_index
    if valid is not None and not bool(valid.any()):
        return logits.sum() * 0.0
    safe_target = target.long() if valid is None else target.masked_fill(~valid, 0).long()
    one_hot_target = F.one_hot(safe_target, num_classes=logits.shape[1])
    one_hot_target = one_hot_target.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
    if valid is not None:
        valid = valid.unsqueeze(1).to(dtype=probabilities.dtype)
        probabilities = probabilities * valid
        one_hot_target = one_hot_target * valid
    if not include_background:
        probabilities = probabilities[:, 1:]
        one_hot_target = one_hot_target[:, 1:]

    reduce_dims = (0, 2, 3)
    intersection = (probabilities * one_hot_target).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot_target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()

def combined_ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    ignore_background: bool = True,
    smooth: float = 1.0,
    class_weights: torch.Tensor | None = None,
    ignore_index: int | None = None,
    ignore_index_reduction: str = "valid_pixels",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return weighted CE+Dice loss and its two components.

    ``ignore_background`` applies to the Dice term only. Cross entropy always
    includes class 0 because background pixels remain valid supervision.
    Weights are used as a direct weighted sum and are not normalized.
    """

    if ce_weight < 0 or dice_weight < 0 or (ce_weight == 0 and dice_weight == 0):
        raise ValueError("ce_weight and dice_weight must be non-negative and not both zero")

    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        if class_weights.ndim != 1 or class_weights.numel() != logits.shape[1]:
            raise ValueError(
                "class_weights must have one value per output class, "
                f"got shape {tuple(class_weights.shape)} for {logits.shape[1]} classes"
            )
        if bool((class_weights < 0).any()) or not bool((class_weights.sum() > 0)):
            raise ValueError("class_weights must be non-negative and not all zero")
    if ignore_index_reduction not in {"valid_pixels", "all_pixels"}:
        raise ValueError("ignore_index_reduction must be 'valid_pixels' or 'all_pixels'")
    if ignore_index is None:
        ce = F.cross_entropy(logits, target, weight=class_weights)
    else:
        valid = target != ignore_index
        safe_target = target.masked_fill(~valid, 0)
        ce_per_pixel = F.cross_entropy(
            logits,
            safe_target,
            weight=class_weights,
            reduction="none",
        )
        valid_float = valid.to(dtype=ce_per_pixel.dtype)
        numerator = (ce_per_pixel * valid_float).sum()
        if ignore_index_reduction == "all_pixels":
            denominator = valid_float.new_tensor(valid_float.numel())
        elif class_weights is None:
            denominator = valid_float.sum()
        else:
            denominator = (class_weights[safe_target] * valid_float).sum()
        ce = numerator / denominator.clamp_min(torch.finfo(ce_per_pixel.dtype).eps)
    if dice_weight:
        dice = soft_dice_loss(
            logits,
            target,
            include_background=not ignore_background,
            smooth=smooth,
            ignore_index=ignore_index,
        )
    else:
        dice = logits.sum() * 0.0
    total = ce_weight * ce + dice_weight * dice
    return total, {"ce_loss": ce.detach(), "dice_loss": dice.detach()}

def masked_balanced_ce_dice(logits, pseudo, weights):
    """Class-balanced weighted CE plus foreground Dice on labeled pixels only."""
    valid = (pseudo != 255) & (weights > 0)
    zero = logits.sum() * 0
    if not bool(valid.any()):
        return zero, zero
    ce = F.cross_entropy(logits, pseudo.long(), ignore_index=255, reduction='none')
    probability = logits.softmax(1)
    masked_weights = weights.detach() * valid
    ce_parts, dice_parts = [], []
    for cls in range(logits.shape[1]):
        positive = valid & (pseudo == cls)
        if not bool(positive.any()):
            continue
        class_weights = masked_weights * positive
        ce_parts.append((ce * class_weights).sum() / class_weights.sum().clamp_min(1e-8))
        if cls > 0:
            intersection = (probability[:, cls] * class_weights).sum()
            denominator = (probability[:, cls] * masked_weights).sum() + class_weights.sum()
            dice_parts.append(1 - (2 * intersection + 1e-6) / (denominator + 1e-6))
    return torch.stack(ce_parts).mean(), torch.stack(dice_parts).mean() if dice_parts else zero

def aligned_cached_targets(pseudo, weights, visible, num_classes, spatial=None, phase=None, sign=None):
    valid=(pseudo!=255)&(weights>0)
    one_hot=F.one_hot(pseudo.clamp(0,num_classes-1).long(),num_classes).permute(0,3,1,2)
    maps=one_hot*(weights.detach()*valid)[:,None]
    channels=torch.cat([maps,visible[:,None].float()],1)
    usable=torch.ones_like(visible)
    if spatial is not None:
        channels=spatial.warp_like(channels,phase,sign)
        usable=spatial.valid_like(channels,phase,sign)[None].expand_as(visible)
    maps=channels[:,:num_classes]
    weight,labels=maps.max(1)
    # expand_as shares storage across batch elements; avoid in-place updates.
    usable=usable&(channels[:,-1]>=1-1e-6)&((maps>1e-8).sum(1)==1)&(weight>0)
    labels=labels.long();labels[~usable]=255
    return labels,weight*usable

def masked_probability_mse(student_probabilities, teacher_probabilities, mask):
    """Average across classes, then selected pixels; empty masks give zero grad."""
    error=(student_probabilities-teacher_probabilities.detach()).square().mean(1)
    return (error*mask).sum()/mask.sum().clamp_min(1)

def aligned_unknown_mask(pseudo,weights,visible,spatial=None,phase=None,sign=None):
    # Original unknowns, not new hard-label interpolation gaps or corruptions.
    unknown=(pseudo==255)|(weights<=0)
    maps=torch.stack([unknown.float(),visible.float()],1)
    valid=torch.ones_like(visible)
    if spatial is not None:
        maps=spatial.warp_like(maps,phase,sign)
        valid=spatial.valid_like(maps,phase,sign)[None].expand_as(visible)
    return valid&(maps[:,0]>=1-1e-6)&(maps[:,1]>=1-1e-6)
