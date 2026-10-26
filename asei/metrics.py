"""Case-mean Dice, area-weighted ASSD and optional per-class 3D LCC."""
from collections import defaultdict
from typing import Any,Dict,Optional,Tuple
import numpy as np
import torch
from scipy import ndimage
import surface_distance as surfdist

def metric_schema(num_classes: int) -> Tuple[Dict[int, str], Tuple[Tuple[str, int], ...]]:
    """Return label names and reporting order for a segmentation head."""

    if num_classes == len(LABEL_NAMES):
        return dict(LABEL_NAMES), REPORT_ORDER

    names = {class_id: f"class_{class_id}" for class_id in range(num_classes)}
    order = tuple((names[class_id], class_id) for class_id in range(1, num_classes))
    return names, order
LABEL_NAMES={0:"background",1:"MYO",2:"LA",3:"LV",4:"AA"}
REPORT_ORDER=(("AA",4),("LA",2),("LV",3),("MYO",1))

def dice_3d(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute binary Dice for two 3D masks.

    A class absent from both masks receives Dice=1.0. This convention keeps
    the per-case macro average well-defined for small test sets.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    size = prediction.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    if size == 0:
        return 1.0
    return float(2.0 * intersection / size)

def voxel_assd(prediction: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Compute CCRC-compatible 3D ASSD with unit voxel spacing.

    ASSD is undefined when exactly one mask is empty, so this function returns
    ``None`` in that case. If both masks are empty, the distance is zero.
    For non-empty masks, this uses area-weighted surfels and averages the two
    directed surface distances, matching ``SFDA-CCRC/metric2.py``.
    """
    prediction = np.ascontiguousarray(prediction, dtype=np.bool_)
    target = np.ascontiguousarray(target, dtype=np.bool_)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim not in (2, 3):
        raise ValueError(f"ASSD expects a 2D or 3D mask, got {prediction.ndim}D")

    prediction_any = bool(prediction.any())
    target_any = bool(target.any())
    if not prediction_any and not target_any:
        return 0.0
    if prediction_any != target_any:
        return None

    surface_distances = surfdist.compute_surface_distances(
        target,
        prediction,
        spacing_mm=(1.0,) * prediction.ndim,
    )
    directed_distances = surfdist.compute_average_surface_distance(surface_distances)
    return float(np.mean(directed_distances))

@torch.no_grad()
def evaluate_pixel_dice(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Stream dataset-wide foreground Dice for slices with no verified 3D order.

    Accumulate counts over all pixels, not per-slice/per-batch Dice averages.
    No case reconstruction, 3D postprocessing or ASSD is meaningful here.
    """
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    num_samples = 0
    for batch in loader:
        image = batch["image"].to(device)
        output = model(image)
        logits = output["out"] if isinstance(output, dict) else output
        prediction = logits.argmax(dim=1)
        target = batch["label"].to(device=device, dtype=torch.long)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/label shape mismatch: {prediction.shape} vs {target.shape}")
        if bool(((target < 0) | (target >= num_classes)).any()):
            raise ValueError(f"Evaluation labels must be in [0, {num_classes - 1}]")
        if logits.shape[1] != num_classes:
            raise ValueError(f"Expected {num_classes} output classes, got {logits.shape[1]}")
        confusion += torch.bincount(
            (target * num_classes + prediction).reshape(-1),
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        num_samples += image.shape[0]
    if num_samples == 0:
        raise ValueError("Evaluation loader produced no samples")

    counts = confusion.cpu().numpy().astype(np.float64)
    label_names, report_order = metric_schema(num_classes)
    result: Dict[str, Any] = {
        "metric_protocol": "dataset_pixel_dice",
        "assd_computed": False,
        "postprocessing": "none",
        "label_mapping": {str(class_id): name for class_id, name in label_names.items()},
        "report_order": [name for name, _ in report_order],
        "num_samples": num_samples,
    }
    dice_values = []
    for class_name, class_id in report_order:
        size = counts[class_id].sum() + counts[:, class_id].sum()
        dice = float(2 * counts[class_id, class_id] / size) if size else 1.0
        result[f"dice_{class_name}"] = dice
        dice_values.append(dice)
    result["mean_dice"] = float(np.mean(dice_values)) if dice_values else 0.0
    return result

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    compute_assd: bool = True,
    largest_component: bool = False,
) -> Dict[str, Any]:
    """Evaluate reconstructed test volumes using 3D Dice and voxel ASSD.

    The loader must provide ``case_id`` and integer ``slice_id`` fields. Each
    case is reconstructed along the last axis before metrics are computed.
    Set ``compute_assd=False`` for fast epoch-level validation; ASSD is then
    omitted and should be computed on the final/best checkpoint only.
    """
    model.eval()
    cases = defaultdict(dict)
    for batch in loader:
        output = model(batch["image"].to(device))
        logits = output["out"] if isinstance(output, dict) else output
        prediction = logits.argmax(dim=1)
        predictions = prediction.detach().cpu().numpy()
        targets = batch["label"].detach().cpu().numpy()
        case_ids = batch["case_id"]
        slice_ids = batch["slice_id"]
        for index, (case_id, slice_id) in enumerate(zip(case_ids, slice_ids)):
            case_key = str(case_id)
            slice_key = int(slice_id)
            if slice_key in cases[case_key]:
                raise ValueError(f"Duplicate prediction for case={case_key}, slice={slice_key}")
            cases[case_key][slice_key] = (predictions[index], targets[index])

    if not cases:
        raise ValueError("Evaluation loader produced no cases")

    label_names, report_order = metric_schema(num_classes)
    case_results = []
    dice_by_class = {class_id: [] for class_id in range(1, num_classes)}
    assd_by_class = {class_id: [] for class_id in range(1, num_classes)} if compute_assd else None
    for case_id, slices in sorted(cases.items()):
        ordered = [slices[slice_id] for slice_id in sorted(slices)]
        prediction_volume = np.stack([item[0] for item in ordered], axis=-1)
        target_volume = np.stack([item[1] for item in ordered], axis=-1)
        if largest_component:
            prediction_volume = _keep_largest_component_per_class(prediction_volume, num_classes)
        per_case = {"case_id": case_id, "num_slices": len(ordered)}
        for class_name, class_id in report_order:
            prediction_mask = prediction_volume == class_id
            target_mask = target_volume == class_id
            dice_value = dice_3d(prediction_mask, target_mask)
            dice_by_class[class_id].append(dice_value)
            per_case[f"dice_{class_name}"] = dice_value
            if compute_assd:
                assd_value = voxel_assd(prediction_mask, target_mask)
                if assd_value is not None:
                    assd_by_class[class_id].append(assd_value)
                per_case[f"assd_{class_name}"] = assd_value
        case_results.append(per_case)

    result: Dict[str, Any] = {
        "metric_protocol": (
            "case_3d_dice_and_ccrc_surface_assd" if compute_assd else "case_3d_dice"
        ) + ("_largest_component" if largest_component else ""),
        "assd_computed": compute_assd,
        "postprocessing": "largest_3d_component_per_foreground_class" if largest_component else "none",
        "label_mapping": {str(class_id): name for class_id, name in label_names.items()},
        "report_order": [class_name for class_name, _ in report_order],
        "num_cases": len(case_results),
        "cases": case_results,
    }
    if compute_assd:
        result["assd_method"] = "surface_distance_area_weighted_surfels"
        result["assd_spacing"] = [1.0, 1.0, 1.0]
    dice_values = []
    assd_class_means = []
    for class_name, class_id in report_order:
        class_dice = float(np.mean(dice_by_class[class_id]))
        result[f"dice_{class_name}"] = class_dice
        dice_values.append(class_dice)
        if compute_assd:
            class_assd = assd_by_class[class_id]
            class_assd_mean = float(np.mean(class_assd)) if class_assd else None
            result[f"assd_{class_name}"] = class_assd_mean
            result[f"assd_valid_cases_{class_name}"] = len(class_assd)
            result[f"assd_excluded_cases_{class_name}"] = len(case_results) - len(class_assd)
            if class_assd_mean is not None:
                assd_class_means.append(class_assd_mean)
    result["mean_dice"] = float(np.mean(dice_values)) if dice_values else 0.0
    if compute_assd:
        # Average class means after excluding undefined surface distances.
        result["mean_assd"] = float(np.mean(assd_class_means)) if assd_class_means else None
    return result

def _keep_largest_component_per_class(prediction: np.ndarray, num_classes: int) -> np.ndarray:
    result = prediction.copy()
    for class_id in range(1, num_classes):
        component_labels, count = ndimage.label(prediction == class_id)
        if count <= 1:
            continue
        sizes = np.bincount(component_labels.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        result[(prediction == class_id) & (component_labels != largest)] = 0
    return result
