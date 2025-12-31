from __future__ import annotations

import torch


@torch.no_grad()
def threshold_preds(preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (preds >= threshold).float()


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / (denominator + 1e-7)


@torch.no_grad()
def compute_batch_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    preds_bin = threshold_preds(preds, threshold=threshold)
    targets_bin = (targets > 0.5).float()

    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets_bin.view(targets_bin.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou = _safe_divide(intersection, union).mean()
    dice = _safe_divide(2 * intersection, preds_flat.sum(dim=1) + targets_flat.sum(dim=1)).mean()

    tp = intersection
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

    precision = _safe_divide(tp, tp + fp).mean()
    recall = _safe_divide(tp, tp + fn).mean()

    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }
