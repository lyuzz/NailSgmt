from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_boundary_map(mask: torch.Tensor, width: int = 1) -> torch.Tensor:
    if width <= 0:
        return torch.zeros_like(mask)
    kernel = 2 * width + 1
    dilation = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)
    erosion = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=width)
    boundary = (dilation - erosion).clamp(min=0.0, max=1.0)
    return boundary


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(torch.sigmoid(preds), targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal.mean()


class DiceBoundaryLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, boundary_weight: float = 0.3, width: int = 1):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.width = width

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(preds)
        dice_loss = self.dice(probs, targets)
        boundary_pred = compute_boundary_map(probs, self.width)
        boundary_target = compute_boundary_map(targets, self.width)
        boundary_loss = self.dice(boundary_pred, boundary_target)
        return self.dice_weight * dice_loss + self.boundary_weight * boundary_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, focal_weight: float = 0.3) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(preds)
        dice_loss = self.dice(probs, targets)
        focal_loss = self.focal(preds, targets)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
