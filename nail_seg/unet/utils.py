from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid


@dataclass
class RunPaths:
    root: Path
    samples: Path
    log_file: Path
    metrics_file: Path
    best_ckpt: Path
    last_ckpt: Path


def create_run_dir(base_dir: str | Path = "runs") -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(base_dir) / timestamp
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        samples=samples,
        log_file=root / "train.log",
        metrics_file=root / "metrics.json",
        best_ckpt=root / "best.pt",
        last_ckpt=root / "last.pt",
    )


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def save_metrics(metrics_path: Path, history: list[dict]) -> None:
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def log_line(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + os.linesep)


def save_sample_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    path: Path,
    max_items: int = 4,
) -> None:
    images = images[:max_items].detach().cpu()
    masks = masks[:max_items].detach().cpu()
    preds = preds[:max_items].detach().cpu()

    overlays = []
    for image, mask, pred in zip(images, masks, preds):
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        mask_bin = (mask.squeeze(0).numpy() > 0.5).astype(np.uint8)
        pred_bin = (pred.squeeze(0).numpy() > 0.5).astype(np.uint8)
        mask_np = mask_bin * 255
        pred_np = pred_bin * 255

        mask_rgb = np.stack([mask_np, np.zeros_like(mask_np), np.zeros_like(mask_np)], axis=-1)
        pred_rgb = np.stack([np.zeros_like(pred_np), pred_np, np.zeros_like(pred_np)], axis=-1)
        overlay = np.clip(image_np * 0.7 + mask_rgb * 0.3, 0, 255).astype(np.uint8)
        overlay_pred = np.clip(image_np * 0.7 + pred_rgb * 0.3, 0, 255).astype(np.uint8)

        mask_boundary = compute_boundary_map(
            torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0).float(), width=1
        )
        pred_boundary = compute_boundary_map(
            torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0).float(), width=1
        )
        mask_boundary_np = (mask_boundary.squeeze().numpy() * 255).astype(np.uint8)
        pred_boundary_np = (pred_boundary.squeeze().numpy() * 255).astype(np.uint8)
        mask_boundary_rgb = np.stack(
            [np.zeros_like(mask_boundary_np), mask_boundary_np, mask_boundary_np], axis=-1
        )
        pred_boundary_rgb = np.stack(
            [pred_boundary_np, pred_boundary_np, np.zeros_like(pred_boundary_np)], axis=-1
        )

        stacked = np.concatenate(
            [image_np, overlay, overlay_pred, mask_boundary_rgb, pred_boundary_rgb], axis=1
        )
        overlays.append(torch.from_numpy(stacked).permute(2, 0, 1))

    grid = make_grid(overlays, nrow=1)
    grid_img = grid.permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(grid_img).save(path)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def iter_batches(loader: Iterable) -> Iterable:
    for batch in loader:
        yield batch


def compute_boundary_map(mask: torch.Tensor, width: int = 1) -> torch.Tensor:
    if width <= 0:
        return torch.zeros_like(mask)
    kernel = 2 * width + 1
    dilation = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width)
    erosion = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=width)
    boundary = (dilation - erosion).clamp(min=0.0, max=1.0)
    return boundary
