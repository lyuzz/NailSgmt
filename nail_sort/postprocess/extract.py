"""Extraction utilities for nail cutouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class NailInstance:
    """Container for a single nail instance."""

    index: int
    mask_full: np.ndarray
    bbox_full: tuple[int, int, int, int]
    rgba_cutout: np.ndarray
    mask_cutout: np.ndarray
    area: int


def extract_nails(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    min_area: int,
    margin: int,
) -> list[NailInstance]:
    """Extract per-nail cutouts from a source image.

    Args:
        image_bgr: Original image in BGR format.
        masks: List of instance masks (uint8 or bool).
        min_area: Minimum mask area to keep.
        margin: Pixel margin to expand bounding boxes.
    """
    nails: list[NailInstance] = []
    height, width = image_bgr.shape[:2]

    for idx, mask in enumerate(masks):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        area = int(np.sum(mask_uint8 > 0))
        if area < min_area:
            continue
        ys, xs = np.where(mask_uint8 > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        x_min, x_max = xs.min(), xs.max() + 1
        y_min, y_max = ys.min(), ys.max() + 1

        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)

        crop_bgr = image_bgr[y_min:y_max, x_min:x_max]
        crop_mask = mask_uint8[y_min:y_max, x_min:x_max]

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([crop_rgb, crop_mask])

        nails.append(
            NailInstance(
                index=idx,
                mask_full=mask_uint8,
                bbox_full=(x_min, y_min, x_max - x_min, y_max - y_min),
                rgba_cutout=rgba,
                mask_cutout=crop_mask,
                area=area,
            )
        )

    return nails


def save_cutouts(out_dir: Path, nails: list[NailInstance]) -> None:
    """Save extracted cutouts and masks to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for nail in nails:
        cutout_path = out_dir / f"nail_{nail.index:02d}.png"
        mask_path = out_dir / f"nail_{nail.index:02d}_mask.png"
        Image.fromarray(nail.rgba_cutout).save(cutout_path)
        Image.fromarray(nail.mask_cutout).save(mask_path)
