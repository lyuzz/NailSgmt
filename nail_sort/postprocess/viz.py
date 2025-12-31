"""Debug visualization helpers for step 2 pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_overlay_debug(image_bgr: np.ndarray, masks: list[np.ndarray], out_path: Path) -> None:
    """Save overlay of mask contours on the original image."""
    overlay = image_bgr.copy()
    rng = np.random.default_rng(42)
    for mask in masks:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.drawContours(overlay, contours, -1, color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path.as_posix(), overlay)


def save_contour_debug(
    image_shape: tuple[int, int],
    masks: list[np.ndarray],
    out_path: Path,
) -> None:
    """Save diagnostic minAreaRect boxes for each mask."""
    height, width = image_shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(123)
    for mask in masks:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        points = np.column_stack(np.where(mask_uint8 > 0))
        if points.size == 0:
            continue
        points_xy = np.column_stack((points[:, 1], points[:, 0])).astype(np.float32)
        rect = cv2.minAreaRect(points_xy)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.drawContours(canvas, [box], 0, color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path.as_posix(), canvas)
