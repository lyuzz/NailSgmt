"""Normalization utilities for nail cutouts."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_nail(
    rgba: np.ndarray,
    mask: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, dict]:
    """Normalize a nail cutout into a target slot size.

    Returns the normalized RGBA image and a metadata dictionary.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("RGBA input must have shape HxWx4")

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    points = np.column_stack(np.where(mask_uint8 > 0))
    if points.size == 0:
        raise ValueError("Mask has no foreground pixels")

    points_xy = np.column_stack((points[:, 1], points[:, 0])).astype(np.float32)
    rect = cv2.minAreaRect(points_xy)
    angle = rect[2]
    rect_w, rect_h = rect[1]
    if rect_w < rect_h:
        rotation = angle
    else:
        rotation = angle + 90.0

    center = (rgba.shape[1] / 2, rgba.shape[0] / 2)
    rot_mat = cv2.getRotationMatrix2D(center, -rotation, 1.0)

    rotated_rgba = cv2.warpAffine(
        rgba,
        rot_mat,
        (rgba.shape[1], rgba.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        mask_uint8,
        rot_mat,
        (rgba.shape[1], rgba.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    flipped_180 = False
    height = rotated_mask.shape[0]
    top_band = rotated_mask[: int(0.2 * height)]
    bottom_band = rotated_mask[int(0.8 * height) :]

    top_width = int(np.sum(np.any(top_band > 0, axis=0)))
    bottom_width = int(np.sum(np.any(bottom_band > 0, axis=0)))

    if top_width > bottom_width:
        rotated_rgba = cv2.rotate(rotated_rgba, cv2.ROTATE_180)
        rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_180)
        flipped_180 = True

    ys, xs = np.where(rotated_mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Rotated mask has no foreground")

    pad = 3
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(rotated_mask.shape[1], x_max + pad)
    y_max = min(rotated_mask.shape[0], y_max + pad)

    crop_rgba = rotated_rgba[y_min:y_max, x_min:x_max]
    crop_mask = rotated_mask[y_min:y_max, x_min:x_max]

    crop_h, crop_w = crop_mask.shape[:2]
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    if scale < 1:
        interp_rgba = cv2.INTER_AREA
    else:
        interp_rgba = cv2.INTER_LINEAR

    resized_rgba = cv2.resize(crop_rgba, (new_w, new_h), interpolation=interp_rgba)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized_rgba

    meta = {
        "rotation_deg": float(-rotation + (180.0 if flipped_180 else 0.0)),
        "scale": float(scale),
        "flipped_180": flipped_180,
        "orig_size": [int(rgba.shape[1]), int(rgba.shape[0])],
        "rot_size": [int(rotated_rgba.shape[1]), int(rotated_rgba.shape[0])],
        "crop_bbox_rot": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
        "resized_size": [int(new_w), int(new_h)],
    }
    return canvas, meta
