"""Template board construction utilities."""

from __future__ import annotations

import numpy as np


def build_template(
    nail_norm_rgba_list: list[np.ndarray],
    layout: str,
    canvas_w: int,
    canvas_h: int,
    bg: str,
    gap: int,
) -> tuple[np.ndarray, list[dict]]:
    """Build a template board and return placements metadata."""
    rows = 1 if layout == "five" else 2
    slots_per_row = 5
    total_slots = rows * slots_per_row

    if bg == "white":
        canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255
    elif bg == "transparent":
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    else:
        raise ValueError("bg must be 'white' or 'transparent'")

    if not nail_norm_rgba_list:
        return canvas, []

    slot_h, slot_w = nail_norm_rgba_list[0].shape[:2]

    available_w = canvas_w - 2 * gap - slot_w * slots_per_row
    spacing = available_w / (slots_per_row - 1) if slots_per_row > 1 else 0
    spacing = max(0, spacing)

    x_positions = [int(round(gap + i * (slot_w + spacing))) for i in range(slots_per_row)]

    if rows == 1:
        y_positions = [int(round((canvas_h - slot_h) / 2))]
    else:
        y_positions = [
            int(round(canvas_h * 0.25 - slot_h / 2)),
            int(round(canvas_h * 0.75 - slot_h / 2)),
        ]

    placements: list[dict] = []
    for idx, nail in enumerate(nail_norm_rgba_list[:total_slots]):
        row = idx // slots_per_row
        col = idx % slots_per_row
        x = x_positions[col]
        y = y_positions[row]
        _alpha_blend(canvas, nail, x, y)
        placements.append({"index": idx, "x": x, "y": y, "w": slot_w, "h": slot_h})

    return canvas, placements


def _alpha_blend(canvas: np.ndarray, fg: np.ndarray, x: int, y: int) -> None:
    """Alpha blend an RGBA foreground onto an RGBA canvas in-place."""
    h, w = fg.shape[:2]
    x_end = min(canvas.shape[1], x + w)
    y_end = min(canvas.shape[0], y + h)
    if x_end <= x or y_end <= y:
        return

    fg_crop = fg[: y_end - y, : x_end - x].astype(np.float32) / 255.0
    bg_crop = canvas[y:y_end, x:x_end].astype(np.float32) / 255.0

    alpha = fg_crop[:, :, 3:4]
    blended = fg_crop[:, :, :3] * alpha + bg_crop[:, :, :3] * (1 - alpha)
    out_alpha = alpha[:, :, 0] + bg_crop[:, :, 3] * (1 - alpha[:, :, 0])

    out = np.zeros_like(bg_crop)
    out[:, :, :3] = blended
    out[:, :, 3] = out_alpha

    canvas[y:y_end, x:x_end] = (out * 255).astype(np.uint8)
