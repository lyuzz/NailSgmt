"""Input detection and mask loading utilities for step 2 pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class MaskSource:
    """Container for resolved mask sources."""

    masks: list[np.ndarray]
    source: str


def resolve_input_images(input_path: Path) -> list[Path]:
    """Resolve input images from a file or directory."""
    if input_path.is_dir():
        images = sorted(
            [
                path
                for path in input_path.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        if not images:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        return images
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_image_bgr(image_path: Path) -> np.ndarray:
    """Load an image as BGR uint8."""
    image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def load_instance_masks(
    image_path: Path,
    image_shape: tuple[int, int],
    mode: str,
    mask_dir: Path | None = None,
    json_path: Path | None = None,
) -> MaskSource:
    """Load instance masks for an image.

    Args:
        image_path: Path to the image.
        image_shape: (height, width) for mask sizing.
        mode: "auto", "mask_dir", or "json".
        mask_dir: Directory containing masks when mode is "mask_dir".
        json_path: JSON file when mode is "json".
    """
    if mode == "auto":
        mask_files = _find_mask_files(image_path)
        if mask_files:
            masks = _load_masks_from_files(mask_files, split_single=len(mask_files) == 1)
            return MaskSource(masks=masks, source="mask_files")
        json_file = _find_json_file(image_path)
        if json_file:
            masks = _load_masks_from_json(json_file, image_shape)
            return MaskSource(masks=masks, source="json")
        raise FileNotFoundError(
            "Auto mode failed: no masks/json found for "
            f"{image_path}. Use --mode mask_dir with --mask_dir or --mode json with --json."
        )

    if mode == "mask_dir":
        if mask_dir is None:
            raise ValueError("--mask_dir is required when mode=mask_dir")
        mask_files = sorted(mask_dir.glob("*.png"))
        if not mask_files:
            raise FileNotFoundError(f"No PNG masks found in: {mask_dir}")
        masks = _load_masks_from_files(mask_files, split_single=False)
        return MaskSource(masks=masks, source="mask_dir")

    if mode == "json":
        if json_path is None:
            raise ValueError("--json is required when mode=json")
        masks = _load_masks_from_json(json_path, image_shape)
        return MaskSource(masks=masks, source="json")

    raise ValueError(f"Unsupported mode: {mode}")


def _find_mask_files(image_path: Path) -> list[Path]:
    stem = image_path.stem
    directory = image_path.parent

    candidates: list[Path] = []
    for name in (f"{stem}_mask.png", f"{stem}_masks.png"):
        path = directory / name
        if path.exists():
            candidates.append(path)

    pattern_files: list[Path] = []
    pattern_files.extend(sorted(directory.glob(f"{stem}_mask_*.png")))
    pattern_files.extend(sorted(directory.glob(f"{stem}_inst_*.png")))
    if pattern_files:
        candidates.extend(pattern_files)

    masks_dir = directory / "masks"
    if masks_dir.exists():
        candidates.extend(sorted(masks_dir.glob(f"{stem}_*.png")))
        stem_dir = masks_dir / stem
        if stem_dir.exists():
            candidates.extend(sorted(stem_dir.glob("*.png")))

    return _unique_paths(candidates)


def _find_json_file(image_path: Path) -> Path | None:
    stem = image_path.stem
    directory = image_path.parent
    for name in (f"{stem}.json", f"{stem}_pred.json", "predictions.json"):
        path = directory / name
        if path.exists():
            return path
    return None


def _load_masks_from_files(paths: Iterable[Path], split_single: bool) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for path in paths:
        mask = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")
        if split_single:
            masks.extend(_split_instances(mask))
        else:
            masks.append(_to_binary_mask(mask))
    if not masks:
        raise ValueError("No masks could be parsed from mask files")
    return masks


def _load_masks_from_json(json_path: Path, image_shape: tuple[int, int]) -> list[np.ndarray]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = _extract_annotations(data)
    if not annotations:
        raise ValueError(f"No annotations found in JSON: {json_path}")
    height, width = image_shape
    masks: list[np.ndarray] = []
    for ann in annotations:
        segmentation = ann.get("segmentation")
        if segmentation is None:
            continue
        if isinstance(segmentation, dict):
            raise ValueError(
                "RLE masks are not supported. Convert to polygons or export instance masks."
            )
        if isinstance(segmentation, list) and segmentation:
            mask = np.zeros((height, width), dtype=np.uint8)
            polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
            for poly in polygons:
                if len(poly) < 6:
                    continue
                coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
                coords = np.round(coords).astype(np.int32)
                cv2.fillPoly(mask, [coords], 255)
            if np.any(mask > 0):
                masks.append(mask)
    if not masks:
        raise ValueError(f"No polygon masks were parsed from JSON: {json_path}")
    return masks


def _extract_annotations(data: object) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return [item for item in data["annotations"] if isinstance(item, dict)]
        if "predictions" in data and isinstance(data["predictions"], list):
            return [item for item in data["predictions"] if isinstance(item, dict)]
    return []


def _split_instances(mask: np.ndarray) -> list[np.ndarray]:
    if mask.ndim == 2:
        unique_vals = np.unique(mask)
        unique_vals = unique_vals[unique_vals != 0]
        if unique_vals.size > 1:
            return [(mask == val).astype(np.uint8) * 255 for val in unique_vals]
        return [_to_binary_mask(mask)]

    if mask.ndim == 3:
        rgb = mask[:, :, :3]
        flat = rgb.reshape(-1, 3)
        unique_vals = np.unique(flat, axis=0)
        unique_vals = unique_vals[~np.all(unique_vals == 0, axis=1)]
        if unique_vals.shape[0] > 1:
            masks = []
            for color in unique_vals:
                match = np.all(rgb == color, axis=2)
                masks.append(match.astype(np.uint8) * 255)
            return masks
        return [_to_binary_mask(rgb)]

    raise ValueError("Unsupported mask dimensions")


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    binary = (mask_gray > 0).astype(np.uint8) * 255
    return binary


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique
