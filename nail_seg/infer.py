from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX inference for nail segmentation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=500)
    return parser.parse_args()


def preprocess(image: np.ndarray, img_size: int) -> np.ndarray:
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    resized = np.transpose(resized, (2, 0, 1))
    return np.expand_dims(resized, axis=0)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    mask_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_fill, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return mask | flood_inv


def postprocess(mask: np.ndarray, min_area: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cleaned = remove_small_components(closed, min_area)
    filled = fill_holes(cleaned)
    return filled


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    alpha = 0.4
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image, 1 - alpha, red, alpha, 0)[mask_bool]
    return overlay


def extract_nail_cutout(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same spatial dimensions")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    alpha = (mask > 0).astype(np.uint8) * 255
    return np.dstack((image_rgb, alpha))


def extract_nail_bbox(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return extract_nail_cutout(image_bgr, mask)
    ys, xs = np.where(mask_bool)
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    cropped_image = image_bgr[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    return extract_nail_cutout(cropped_image, cropped_mask)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    onnx_path = Path(args.onnx)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    image_bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = image.shape[:2]

    input_tensor = preprocess(image, args.img_size)

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": input_tensor})
    prob = outputs[0][0, 0]
    prob_resized = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    mask = (prob_resized >= args.threshold).astype(np.uint8) * 255
    mask = postprocess(mask, args.min_area)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "mask.png"
    overlay_path = out_dir / "overlay.png"
    cutout_path = out_dir / "nails_cutout.png"
    bbox_path = out_dir / "nails_bbox.png"

    Image.fromarray(mask).save(mask_path)
    overlay = overlay_mask(image, mask)
    Image.fromarray(overlay).save(overlay_path)

    cutout = extract_nail_cutout(image_bgr, mask)
    Image.fromarray(cutout).save(cutout_path)
    bbox_cutout = extract_nail_bbox(image_bgr, mask)
    Image.fromarray(bbox_cutout).save(bbox_path)

    print(f"Saved mask to {mask_path}")
    print(f"Saved overlay to {overlay_path}")
    print(f"Saved cutout to {cutout_path}")
    print(f"Saved bbox cutout to {bbox_path}")


if __name__ == "__main__":
    main()
