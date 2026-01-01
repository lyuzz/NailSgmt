from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX inference for nail segmentation")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ext", type=str, default="jpg,jpeg,png,webp")
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


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_images(input_dir: Path, extensions: set[str]) -> list[Path]:
    return [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]


def run_inference(
    image_path: Path,
    session: ort.InferenceSession,
    img_size: int,
    threshold: float,
    min_area: int,
    masks_dir: Path,
    overlays_dir: Path,
    cutouts_dir: Path,
    bboxes_dir: Path,
) -> None:
    image_bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = image.shape[:2]

    input_tensor = preprocess(image, img_size)

    outputs = session.run(None, {"input": input_tensor})
    prob = outputs[0][0, 0]
    prob_resized = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    mask = (prob_resized >= threshold).astype(np.uint8) * 255
    mask = postprocess(mask, min_area)

    stem = image_path.stem
    mask_path = masks_dir / f"{stem}_mask.png"
    overlay_path = overlays_dir / f"{stem}_overlay.png"
    cutout_path = cutouts_dir / f"{stem}_nails_cutout.png"
    bbox_path = bboxes_dir / f"{stem}_nails_bbox.png"

    Image.fromarray(mask).save(mask_path)
    overlay = overlay_mask(image, mask)
    Image.fromarray(overlay).save(overlay_path)

    cutout = extract_nail_cutout(image_bgr, mask)
    Image.fromarray(cutout).save(cutout_path)
    bbox_cutout = extract_nail_bbox(image_bgr, mask)
    Image.fromarray(bbox_cutout).save(bbox_path)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    onnx_path = Path(args.onnx)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    extensions = {f".{ext.strip().lower()}" for ext in args.ext.split(",") if ext.strip()}
    if not extensions:
        raise ValueError("No valid extensions provided via --ext.")

    image_paths = collect_images(input_dir, extensions)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir} with extensions: {sorted(extensions)}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = onnx_path.parent / "infer" / get_timestamp()

    masks_dir = ensure_dir(out_dir / "masks")
    overlays_dir = ensure_dir(out_dir / "overlays")
    cutouts_dir = ensure_dir(out_dir / "cutouts")
    bboxes_dir = ensure_dir(out_dir / "bboxes")

    successes = 0
    failures: list[Path] = []

    for image_path in image_paths:
        try:
            run_inference(
                image_path=image_path,
                session=session,
                img_size=args.img_size,
                threshold=args.threshold,
                min_area=args.min_area,
                masks_dir=masks_dir,
                overlays_dir=overlays_dir,
                cutouts_dir=cutouts_dir,
                bboxes_dir=bboxes_dir,
            )
            successes += 1
        except Exception as exc:  # noqa: BLE001 - continue processing other images
            failures.append(image_path)
            print(f"Failed processing {image_path}: {exc}")

    print(f"Images found: {len(image_paths)}")
    print(f"Succeeded: {successes}")
    print(f"Failed: {len(failures)}")
    print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
