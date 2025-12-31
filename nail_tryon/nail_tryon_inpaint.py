"""Sequential Nail Virtual Try-On (Diffusers Inpaint + ONNX Nail Segmentation).

Usage:
  python nail_tryon/nail_tryon_inpaint.py \
    --img_path <path/to/hand.jpg> \
    --onnx_path <path/to/nail_seg.onnx> \
    --refs_dir <dir/with/5_ref_images> \
    --out_dir <output/dir> \
    --model_id <optional HF model id, default to an SDXL inpaint model>

Outputs:
  - out_dir/preview.png
  - out_dir/nail_0.png ... out_dir/nail_4.png (optional RGBA layers)
  - out_dir/mask_0.png ... out_dir/mask_4.png (optional debug masks)
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
)
from PIL import Image


DEFAULT_SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_SD15_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"


@dataclass
class RoiCrop:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class RawPrediction:
    pred: np.ndarray
    input_shape: Tuple[int, int]
    orig_shape: Tuple[int, int]


def load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_refs(refs_dir: str) -> List[Tuple[str, Image.Image]]:
    if not os.path.isdir(refs_dir):
        raise FileNotFoundError(f"Refs dir not found: {refs_dir}")
    filenames = sorted(
        [
            name
            for name in os.listdir(refs_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
    )
    if len(filenames) != 5:
        raise ValueError(
            f"refs_dir must contain exactly 5 images, found {len(filenames)}"
        )
    refs = []
    for name in filenames:
        path = os.path.join(refs_dir, name)
        refs.append((name, Image.open(path).convert("RGB")))
    return refs


def run_onnx_nail_segmentation(
    session: ort.InferenceSession, img_rgb_uint8: np.ndarray
) -> RawPrediction:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX session has no inputs")
    input_name = inputs[0].name
    input_shape = inputs[0].shape

    orig_h, orig_w = img_rgb_uint8.shape[:2]
    resize_h = orig_h
    resize_w = orig_w
    if len(input_shape) == 4:
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            resize_h, resize_w = input_shape[2], input_shape[3]
    elif len(input_shape) == 3:
        if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
            resize_h, resize_w = input_shape[1], input_shape[2]

    if (resize_h, resize_w) != (orig_h, orig_w):
        resized = cv2.resize(img_rgb_uint8, (resize_w, resize_h))
    else:
        resized = img_rgb_uint8

    img = resized.astype(np.float32) / 255.0
    if len(input_shape) == 4:
        img = np.transpose(img, (2, 0, 1))[None, ...]
    elif len(input_shape) == 3:
        img = np.transpose(img, (2, 0, 1))

    outputs = session.run(None, {input_name: img})
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs")
    pred = outputs[0]
    return RawPrediction(pred=pred, input_shape=(resize_h, resize_w), orig_shape=(orig_h, orig_w))


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.float32)
        if mask.max() <= 1.0:
            mask = mask * 255.0
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask


def _score_component(area: int, bbox: Tuple[int, int, int, int], centroid: Tuple[float, float], shape: Tuple[int, int]) -> float:
    x, y, w, h = bbox
    h_img, w_img = shape
    if h == 0:
        return 0.0
    ratio = w / h
    ratio_penalty = 1.0 - min(abs(ratio - 0.6), 1.0) * 0.5
    area_score = area
    cy = centroid[1]
    pos_bonus = 1.1 if cy < h_img * 0.7 else 0.9
    return area_score * ratio_penalty * pos_bonus


def postprocess_to_instance_masks(
    raw_pred: RawPrediction | np.ndarray, shape: Tuple[int, int]
) -> List[np.ndarray]:
    if isinstance(raw_pred, RawPrediction):
        pred = raw_pred.pred
        pred_h, pred_w = raw_pred.input_shape
        orig_h, orig_w = raw_pred.orig_shape
    else:
        pred = raw_pred
        pred_h, pred_w = pred.shape[-2], pred.shape[-1]
        orig_h, orig_w = shape

    pred = np.array(pred)
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)

    instance_masks: List[np.ndarray] = []
    if pred.ndim == 2:
        semantic = pred
    elif pred.ndim == 3 and pred.shape[1:] == (pred_h, pred_w) and pred.shape[0] <= 10:
        for idx in range(pred.shape[0]):
            mask = pred[idx]
            instance_masks.append(_to_uint8_mask(mask))
        semantic = None
    else:
        semantic = pred.squeeze()
        if semantic.ndim != 2:
            semantic = semantic.reshape(pred_h, pred_w)

    if instance_masks:
        resized_masks = []
        for mask in instance_masks:
            if mask.shape != (orig_h, orig_w):
                mask = _resize_mask(mask, (orig_h, orig_w))
            resized_masks.append((mask > 0).astype(np.uint8) * 255)
        return resized_masks

    if semantic is None:
        return []

    if semantic.shape != (orig_h, orig_w):
        semantic = _resize_mask(semantic, (orig_h, orig_w))

    if semantic.dtype != np.uint8:
        semantic = semantic.astype(np.float32)
        if semantic.max() <= 1.0:
            semantic = semantic * 255.0
        semantic = semantic.astype(np.uint8)

    _, binary = cv2.threshold(semantic, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    h_img, w_img = shape
    min_area = max(50, int(0.0005 * h_img * w_img))
    max_area = int(0.2 * h_img * w_img)

    components = []
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        centroid = centroids[idx]
        score = _score_component(area, (x, y, w, h), centroid, shape)
        components.append((score, idx, area, (x, y, w, h), centroid))

    if not components:
        return []

    components.sort(key=lambda item: item[0], reverse=True)
    selected = components[:5]

    masks = []
    for _, idx, _, _, _ in selected:
        mask = (labels == idx).astype(np.uint8) * 255
        masks.append(mask)
    return masks


def sort_masks_left_to_right(masks: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
    ordered = []
    for mask in masks:
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        ordered.append((mask, (cx, cy)))
    ordered.sort(key=lambda item: item[1][0])
    return ordered


def compute_bbox_and_crop(mask: np.ndarray, margin: int, shape: Tuple[int, int]) -> RoiCrop:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return RoiCrop(0, 0, shape[1], shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, shape[1])
    y1 = min(int(ys.max()) + margin + 1, shape[0])
    return RoiCrop(x0, y0, x1, y1)


def feather_mask(mask_roi_uint8: np.ndarray, feather_px: int, anti_bleed: bool = True) -> np.ndarray:
    mask = (mask_roi_uint8 > 0).astype(np.float32)
    if anti_bleed:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    if feather_px <= 0:
        return mask
    ksize = max(1, feather_px * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), feather_px)
    blurred = np.clip(blurred, 0.0, 1.0)
    return blurred


def _dominant_color_names(ref_img: Image.Image, max_colors: int = 3) -> List[str]:
    palette = [
        ("red", (220, 20, 60)),
        ("pink", (255, 105, 180)),
        ("orange", (255, 165, 0)),
        ("yellow", (255, 215, 0)),
        ("green", (34, 139, 34)),
        ("teal", (0, 128, 128)),
        ("blue", (30, 144, 255)),
        ("purple", (128, 0, 128)),
        ("white", (245, 245, 245)),
        ("gray", (128, 128, 128)),
        ("black", (20, 20, 20)),
        ("brown", (139, 69, 19)),
        ("gold", (212, 175, 55)),
        ("silver", (192, 192, 192)),
    ]

    def closest_name(color: Tuple[int, int, int]) -> str:
        distances = []
        for name, ref in palette:
            dist = sum((c - r) ** 2 for c, r in zip(color, ref))
            distances.append((dist, name))
        distances.sort(key=lambda item: item[0])
        return distances[0][1]

    small = ref_img.resize((64, 64))
    quantized = small.convert("P", palette=Image.ADAPTIVE, colors=max_colors)
    palette_colors = quantized.getpalette()
    color_counts = quantized.getcolors()
    if not color_counts:
        return []
    color_counts.sort(reverse=True)

    names = []
    for count, idx in color_counts[:max_colors]:
        base = idx * 3
        color = tuple(palette_colors[base : base + 3])
        names.append(closest_name(color))
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def build_prompt(ref_img: Image.Image) -> str:
    colors = _dominant_color_names(ref_img)
    if colors:
        color_phrase = " and ".join(colors)
        style_hint = f"Apply a nail art design with {color_phrase} colors."
    else:
        style_hint = "Apply a detailed nail art design."
    return (
        "Realistic photo of a hand. Repaint ONLY the nail inside the mask. "
        "Keep skin texture, finger shape, lighting, and background unchanged. "
        "Apply the nail design style from the reference. "
        + style_hint
    )


def inpaint_roi(
    pipeline,
    roi_img: np.ndarray,
    roi_mask: np.ndarray,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    strength: float,
    device: torch.device,
) -> np.ndarray:
    generator = torch.Generator(device=device).manual_seed(seed)
    roi_pil = Image.fromarray(roi_img)
    mask_pil = Image.fromarray((roi_mask > 0).astype(np.uint8) * 255)

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=roi_pil,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    )
    output = result.images[0]
    return np.array(output.convert("RGB"), dtype=np.uint8)


def composite_roi(
    base_img: np.ndarray, gen_roi: np.ndarray, alpha_roi: np.ndarray, crop: RoiCrop
) -> np.ndarray:
    out = base_img.copy()
    roi_base = out[crop.y0 : crop.y1, crop.x0 : crop.x1]
    alpha = alpha_roi[..., None]
    blended = gen_roi.astype(np.float32) * alpha + roi_base.astype(np.float32) * (1 - alpha)
    out[crop.y0 : crop.y1, crop.x0 : crop.x1] = blended.astype(np.uint8)
    return out


def export_rgba_layer(
    full_img: np.ndarray, mask: np.ndarray, out_path: str
) -> None:
    h, w = full_img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bin = (mask > 0)
    rgba[mask_bin, :3] = full_img[mask_bin]
    rgba[..., 3] = (mask_bin * 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(out_path)


def configure_pipeline(model_id: Optional[str]) -> Tuple[object, torch.device, str]:
    def is_sdxl_model(value: str) -> bool:
        lowered = value.lower()
        return "sdxl" in lowered or "stable-diffusion-xl" in lowered

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    if model_id:
        if is_sdxl_model(model_id):
            pipeline_cls = StableDiffusionXLInpaintPipeline
        else:
            pipeline_cls = StableDiffusionInpaintPipeline
        chosen_model = model_id
    else:
        if has_cuda:
            pipeline_cls = StableDiffusionXLInpaintPipeline
            chosen_model = DEFAULT_SDXL_INPAINT_MODEL
        else:
            pipeline_cls = StableDiffusionInpaintPipeline
            chosen_model = DEFAULT_SD15_INPAINT_MODEL

    pipeline = pipeline_cls.from_pretrained(
        chosen_model,
        torch_dtype=dtype,
    )
    pipeline.to(device)
    return pipeline, device, chosen_model


def resolve_path(path: str, base_dirs: Sequence[str], must_exist: bool = True) -> str:
    if os.path.isabs(path):
        if must_exist and not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        return path

    candidates = [os.path.abspath(os.path.join(base, path)) for base in base_dirs]
    if must_exist:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        tried = ", ".join(candidates)
        raise FileNotFoundError(f"Path not found: {path}. Tried: {tried}")

    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Nail Virtual Try-On")
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--refs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--roi_margin", type=int, default=80)
    parser.add_argument("--feather_px", type=int, default=6)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--export_layers", type=int, default=1)
    parser.add_argument("--save_debug", type=int, default=1)
    parser.add_argument("--use_ip_adapter", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_dirs = [os.getcwd(), script_dir, repo_root]

    img_path = resolve_path(args.img_path, base_dirs, must_exist=True)
    onnx_path = resolve_path(args.onnx_path, base_dirs, must_exist=True)
    refs_dir = resolve_path(args.refs_dir, base_dirs, must_exist=True)
    out_dir = resolve_path(args.out_dir, base_dirs, must_exist=False)

    os.makedirs(out_dir, exist_ok=True)

    img = load_image_rgb(img_path)
    refs = load_refs(refs_dir)

    session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    raw_pred = run_onnx_nail_segmentation(session, img)
    masks = postprocess_to_instance_masks(raw_pred, img.shape[:2])

    if not masks:
        logging.warning("No nail masks detected.")
        preview_path = os.path.join(out_dir, "preview.png")
        Image.fromarray(img).save(preview_path)
        return

    ordered = sort_masks_left_to_right(masks)
    if len(ordered) != 5:
        logging.warning("Detected %d nails, expected 5.", len(ordered))

    for idx, (_, centroid) in enumerate(ordered):
        logging.info("Nail %d centroid: (%.1f, %.1f)", idx, centroid[0], centroid[1])

    pipeline, device, chosen_model = configure_pipeline(args.model_id)
    logging.info("Using inpaint model: %s", chosen_model)

    if args.use_ip_adapter:
        logging.info("IP-Adapter reference conditioning requested but not implemented yet.")

    composite = img.copy()

    nail_outputs = []
    for idx, (mask, _) in enumerate(ordered):
        if idx >= len(refs):
            break
        ref_name, ref_img = refs[idx]
        logging.info("Applying ref %s to nail %d", ref_name, idx)

        crop = compute_bbox_and_crop(mask, args.roi_margin, img.shape[:2])
        roi_img = composite[crop.y0 : crop.y1, crop.x0 : crop.x1]
        roi_mask = mask[crop.y0 : crop.y1, crop.x0 : crop.x1]

        alpha = feather_mask(roi_mask, args.feather_px, anti_bleed=True)
        prompt = build_prompt(ref_img)
        negative_prompt = (
            "skin changes, finger deformation, extra nails, artifacts, blurry, "
            "low quality, text, watermark"
        )

        gen_roi = inpaint_roi(
            pipeline,
            roi_img,
            roi_mask,
            prompt,
            negative_prompt,
            seed=args.seed + idx,
            steps=args.steps,
            guidance=args.guidance,
            strength=args.strength,
            device=device,
        )

        composite = composite_roi(composite, gen_roi, alpha, crop)
        nail_outputs.append((idx, mask.copy()))

        if args.save_debug:
            mask_path = os.path.join(out_dir, f"mask_{idx}.png")
            cv2.imwrite(mask_path, mask)

    preview_path = os.path.join(out_dir, "preview.png")
    Image.fromarray(composite).save(preview_path)

    if args.export_layers:
        for idx, mask in nail_outputs:
            out_path = os.path.join(out_dir, f"nail_{idx}.png")
            export_rgba_layer(composite, mask, out_path)


if __name__ == "__main__":
    main()
