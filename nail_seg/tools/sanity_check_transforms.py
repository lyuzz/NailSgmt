from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image

from unet.dataset import IMAGENET_MEAN, IMAGENET_STD, NailSegDataset, build_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check train transforms")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 0] = 255
    alpha = 0.4
    mask_bool = mask > 0
    overlay[mask_bool] = (image[mask_bool] * (1 - alpha) + red[mask_bool] * alpha).astype(
        np.uint8
    )
    return overlay


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = image * std + mean
    image = (image.clamp(0, 1) * 255).byte()
    return image.permute(1, 2, 0).cpu().numpy()


def inspect_mask_interpolation(transforms: A.Compose) -> list[str]:
    issues: list[str] = []
    for t in transforms.transforms:
        if isinstance(t, (A.Resize, A.Affine)):
            mask_interp = getattr(t, "mask_interpolation", None)
            if mask_interp is not None and mask_interp != cv2.INTER_NEAREST:
                issues.append(
                    f"{t.__class__.__name__} mask_interpolation={mask_interp} (expected {cv2.INTER_NEAREST})"
                )
    return issues


def summarize_mask(mask: torch.Tensor) -> tuple[list[float], float, bool]:
    mask_flat = mask.flatten()
    unique_vals = torch.unique(mask_flat).cpu().numpy().tolist()
    foreground_ratio = mask_flat.float().mean().item()
    is_binary = torch.all(
        torch.logical_or(torch.isclose(mask_flat, torch.tensor(0.0)), torch.isclose(mask_flat, torch.tensor(1.0)))
    ).item()
    return unique_vals, foreground_ratio, bool(is_binary)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images" / "train"
    masks_dir = data_dir / "masks" / "train"

    transforms = build_train_transforms(args.img_size)
    dataset = NailSegDataset(images_dir, masks_dir, transforms)

    out_dir = Path("runs") / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    interpolation_issues = inspect_mask_interpolation(transforms)
    if interpolation_issues:
        print("Mask interpolation issues detected:")
        for issue in interpolation_issues:
            print(f"- {issue}")
    else:
        print("Mask interpolation check: all relevant transforms use NEAREST.")

    any_non_binary = False
    for idx in indices:
        image_tensor, mask_tensor = dataset[idx]
        mask_tensor = mask_tensor.squeeze(0)
        unique_vals, foreground_ratio, is_binary = summarize_mask(mask_tensor)
        any_non_binary = any_non_binary or not is_binary

        print(f"Sample {idx} unique mask values: {unique_vals}")
        print(f"Sample {idx} foreground ratio: {foreground_ratio:.4f}")
        if not is_binary:
            print(f"Sample {idx} mask contains non-binary values.")

        image_np = denormalize_image(image_tensor)
        mask_np = (mask_tensor > 0.5).cpu().numpy().astype(np.uint8) * 255
        overlay = overlay_mask(image_np, mask_np)
        Image.fromarray(overlay).save(out_dir / f"train_{idx}_overlay.png")

    print(f"Saved overlays to {out_dir}")

    if interpolation_issues or any_non_binary:
        print(
            "Conclusion: Potential interpolation issues detected (non-NEAREST mask interpolation "
            "or non-binary mask values). Inspect overlays for gray boundaries or misalignment."
        )
    else:
        print(
            "Conclusion: No evidence of interpolation-induced gray boundaries; masks remain binary "
            "and transforms use NEAREST for masks. Check saved overlays for any visual misalignment."
        )


if __name__ == "__main__":
    main()
