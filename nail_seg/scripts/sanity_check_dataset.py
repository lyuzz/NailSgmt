from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check nail dataset")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5)
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


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images" / args.split
    masks_dir = data_dir / "masks" / args.split

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing images/masks directory for split {args.split}")

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    mask_paths = sorted([p for p in masks_dir.iterdir() if p.suffix.lower() == ".png"])

    image_map = {p.stem: p for p in image_paths}
    image_stems = set(image_map.keys())
    mask_stems = {p.stem for p in mask_paths}
    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    print(f"Images: {len(image_paths)}")
    print(f"Masks: {len(mask_paths)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Missing images: {len(missing_images)}")

    if missing_masks:
        print("Example missing mask stems:", missing_masks[:5])
    if missing_images:
        print("Example missing image stems:", missing_images[:5])

    if args.save_overlays:
        out_dir = data_dir / "overlays" / args.split
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in list(image_stems & mask_stems)[: args.num_samples]:
            image = np.array(Image.open(image_map[stem]).convert("RGB"))
            mask_path = masks_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path).convert("L"))
            overlay = overlay_mask(image, mask)
            Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")
        print(f"Saved overlays to {out_dir}")


if __name__ == "__main__":
    main()
