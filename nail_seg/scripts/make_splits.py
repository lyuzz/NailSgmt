from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val splits")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Images or masks directory does not exist.")

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise RuntimeError("No images found to split.")

    image_map = {p.stem: p for p in image_paths}
    stems = list(image_map.keys())
    for stem in stems:
        mask_path = masks_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for stem: {stem}")

    random.seed(args.seed)
    random.shuffle(stems)
    split_idx = int(len(stems) * args.train_ratio)
    train_stems = stems[:split_idx]
    val_stems = stems[split_idx:]

    for split_name, split_stems in [("train", train_stems), ("val", val_stems)]:
        split_img_dir = out_dir / "images" / split_name
        split_mask_dir = out_dir / "masks" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            image_path = image_map[stem]
            mask_path = masks_dir / f"{stem}.png"
            shutil.copy2(image_path, split_img_dir / image_path.name)
            shutil.copy2(mask_path, split_mask_dir / mask_path.name)

    print(f"Train: {len(train_stems)} samples")
    print(f"Val: {len(val_stems)} samples")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
