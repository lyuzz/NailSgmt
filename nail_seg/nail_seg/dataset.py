from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NailSegDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform: Optional[Callable] = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        self.image_paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir}")

        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for image {img_path.name} at {mask_path}"
                )
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed["image"]
            mask_np = transformed["mask"]

        if isinstance(image_np, torch.Tensor):
            image_tensor = image_np
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        if isinstance(mask_np, torch.Tensor):
            mask_tensor = mask_np
        else:
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return image_tensor, mask_tensor


def build_train_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(p=0.3),
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
