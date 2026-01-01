from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NailSegDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: Optional[Callable] = None,
        mask_erosion_px: int = 0,
        roi_focus: bool = False,
        roi_padding: float = 0.15,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_erosion_px = mask_erosion_px
        self.roi_focus = roi_focus
        self.roi_padding = roi_padding

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

    def _erode_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_erosion_px <= 0:
            return mask
        kernel_size = 2 * self.mask_erosion_px + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return eroded.astype(np.float32)

    def _apply_roi_crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.roi_focus:
            return image, mask

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return image, mask

        height, width = mask.shape[:2]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        pad_x = int((x_max - x_min + 1) * self.roi_padding)
        pad_y = int((y_max - y_min + 1) * self.roi_padding)

        x_min = max(x_min - pad_x, 0)
        x_max = min(x_max + pad_x, width - 1)
        y_min = max(y_min - pad_y, 0)
        y_max = min(y_max + pad_y, height - 1)

        image = image[y_min : y_max + 1, x_min : x_max + 1]
        mask = mask[y_min : y_max + 1, x_min : x_max + 1]
        return image, mask

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.float32)
        mask_np = self._erode_mask(mask_np)
        image_np, mask_np = self._apply_roi_crop(image_np, mask_np)

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

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()

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
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=10, p=0.5),
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
