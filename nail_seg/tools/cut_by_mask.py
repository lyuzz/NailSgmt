import cv2
import numpy as np
from pathlib import Path
import argparse


# ======================
# 参数解析
# ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch cut images by mask (same folder)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="包含 image + mask 的文件夹路径"
    )
    return parser.parse_args()


# ======================
# 主逻辑
# ======================
def main():
    args = parse_args()
    root = Path(args.dir)

    if not root.exists():
        raise FileNotFoundError(f"❌ 文件夹不存在: {root}")

    # 找所有 png，但排除 mask / cut
    img_paths = sorted([
        p for p in root.glob("*.png")
        if not p.stem.endswith("_mask")
        and not p.stem.endswith("_cut")
    ])

    if not img_paths:
        raise RuntimeError("❌ 未找到可处理的原图")

    for img_path in img_paths:
        stem = img_path.stem                    # xxx
        mask_path = root / f"{stem}_mask.png"   # xxx_mask.png
        out_path = root / f"{stem}_cut.png"     # xxx_cut.png

        if not mask_path.exists():
            print(f"⚠️ 跳过 {stem}：未找到 mask")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"❌ 读取失败：{stem}")
            continue

        if img.shape[:2] != mask.shape[:2]:
            print(f"❌ 尺寸不一致：{stem}")
            continue

        # ======================
        # 二值化 mask
        # ======================
        _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # ======================
        # （可选）mask 修复
        # ======================
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        # ======================
        # 生成 RGBA
        # ======================
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, mask_bin])

        cv2.imwrite(str(out_path), rgba)
        print(f"✅ {stem} → {out_path.name}")

    print("🎉 全部处理完成")


if __name__ == "__main__":
    main()
