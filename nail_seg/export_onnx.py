from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from unet.models import MobileUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export nail segmentation model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    model = MobileUNet(encoder_pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    out_dir = checkpoint_path.parent
    out_name = f"{checkpoint_path.stem}.onnx"
    out_path = out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"ONNX output already exists: {out_path}")

    opset_version = 18
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["prob"],
        dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
    )
    print(f"Exported ONNX opset version: {opset_version}")

    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"input": dummy.cpu().numpy()})
    prob = outputs[0]
    if prob.shape[1] != 1:
        raise RuntimeError(f"Unexpected output shape: {prob.shape}")
    if not (0.0 <= prob.min() and prob.max() <= 1.0):
        raise RuntimeError("ONNX output is not in [0, 1] range")

    print(f"ONNX model exported to: {out_path}")


if __name__ == "__main__":
    main()
