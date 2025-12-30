# nail_seg

Binary fingernail segmentation (background vs fingernail) using a lightweight Mobile-UNet.

## Dataset preparation
Expected directory structure:

```
<data_dir>/
  images/
    train/
    val/
  masks/
    train/
    val/
```

- Images: `*.jpg` or `*.png`
- Masks: `*.png`
- Mask filename must match image stem (e.g., `img_01.jpg` → `img_01.png`)
- Masks are single-channel; any value > 0 is treated as nail.

Use `scripts/make_splits.py` if you have a flat dataset.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --data_dir data --img_size 256 --batch_size 16 --epochs 50 --lr 3e-4
```

Resume training:

```bash
python train.py --data_dir data --resume runs/<timestamp>/last.pt
```

## Export to ONNX

```bash
python export_onnx.py --checkpoint runs/<timestamp>/best.pt --out_path runs/<timestamp>/model.onnx
```

## Inference

```bash
python infer.py --onnx runs/<timestamp>/model.onnx --image path/to/image.jpg --out_dir outputs
```

## Troubleshooting

- **Missing masks**: Ensure every image has a matching mask in `masks/<split>` with the same stem.
- **Shape mismatch**: Ensure masks are single-channel PNG files and images/masks match dimensions before resizing.
- **CPU training tips**: Reduce `--batch_size`, increase `--num_workers 0`, and lower `--img_size` if needed.
