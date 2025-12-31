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

## Installation (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA acceleration, install the NVIDIA GPU build of PyTorch that matches your CUDA version:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Replace `cu126` with the CUDA version supported by your driver (see the PyTorch install guide).

## Training

```bash
python train.py --data_dir data --epochs 50
```

GPU training (CUDA):

```bash
python train.py --data_dir data --device cuda --epochs 50 --save_samples
```

If you have multiple GPUs, specify an index (e.g. `--device cuda:1`).

Resume training:

```bash
python train.py --data_dir data --resume runs/<timestamp>/last.pt
```

## Export to ONNX

```bash
python export_onnx.py --checkpoint runs/<timestamp>/best.pt
```
The exported file will be saved alongside the checkpoint as `runs/<timestamp>/best.onnx`.

## Inference

```bash
python infer.py --onnx runs/<timestamp>/best.onnx --input_dir data/test
```
Place the test dataset in the data/test directory before running the command.

## Troubleshooting

- **Missing masks**: Ensure every image has a matching mask in `masks/<split>` with the same stem.
- **Shape mismatch**: Ensure masks are single-channel PNG files and images/masks match dimensions before resizing.
- **CPU training tips**: Reduce `--batch_size`, increase `--num_workers 0`, and lower `--img_size` if needed.
- **Python 3.10** has been tested and is compatible.
