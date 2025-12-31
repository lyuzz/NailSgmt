# NailSgmt

NailSgmt is a lightweight fingernail segmentation project built around a Mobile-UNet model. It includes training, inference, and ONNX export utilities for creating a binary mask that separates fingernail pixels from background.

## What’s inside

- **Training pipeline** with configurable dataset paths, image sizes, and optimization settings.
- **Inference script** for running a trained ONNX model on a directory of images.
- **ONNX export** for deploying the best checkpoint.
- **Helper tools** for dataset preparation and auxiliary workflows.

## Repository layout

```
.
├── nail_seg/           # Main package, training/inference/export scripts
│   └── README.md       # Detailed usage for training, inference, and export
├── nail_sort/          # Python package (datasets, models, losses, metrics, sorting)
└── nail_tryon/         # Nail virtual try-on utility (see nail_tryon/README.md)
```

## Quick start

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r nail_seg/requirements.txt
   ```

2. Prepare your dataset in the expected directory structure:

   ```text
   <data_dir>/
     images/
       train/
       val/
     masks/
       train/
       val/
   ```

3. Train a model:

   ```bash
   python nail_seg/train.py --data_dir data --img_size 256 --batch_size 16 --epochs 50 --lr 3e-4
   ```

4. Export to ONNX:

   ```bash
   python nail_seg/export_onnx.py --checkpoint runs/<timestamp>/best.pt
   ```

5. Run inference:

   ```bash
   python nail_seg/infer.py --onnx runs/<timestamp>/best.onnx --input_dir data/test
   ```

For full details on dataset preparation, troubleshooting, and advanced options, see [`nail_seg/README.md`](nail_seg/README.md).

The `nail_sort` module orders the five detected fingers from thumb to pinky
(大拇指 → 食指 → 中指 → 无名指 → 小拇指) for downstream template generation.

## Nail virtual try-on setup

The virtual try-on script has its own virtual environment inside `nail_tryon/` so
the dependency set stays isolated from the training stack.

```bash
cd nail_tryon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Usage details and flags are documented in [`nail_tryon/README.md`](nail_tryon/README.md).

## License

This project is licensed under the terms in [LICENSE](LICENSE).
