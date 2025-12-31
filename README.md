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
│   ├── README.md       # Detailed usage for training, inference, and export
│   └── nail_seg/       # Python package (datasets, models, losses, metrics)
└── tools/              # Supporting utilities (see tools/README.md)
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

## License

This project is licensed under the terms in [LICENSE](LICENSE).
