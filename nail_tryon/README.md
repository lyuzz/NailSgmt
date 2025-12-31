# Nail Try-On Tool

## Sequential Nail Virtual Try-On (Diffusers Inpaint + ONNX Nail Segmentation)

`nail_tryon/nail_tryon_inpaint.py` applies 5 nail design references to a hand photo by:

- Running an ONNX nail segmentation model to obtain up to 5 nail masks.
- Sorting nails left-to-right by centroid X to bind refs `[0..4]`.
- Inpainting each nail in a cropped ROI with diffusers.
- Compositing back using feathered mask alpha so pixels outside masks remain unchanged.
- Optionally exporting full-size RGBA layers and debug masks.

## Environment setup

Create a dedicated virtual environment inside the folder and install requirements:

```bash
cd nail_tryon
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

```bash
python nail_tryon/nail_tryon_inpaint.py \
  --img_path path/to/hand.jpg \
  --onnx_path path/to/nail_seg.onnx \
  --refs_dir path/to/refs_dir \
  --out_dir path/to/output \
  --model_id diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
  --seed 123 \
  --roi_margin 80 \
  --feather_px 6 \
  --strength 0.45 \
  --guidance 6.0 \
  --steps 30 \
  --export_layers 1 \
  --save_debug 1
```

### Outputs

- `out_dir/preview.png`
- `out_dir/nail_0.png ... out_dir/nail_4.png` (if `--export_layers=1`)
- `out_dir/mask_0.png ... out_dir/mask_4.png` (if `--save_debug=1`)

The script logs detected nail counts, centroids, and which ref is applied to each nail.
