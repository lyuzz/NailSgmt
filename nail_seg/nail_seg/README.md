# Step 2 Template Pipeline

This module builds on Step 1 masks to produce per-nail cutouts, normalized nails, and a template board image.

## Install Dependencies

From the repo root:

```bash
pip install -r nail_seg/requirements.txt
```

## Usage

Single image (auto-detect masks/json):

```bash
python -m nail_seg.step2_template \
  --input path/to/image.png \
  --output runs/step2_demo \
  --mode auto \
  --canvas_w 1024 --canvas_h 512 \
  --slot_w 220 --slot_h 360 \
  --layout five \
  --bg white \
  --min_area 800 \
  --margin 10 \
  --sort left_to_right \
  --debug
```

Directory input (auto mode only):

```bash
python -m nail_seg.step2_template \
  --input path/to/images_dir \
  --output runs/step2_batch \
  --debug
```

Explicit mask directory (single image):

```bash
python -m nail_seg.step2_template \
  --input path/to/image.png \
  --output runs/step2_demo \
  --mode mask_dir \
  --mask_dir path/to/masks
```

JSON polygons (single image):

```bash
python -m nail_seg.step2_template \
  --input path/to/image.png \
  --output runs/step2_demo \
  --mode json \
  --json path/to/annotations.json
```

## Outputs

Given `--output OUT`, the pipeline writes:

```
OUT/
  nails/
    nail_00.png
    nail_00_mask.png
    nail_00_norm.png
  template/
    board.png
    meta.json
  debug/
    overlay.png
    contours.png
```

If the input is a directory, outputs are grouped by image stem:

```
OUT/
  image_stem_1/
    nails/
    template/
    debug/
  image_stem_2/
    ...
```
