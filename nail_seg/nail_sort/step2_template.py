"""Step 2 pipeline: cutout, normalize, and template board assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .postprocess.extract import NailInstance, extract_nails, save_cutouts
from .postprocess.io_detect import (
    load_image_bgr,
    load_instance_masks,
    resolve_input_images,
)
from .postprocess.normalize import normalize_nail
from .postprocess.template import build_template
from .postprocess.viz import save_contour_debug, save_overlay_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2 nail template pipeline")
    parser.add_argument("--input", required=True, type=str, help="Image path or directory")
    parser.add_argument("--output", required=True, type=str, help="Output directory")
    parser.add_argument("--mode", choices=["auto", "mask_dir", "json"], default="auto")
    parser.add_argument("--mask_dir", type=str, help="Directory with instance masks")
    parser.add_argument("--json", dest="json_path", type=str, help="JSON annotations file")
    parser.add_argument("--canvas_w", type=int, default=1024)
    parser.add_argument("--canvas_h", type=int, default=512)
    parser.add_argument("--slot_w", type=int, default=220)
    parser.add_argument("--slot_h", type=int, default=360)
    parser.add_argument("--layout", choices=["five", "ten"], default="five")
    parser.add_argument("--bg", choices=["white", "transparent"], default="white")
    parser.add_argument("--min_area", type=int, default=800)
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--sort", choices=["left_to_right", "area_desc"], default="left_to_right")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def sort_nails(nails: list[NailInstance], mode: str) -> list[NailInstance]:
    if mode == "left_to_right":
        sorted_nails = sorted(nails, key=lambda n: n.bbox_full[0] + n.bbox_full[2] / 2)
    elif mode == "area_desc":
        sorted_nails = sorted(nails, key=lambda n: n.area, reverse=True)
    else:
        raise ValueError(f"Unsupported sort mode: {mode}")
    for idx, nail in enumerate(sorted_nails):
        nail.index = idx
    return sorted_nails


def _serialize_params(args: argparse.Namespace) -> dict:
    return {
        "mode": args.mode,
        "canvas_w": args.canvas_w,
        "canvas_h": args.canvas_h,
        "slot_w": args.slot_w,
        "slot_h": args.slot_h,
        "layout": args.layout,
        "bg": args.bg,
        "min_area": args.min_area,
        "margin": args.margin,
        "sort": args.sort,
        "debug": args.debug,
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    images = resolve_input_images(input_path)
    if input_path.is_dir() and args.mode != "auto":
        raise ValueError("Directory input is only supported with --mode auto")

    for image_path in images:
        image_bgr = load_image_bgr(image_path)
        image_shape = image_bgr.shape[:2]

        mask_source = load_instance_masks(
            image_path,
            image_shape,
            mode=args.mode,
            mask_dir=Path(args.mask_dir) if args.mask_dir else None,
            json_path=Path(args.json_path) if args.json_path else None,
        )

        nails = extract_nails(image_bgr, mask_source.masks, args.min_area, args.margin)
        if not nails:
            raise ValueError(f"No nails found after filtering for {image_path}")

        nails = sort_nails(nails, args.sort)

        base_dir = output_path / image_path.stem if input_path.is_dir() else output_path
        nails_dir = base_dir / "nails"
        template_dir = base_dir / "template"
        debug_dir = base_dir / "debug"

        nails_dir.mkdir(parents=True, exist_ok=True)
        template_dir.mkdir(parents=True, exist_ok=True)
        if args.debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        save_cutouts(nails_dir, nails)

        normalized_items: list[dict] = []
        for nail in nails:
            norm_rgba, norm_meta = normalize_nail(
                nail.rgba_cutout, nail.mask_cutout, args.slot_w, args.slot_h
            )
            norm_path = nails_dir / f"nail_{nail.index:02d}_norm.png"
            Image.fromarray(norm_rgba).save(norm_path)
            normalized_items.append({"nail": nail, "meta": norm_meta, "path": norm_path, "rgba": norm_rgba})

        total_slots = 5 if args.layout == "five" else 10
        kept_items = normalized_items[:total_slots]
        discarded_items = normalized_items[total_slots:]

        board_rgba, placements = build_template(
            [item["rgba"] for item in kept_items],
            layout=args.layout,
            canvas_w=args.canvas_w,
            canvas_h=args.canvas_h,
            bg=args.bg,
            gap=args.margin,
        )

        board_path = template_dir / "board.png"
        if args.bg == "white":
            board_rgb = board_rgba[:, :, :3]
            Image.fromarray(board_rgb).save(board_path)
        else:
            Image.fromarray(board_rgba).save(board_path)

        meta = {
            "source_image": image_path.as_posix(),
            "params": _serialize_params(args),
            "nails": [
                {
                    "index": item["nail"].index,
                    "area": item["nail"].area,
                    "bbox_full": list(item["nail"].bbox_full),
                    "norm_meta": item["meta"],
                    "files": {
                        "cutout": f"nails/nail_{item['nail'].index:02d}.png",
                        "mask": f"nails/nail_{item['nail'].index:02d}_mask.png",
                        "norm": f"nails/nail_{item['nail'].index:02d}_norm.png",
                    },
                }
                for item in normalized_items
            ],
            "placements": placements,
            "discarded": [
                {
                    "index": item["nail"].index,
                    "area": item["nail"].area,
                }
                for item in discarded_items
            ],
        }

        meta_path = template_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")

        if args.debug:
            save_overlay_debug(image_bgr, mask_source.masks, debug_dir / "overlay.png")
            save_contour_debug(image_shape, mask_source.masks, debug_dir / "contours.png")

        print(f"Processed {image_path} -> {base_dir}")


if __name__ == "__main__":
    main()
