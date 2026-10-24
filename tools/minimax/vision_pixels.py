#!/usr/bin/env python3
"""vision_pixels.py - the checkpoint processor's pixel_values for the vision parity check.

    python tools/minimax/vision_pixels.py <checkpoint-dir> image.png --out a

Writes `a.pv.npy` (float32 [n_patch, 1176]) and `a.grid.npy` (int64 [1,3]) with the official MiniMax processor
(`image_processor.py` from the checkpoint, bicubic + smart_resize + mean/std).  Then:

    strata-minimax --mmproj mmproj.gguf --minimax-vision-pixels a.pv.npy --minimax-vision-size 644x476 \
        --minimax-vision-out a.engine.npy
    python ref_vision.py --model <checkpoint-dir> --pixel-values a.pv.npy --grid a.grid.npy --out a.ref.npy
    python check_vision_parity.py a.ref.npy a.engine.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the HF checkpoint folder")
    ap.add_argument("image")
    ap.add_argument("--out", help="output prefix (default: the image's path)")
    args = ap.parse_args(argv)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    im = Image.open(args.image).convert("RGB")
    batch = proc(images=[im], text="]<]image[>[", return_tensors="pt")
    pv, grid = batch["pixel_values"], batch["image_grid_thw"]
    out = Path(args.out or args.image)
    np.save(str(out) + ".pv.npy", pv.numpy())
    np.save(str(out) + ".grid.npy", grid.numpy())
    h = int(grid[0, 1]) * 14
    w = int(grid[0, 2]) * 14
    print(f"{args.image}: {im.width}x{im.height} -> {w}x{h}, {int(grid.prod()) // 4} image tokens; "
          f"{out}.pv.npy {tuple(pv.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
