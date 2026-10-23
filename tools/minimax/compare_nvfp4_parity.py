#!/usr/bin/env python3
"""Compare MM-NVFP4 engine layer outputs with the FP32 reference dumps.

The reference (`tools/minimax/ref`, `_local/ref_out/dumps/short|long`) writes one
directory per layer (`dump-L00`, `dump-L03`, `dump-L40`) containing `layer_out.npy`
[1, T, 6144].  `strata-minimax --minimax-dump-dir DIR --minimax-dump-layers 0,3`
writes `DIR/L0.npy`, `DIR/L3.npy` [T, 6144] for a truncated file whose source
layers come from `strata.source_layers`.

    python tools/minimax/compare_nvfp4_parity.py \
        --ref  _local/ref_out/dumps/short --ours E:/mm-nvfp4/out/dump-l03 --layers 0,3

Prints max |diff|, relative L2 and cosine per layer. The engine is fp32 with Q8_0
activations on the expert dot products and an fp16 KV cache, so the expected size
of a difference is the quantized-activation noise floor, not zero.
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def _rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a.reshape(a.shape[0], -1)


def stats(ref: np.ndarray, ours: np.ndarray) -> dict:
    ref = _rows(ref).astype(np.float64)
    ours = _rows(ours).astype(np.float64)
    d = np.abs(ref - ours)
    denom = np.linalg.norm(ref) * np.linalg.norm(ours)
    return {
        "rows": ref.shape[0],
        "cols": ref.shape[1],
        "max_abs": float(d.max()),
        "ref_absmax": float(np.abs(ref).max()),
        "rel_max": float(d.max() / np.abs(ref).max()) if np.abs(ref).max() > 0 else 0.0,
        "rel_l2": float(np.linalg.norm(ref - ours) / np.linalg.norm(ref)),
        "cos": float((ref * ours).sum() / denom) if denom > 0 else 0.0,
    }


def find_our_file(root: str, layer: int) -> str | None:
    for name in (f"L{layer}.npy", f"dump-L{layer}/layer_out.npy", f"dump-L{layer:02d}/layer_out.npy",
                 f"dump-L{layer:03d}/layer_out.npy"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference dumps root (holds dump-LNN/layer_out.npy)")
    ap.add_argument("--ours", required=True, help="engine --minimax-dump-dir")
    ap.add_argument("--layers", default="0,3,40")
    ap.add_argument("--tensor", default="layer_out")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    bad = 0
    for tok in args.layers.split(","):
        if not tok:
            continue
        layer = int(tok)
        ref_path = os.path.join(args.ref, f"dump-L{layer:02d}", args.tensor + ".npy")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(args.ref, f"dump-L{layer}", args.tensor + ".npy")
        our_path = find_our_file(args.ours, layer)
        if not os.path.exists(ref_path) or our_path is None:
            print(f"layer {layer:2d}: SKIP (ref {os.path.exists(ref_path)}, ours {our_path is not None})")
            continue
        ref = np.load(ref_path)
        ours = np.load(our_path)
        if ref.size != ours.size:
            print(f"layer {layer:2d}: FAIL shape {ref.shape} vs {ours.shape}")
            bad += 1
            continue
        s = stats(ref, ours)
        rows.append({"layer": layer, **s})
        print(f"layer {layer:2d}: T={s['rows']:4d}  max|diff|={s['max_abs']:.6e}  "
              f"ref max|.|={s['ref_absmax']:.4e}  rel_max={s['rel_max']:.3e}  rel_l2={s['rel_l2']:.3e}  cos={s['cos']:.9f}")
    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
