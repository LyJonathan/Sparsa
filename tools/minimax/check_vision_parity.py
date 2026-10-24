#!/usr/bin/env python3
"""check_vision_parity.py - per-token cosine between the engine's image embeddings and the HF reference.

    python tools/minimax/check_vision_parity.py a.ref.npy a.engine.npy [--min 0.999]

Both files are [n_tokens, 6144] fp32.  Prints the minimum/mean row cosine and the maximum absolute difference,
and exits 1 when the minimum is below --min.  The engine writes its file with:

    strata-minimax --mmproj mmproj.gguf --minimax-vision-pixels a.pv.npy --minimax-vision-size 644x476 \
        --minimax-vision-out a.engine.npy
"""
from __future__ import annotations

import argparse

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("engine")
    ap.add_argument("--min", type=float, default=0.999)
    args = ap.parse_args(argv)
    ref = np.load(args.reference).astype(np.float64)
    got = np.load(args.engine).astype(np.float64)
    if ref.shape != got.shape:
        raise SystemExit(f"shapes differ: reference {ref.shape}, engine {got.shape}")
    a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    b = got / np.linalg.norm(got, axis=1, keepdims=True)
    cos = (a * b).sum(1)
    print(f"tokens={ref.shape[0]} dim={ref.shape[1]} cosine min={cos.min():.8f} mean={cos.mean():.8f} "
          f"maxabs={np.abs(ref - got).max():.4f} refmax={np.abs(ref).max():.2f}")
    return 0 if cos.min() >= args.min else 1


if __name__ == "__main__":
    raise SystemExit(main())
