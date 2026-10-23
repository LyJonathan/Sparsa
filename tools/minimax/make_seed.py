#!/usr/bin/env python3
"""Reconstruct a layer's input (residual stream) from a reference dump, as an npy seed for the engine.

The reference (`tools/minimax/ref/model_ref.py`) captures a decoder layer's internals but not the residual
stream entering it.  It is recoverable exactly: `layer_out = x_in + attn_out + moe_out`, and the dumps hold
`layer_out`, `attn_out`, `shared_expert_out` and every hit expert's weighted output.  So

    x_in = layer_out - attn_out - (2 * sum_e expert_e_out + shared_expert_out)

with the routed scale (2.0 for M3.1).  `--minimax-seed-hidden` then feeds row t as token t's input, which lets a
one-layer GGUF be compared stage by stage against that layer of the full reference.

    python tools/minimax/make_seed.py --dump _local/ref_out/dumps/short/dump-L40 --out seed-L40.npy
"""

import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="a reference dump-LNN directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--routed-scale", type=float, default=2.0)
    args = ap.parse_args()

    d = args.dump
    layer_out = np.load(os.path.join(d, "layer_out.npy")).reshape(-1, 6144).astype(np.float64)
    attn_out = np.load(os.path.join(d, "attn_out.npy")).reshape(-1, 6144).astype(np.float64)
    shared = np.load(os.path.join(d, "shared_expert_out.npy")).reshape(-1, 6144).astype(np.float64)
    routed = np.zeros_like(layer_out)
    n_expert = 0
    for name in sorted(os.listdir(d)):
        if name.startswith("expert_") and name.endswith("_out.npy"):
            e = name[len("expert_"):-len("_out.npy")]
            out = np.load(os.path.join(d, name)).astype(np.float64)
            tok = np.load(os.path.join(d, f"expert_{e}_token_idx.npy")).reshape(-1).astype(np.int64)
            routed[tok] += out
            n_expert += 1
    x_in = layer_out - attn_out - (args.routed_scale * routed + shared)
    np.save(args.out, x_in.astype(np.float32))
    print(f"{os.path.basename(d)}: {layer_out.shape[0]} rows, {n_expert} experts, "
          f"x_in max abs {np.abs(x_in).max():.4f}, mean abs {np.abs(x_in).mean():.6f}")
    # Consistency: rmsnorm(x_in) must equal the dumped attn_input up to the (baked) norm weight.
    if os.path.exists(os.path.join(d, "attn_input.npy")):
        ai = np.load(os.path.join(d, "attn_input.npy")).reshape(-1, 6144)
        rms = np.sqrt((x_in * x_in).mean(axis=1, keepdims=True) + 1e-6)
        unit = x_in / rms
        # attn_input = unit * (1 + gamma); estimate the per-dimension ratio from the mean over rows
        g = (ai / unit).mean(axis=0)
        err = np.abs(ai - unit * g)
        print(f"  implied (1+gamma): mean {g.mean():.4f}, range {g.min():.4f}..{g.max():.4f}; "
              f"residual max abs {err.max():.3e}")


if __name__ == "__main__":
    main()
