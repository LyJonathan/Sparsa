#!/usr/bin/env python3
"""Rebuild the routed-expert input of one layer from a reference dump.

The per-op dumps record `attn_input` (the input_layernorm output fed to
self-attention), `attn_out` (the attention branch before its residual add) and
`router_logits`. The expert input is

    x = post_attention_layernorm(residual + attn_out)

with `residual` the layer input. It is not dumped directly, but RMSNorm only
changes the scale of `residual`, so `residual` is `attn_input / (w_in + 1)`
times one unknown scalar per token. That scalar is fitted against the dumped
router logits (128 equations per token). The result is checked against the
dumped `expert_XXX_gate_up` of every expert that has tokens.

The script needs torch + safetensors and the checkpoint's config/index. It
reads only the tensors it needs from the shard that holds the layer.

Example:
    python exp_input.py --model models/MiniMax-M3.1-Flash \
        --dump _local/ref_out/dumps/short/dump-L40 --layer 40 \
        --out exp-x-L40.npy
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def load_layer_tensors(model: str, layer: int):
    from safetensors import safe_open

    with open(os.path.join(model, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)["weight_map"]
    base = f"language_model.model.layers.{layer}"
    names = {
        "in": base + ".input_layernorm.weight",
        "post": base + ".post_attention_layernorm.weight",
        "gate": base + ".block_sparse_moe.gate.weight",
    }
    shard = index[names["in"]]
    with safe_open(os.path.join(model, shard), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(n).float().numpy().astype(np.float64) for k, n in names.items()}


def reconstruct(model: str, dump: str, layer: int, eps: float = 1e-6):
    tensors = load_layer_tensors(model, layer)
    ai = np.load(os.path.join(dump, "attn_input.npy")).reshape(-1, tensors["in"].shape[0])
    ao = np.load(os.path.join(dump, "attn_out.npy")).reshape(-1, ai.shape[1])
    logits = np.load(os.path.join(dump, "router_logits.npy")).reshape(ai.shape[0], -1)

    u = ai / (tensors["in"] + 1.0)
    post_w = tensors["post"] + 1.0
    gate_w = tensors["gate"]

    def cost(t: int, s: float) -> float:
        x = rms_norm(s * u[t] + ao[t], post_w, eps)
        p = x @ gate_w.T
        return float(np.linalg.norm(p - logits[t]) / np.linalg.norm(logits[t]))

    # Golden-section search on log(s); the dump scales land around 20-40.
    lo, hi = np.log(1e-2), np.log(2e3)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    scales = np.empty(ai.shape[0])
    for t in range(ai.shape[0]):
        a, b = lo, hi
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = cost(t, np.exp(c)), cost(t, np.exp(d))
        for _ in range(28):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = cost(t, np.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = cost(t, np.exp(d))
        scales[t] = np.exp((a + b) / 2.0)

    x = np.stack([rms_norm(scales[t] * u[t] + ao[t], post_w, eps) for t in range(ai.shape[0])])
    return x.astype(np.float32), scales, logits


def verify(model: str, dump: str, layer: int, x: np.ndarray) -> list[tuple[str, float]]:
    from safetensors import safe_open

    with open(os.path.join(model, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)["weight_map"]
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts."
    out = []
    for name in sorted(os.listdir(dump)):
        if not (name.startswith("expert_") and name.endswith("_gate_up.npy")):
            continue
        e = name[len("expert_") : -len("_gate_up.npy")]
        ref = np.load(os.path.join(dump, name)).astype(np.float64)
        idx = np.load(os.path.join(dump, f"expert_{e}_token_idx.npy"))
        # gate_up = [gate; up] in one 2*inter wide vector
        guard = np.load(os.path.join(dump, name))
        width = guard.shape[-1] // 2
        pred = []
        for k in ("w1", "w3"):
            key = base + f"{int(e)}.{k}"
            shard = index[key + ".weight_packed"]
            with safe_open(os.path.join(model, shard), framework="pt", device="cpu") as f:
                packed = f.get_tensor(key + ".weight_packed").numpy()
                scale = f.get_tensor(key + ".weight_scale").float().numpy().astype(np.float64)
                g = float(f.get_tensor(key + ".weight_global_scale"))
            w = dequant_nvfp4(packed, scale, g)
            pred.append(x[idx] @ w.T)
        pred = np.concatenate(pred, axis=1)
        out.append((name, float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))))
        assert pred.shape[-1] == 2 * width
    return out


def dequant_nvfp4(packed: np.ndarray, scale: np.ndarray, global_scale: float) -> np.ndarray:
    mag = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
    lo = (packed & 0x0F).astype(np.int32)
    hi = (packed >> 4).astype(np.int32)
    nib = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int32)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    return (mag[nib & 7] * np.where(nib & 8, -1, 1)) * np.repeat(scale, 16, axis=1) / global_scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    x, scales, _ = reconstruct(args.model, args.dump, args.layer, args.eps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, x)
    print(f"wrote {args.out} shape {x.shape}")
    print(f"fitted residual scale min {scales.min():.3f} median {np.median(scales):.3f} max {scales.max():.3f}")
    if not args.no_verify:
        checks = verify(args.model, args.dump, args.layer, x)
        if not checks:
            print("no expert_XXX_gate_up dumps to verify against")
        for name, err in checks:
            print(f"verify {name}: {err:.3e}")
        worst = max((e for _, e in checks), default=0.0)
        print(f"verify worst {worst:.3e} over {len(checks)} experts")
        if worst > 5e-3:
            print("WARNING: reconstruction error is above 5e-3; the activation file may be wrong")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
