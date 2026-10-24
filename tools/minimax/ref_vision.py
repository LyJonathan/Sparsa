#!/usr/bin/env python3
"""ref_vision.py - transformers' fp32 CPU reference for the MiniMax-M3.1-Flash vision tower + projector.

Runs only the 0.87 B vision side (never the 427 B language model).  Input is the checkpoint processor's
`pixel_values` and `image_grid_thw` saved as .npy (float32 [n, 1176] and int64 [1, 3]); output is the merged
image embeddings [n/4, 6144], i.e. what the engine splices at the image tokens.

Needs transformers 5.20.2 (the first release with minimax_m3_vl), torch CPU and safetensors:

    python ref_vision.py --model <checkpoint-dir> --pixel-values a.pv.npy --grid a.grid.npy --out a.ref.npy

PyTorch's CPU float32 path is deterministic enough for the 0.999 cosine gate; the engine reads the same
pixel_values through tools/minimax/check_vision_parity.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLVisionModel, MiniMaxM3VLMultiModalProjector)
from transformers.vision_utils import get_vision_position_ids


def load_weights(model: Path, vision, proj):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    want = {k: v for k, v in index.items()
            if k.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp."))}
    by_shard = {}
    for k, shard in want.items():
        by_shard.setdefault(shard, []).append(k)
    raw = {}
    for shard, keys in by_shard.items():
        with safe_open(str(model / shard), framework="pt", device="cpu") as f:
            for k in keys:
                raw[k] = f.get_tensor(k).to(torch.float32)

    sd = {}
    for k, t in raw.items():
        if k.startswith("vision_tower.vision_model."):
            short = k[len("vision_tower.vision_model."):].replace("encoder.layers.", "layers.")
            sd[short.replace("embeddings.patch_embedding.", "embeddings.proj.")] = t
    missing, unexpected = vision.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    psd = {
        "linear_1.weight": raw["multi_modal_projector.linear_1.weight"],
        "linear_1.bias": raw["multi_modal_projector.linear_1.bias"],
        "linear_2.weight": raw["multi_modal_projector.linear_2.weight"],
        "linear_2.bias": raw["multi_modal_projector.linear_2.bias"],
        "merge_linear_1.weight": raw["patch_merge_mlp.linear_1.weight"],
        "merge_linear_1.bias": raw["patch_merge_mlp.linear_1.bias"],
        "merge_linear_2.weight": raw["patch_merge_mlp.linear_2.weight"],
        "merge_linear_2.bias": raw["patch_merge_mlp.linear_2.bias"],
    }
    missing, unexpected = proj.load_state_dict(psd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the HF checkpoint folder")
    ap.add_argument("--pixel-values", required=True, help="float32 .npy [n_patch, 1176] from the checkpoint processor")
    ap.add_argument("--grid", required=True, help="the processor's image_grid_thw .npy [1, 3]")
    ap.add_argument("--out", required=True, help="where the [n/4, 6144] fp32 .npy goes")
    args = ap.parse_args(argv)

    cfg = MiniMaxM3VLConfig.from_pretrained(args.model)
    vision = MiniMaxM3VLVisionModel._from_config(cfg.vision_config, attn_implementation="eager").eval().float()
    proj = MiniMaxM3VLMultiModalProjector(cfg).eval().float()
    load_weights(Path(args.model), vision, proj)
    for p in list(vision.parameters()) + list(proj.parameters()):
        p.requires_grad_(False)

    pv = torch.from_numpy(np.load(args.pixel_values)).float()
    grid = torch.from_numpy(np.load(args.grid))
    with torch.no_grad():
        pos = get_vision_position_ids(grid, cfg.vision_config.spatial_merge_size, include_temporal=True)
        hs = vision(pixel_values=pv, grid_thw=grid, position_ids=pos).last_hidden_state
        emb = proj(hs.squeeze(0))
    np.save(args.out, emb.numpy())
    print(f"{args.out}: {tuple(emb.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
