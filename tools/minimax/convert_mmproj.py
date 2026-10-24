#!/usr/bin/env python3
"""convert_mmproj.py - MiniMax-M3.1-Flash vision tower + projector -> llama.cpp `mmproj` GGUF (F16).

Reads only the vision-side tensors of the released checkpoint (`vision_tower.*`, `multi_modal_projector.*`,
`patch_merge_mlp.*`) and writes the file `strata-minimax --mmproj` and llama.cpp's mtmd both read:

    python tools/minimax/convert_mmproj.py <checkpoint-dir> -o mmproj-MiniMax-M3.1-Flash-F16.gguf

The layout follows llama.cpp's conversion/minimax.py (MIT): the Conv3d patch embedding is split into its two
temporal 2-D slices (`v.patch_embd.weight` and `.1`), ViT q/k are permuted from HF's [T H W | T H W | pad] to
llama.cpp's [T Tdup H Hdup W Wdup pad], and the projector is `mm.1`/`mm.2` + `mm.merger.fc1`/`fc2`.

The engine applies the reference rotation to that layout: with a single image (t = 0) the T pair is an
identity, so the H pair is (26+i, 39+i) and W is (52+i, 65+i) - transformers' 3-D RoPE with the temporal axis
at zero.  See src/minimax/vision.cpp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType

VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")

# HF -> (llama.cpp mmproj name, kind)
RENAME = {
    "vision_tower.vision_model.pre_layrnorm.weight": "v.pre_ln.weight",
    "vision_tower.vision_model.pre_layrnorm.bias": "v.pre_ln.bias",
    "multi_modal_projector.linear_1.weight": "mm.1.weight",
    "multi_modal_projector.linear_1.bias": "mm.1.bias",
    "multi_modal_projector.linear_2.weight": "mm.2.weight",
    "multi_modal_projector.linear_2.bias": "mm.2.bias",
    "patch_merge_mlp.linear_1.weight": "mm.merger.fc1.weight",
    "patch_merge_mlp.linear_1.bias": "mm.merger.fc1.bias",
    "patch_merge_mlp.linear_2.weight": "mm.merger.fc2.weight",
    "patch_merge_mlp.linear_2.bias": "mm.merger.fc2.bias",
}


def permute_vit_qk(t: torch.Tensor, n_head: int) -> torch.Tensor:
    """HF [Ta Ha Wa | Tb Hb Wb | pad] -> llama.cpp [Ta Tb | Ha Hb | Wa Wb | pad] (llama.cpp's converter)."""
    d_head = t.shape[0] // n_head
    axis_dim = 2 * ((2 * (d_head // 2) // 3) // 2)
    ah = axis_dim // 2
    half = 3 * ah
    perm = (list(range(0, ah)) + list(range(half, half + ah)) + list(range(ah, 2 * ah)) +
            list(range(half + ah, half + 2 * ah)) + list(range(2 * ah, 3 * ah)) +
            list(range(half + 2 * ah, half + 3 * ah)) + list(range(2 * half, d_head)))
    assert len(perm) == d_head and sorted(perm) == list(range(d_head))
    idx = torch.tensor(perm, dtype=torch.long)
    if t.ndim == 2:
        return t.reshape(n_head, d_head, t.shape[1])[:, idx, :].reshape(t.shape)
    return t.reshape(n_head, d_head)[:, idx].reshape(t.shape)


def layer_name(key: str) -> str | None:
    if key in RENAME:
        return RENAME[key]
    if not key.startswith("vision_tower.vision_model.encoder.layers."):
        return None
    rest = key[len("vision_tower.vision_model.encoder.layers."):]
    n, _, tail = rest.partition(".")
    tail = (tail.replace("layer_norm1.", "ln1.")
                .replace("layer_norm2.", "ln2.")
                .replace("self_attn.q_proj.", "attn_q.")
                .replace("self_attn.k_proj.", "attn_k.")
                .replace("self_attn.v_proj.", "attn_v.")
                .replace("self_attn.out_proj.", "attn_out.")
                .replace("mlp.fc1.", "ffn_up.")
                .replace("mlp.fc2.", "ffn_down."))
    return f"v.blk.{n}.{tail}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the HF checkpoint folder (config.json + model-*.safetensors)")
    ap.add_argument("-o", "--outfile", required=True, help="the mmproj GGUF to write")
    args = ap.parse_args(argv)
    model = Path(args.model)
    cfg = json.loads((model / "config.json").read_text(encoding="utf-8"))
    vc = cfg["vision_config"]
    comp = vc.get("img_token_compression_config", {})

    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    wanted = {k: v for k, v in index.items() if k.startswith(VISION_PREFIXES)}
    by_shard: dict[str, list[str]] = {}
    for k, shard in wanted.items():
        by_shard.setdefault(shard, []).append(k)
    print(f"vision tensors: {len(wanted)} in {len(by_shard)} shard(s)")

    out = GGUFWriter(str(args.outfile), "clip", use_temp_file=False)
    out.add_type("mmproj")
    out.add_name("MiniMax M3.1 Flash")
    out.add_size_label("866M")
    out.add_file_type(1)
    out.add_quantization_version(2)
    out.add_license("other")
    out.add_license_name("minimax-community")
    out.add_license_link("LICENSE")
    out.add_tags(["multimodal", "moe", "agent", "coding", "video", "image-text-to-text"])
    out.add_sampling_temp(1.0)
    out.add_sampling_top_p(0.95)
    out.add_clip_has_vision_encoder(True)
    out.add_clip_projector_type("minimax_m3")
    out.add_vision_projection_dim(int(cfg.get("projector_hidden_size") or cfg["text_config"]["hidden_size"]))
    out.add_vision_image_size(int(vc.get("image_size") or 672))
    out.add_vision_patch_size(int(vc["patch_size"]))
    out.add_vision_embedding_length(int(vc["hidden_size"]))
    out.add_vision_feed_forward_length(int(vc["intermediate_size"]))
    out.add_vision_block_count(int(vc["num_hidden_layers"]))
    out.add_vision_head_count(int(vc["num_attention_heads"]))
    out.add_vision_image_mean([float(x) for x in (0.48145466, 0.4578275, 0.40821073)])
    out.add_vision_image_std([float(x) for x in (0.26862954, 0.26130258, 0.27577711)])
    out.add_vision_attention_layernorm_eps(float(vc.get("layer_norm_eps", 1e-5)))
    out.add_vision_spatial_merge_size(int(comp.get("spatial_merge_size", 2)))
    out.add_vision_use_gelu(True)

    n_head = int(vc["num_attention_heads"])
    written = 0
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(model / shard), framework="pt", device="cpu") as f:
            for key in keys:
                t = f.get_tensor(key).to(torch.float32)
                if key == "vision_tower.vision_model.embeddings.patch_embedding.weight":
                    if t.ndim != 5:
                        raise SystemExit(f"unexpected patch_embedding rank {t.ndim}")
                    base = "v.patch_embd.weight"
                    for k in range(t.shape[2]):
                        name = base if k == 0 else f"{base}.{k}"
                        out.add_tensor(name, t[:, :, k, ...].to(torch.float16).contiguous().numpy(),
                                       raw_dtype=GGMLQuantizationType.F16)
                        written += 1
                    continue
                name = layer_name(key)
                if name is None:
                    raise SystemExit(f"no mmproj mapping for {key}")
                if ".attn_q." in name or ".attn_k." in name:
                    t = permute_vit_qk(t, n_head)
                dtype = GGMLQuantizationType.F32 if t.ndim == 1 else GGMLQuantizationType.F16
                arr = t.float().contiguous().numpy() if t.ndim == 1 else t.half().contiguous().numpy()
                out.add_tensor(name, arr, raw_dtype=dtype)
                written += 1
        print(f"  {shard}: done")
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file(progress=True)
    out.close()
    print(f"wrote {args.outfile} ({written} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
