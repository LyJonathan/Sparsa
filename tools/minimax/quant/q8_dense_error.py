"""Q8_0 re-quantization error on the checkpoint's MXFP8 side (attention / dense / shared).

The engine has no MXFP8 ggml type, so MM-NVFP4 writes those tensors as Q8_0. This measures the
extra error that choice adds, per tensor, on real checkpoint tensors:

    rel_l2 = || Q8_0(dequant_mxfp8(W)) - dequant_mxfp8(W) || / || dequant_mxfp8(W) ||

(fp32 throughout; Q8_0's fp16 block scale is applied as ggml does.)
"""
import json, struct, sys
from pathlib import Path
import numpy as np

MODEL = Path(r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash")

_E4M3 = np.zeros(256, np.float32)
for b in range(256):
    m = b & 0x7F
    if m == 0 or m == 0x7F:
        v = 0.0
    else:
        e, man = (m >> 3) & 0xF, m & 7
        v = (man * 2.0 ** -9) if e == 0 else (1.0 + man / 8.0) * 2.0 ** (e - 7)
    _E4M3[b] = -v if b & 0x80 else v


def get(name, hdr_cache={}):
    idx = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    shard = MODEL / idx[name]
    if shard not in hdr_cache:
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            hdr_cache[shard] = (json.loads(f.read(hlen)), 8 + hlen)
    header, d0 = hdr_cache[shard]
    meta = header[name]
    a, b = meta["data_offsets"]
    dt = {"F8_E4M3": np.uint8, "U8": np.uint8, "F32": np.float32}[meta["dtype"]]
    arr = np.memmap(shard, dtype=dt, mode="r", offset=d0 + a, shape=((b - a) // np.dtype(dt).itemsize,))
    return np.asarray(arr).reshape(meta["shape"])


def dequant_mxfp8(w, s):
    d = np.exp2(s.astype(np.float32) - 127.0)
    return _E4M3[w] * np.repeat(d, 32, axis=1)


def q8_0_rel_l2(x):
    o, i = x.shape
    blocks = x.astype(np.float32).reshape(o, i // 32, 32)
    amax = np.abs(blocks).max(axis=2)
    d = (amax / 127.0).astype(np.float16).astype(np.float32)
    d[d == 0] = 1.0
    q = np.rint(blocks / d[:, :, None]).clip(-127, 127)
    back = (q * d[:, :, None]).reshape(o, i)
    return float(np.linalg.norm(back - x) / np.linalg.norm(x))


def main():
    cases = [
        ("language_model.model.layers.0.self_attn.q_proj", "layer 0 q_proj [8192,6144]"),
        ("language_model.model.layers.0.self_attn.o_proj", "layer 0 o_proj [6144,8192]"),
        ("language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj", "layer 3 shared gate [3072,6144]"),
        ("language_model.model.layers.3.block_sparse_moe.shared_experts.down_proj", "layer 3 shared down [6144,3072]"),
        ("language_model.model.layers.40.self_attn.index_q_proj", "layer 40 index q [512,6144]"),
        ("language_model.model.layers.0.mlp.gate_proj", "layer 0 dense gate [12288,6144]"),
    ]
    rels = []
    for base, label in cases:
        w = get(base + ".weight")
        s = get(base + ".weight_scale_inv")
        x = dequant_mxfp8(w, s)
        r = q8_0_rel_l2(x)
        rels.append(r)
        print(f"{label:34s} Q8_0 extra rel_l2 = {r:.4e}")
    print(f"\nmedian {np.median(rels):.4e}  max {max(rels):.4e}  ({len(rels)} tensors)")


if __name__ == "__main__":
    main()
