#!/usr/bin/env python3
"""Check an MM-NVFP4 expert tensor against spec A, byte for byte.

Reads the GGUF's GGML_TYPE_NVFP4 blocks with its own parser (no ggml, no gguf-py),
dequantizes them with the side divisor `blk.N.ffn_*_exps.scale`, dequantizes the
source checkpoint with spec A, and compares in fp32. The pass condition is
max |diff| == 0.

    python tools/minimax/quant/verify_nvfp4_gguf.py \
        --gguf E:/mm-nvfp4/MM-NVFP4-L0-3.gguf \
        --model E:/.../models/MiniMax-M3.1-Flash

Layer mapping comes from `strata.source_layers` in the GGUF. Every routed expert
matrix of every expert tensor in the file is checked (gate, up, down).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# E4M3 (bias 7) byte -> fp32, exactly as dequant.hpp's e4m3_to_f32.
_E4M3 = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _mag = _b & 0x7F
    if _mag == 0 or _mag == 0x7F:
        _v = 0.0
    else:
        _exp = (_mag >> 3) & 0xF
        _man = _mag & 0x7
        _v = (float(_man) * 2.0 ** -9) if _exp == 0 else (1.0 + _man / 8.0) * (2.0 ** (_exp - 7))
    _E4M3[_b] = -_v if (_b & 0x80) else _v

GGUF_MAGIC = 0x46554747
# ggml type ids this file can hold
TYPE_F32, TYPE_Q8_0, TYPE_Q6_K, TYPE_NVFP4 = 0, 8, 14, 40
TYPE_NAMES = {0: "f32", 8: "q8_0", 14: "q6_k", 40: "nvfp4"}

KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        magic, version = struct.unpack("<II", self.f.read(8))
        if magic != GGUF_MAGIC or version != 3:
            raise SystemExit(f"{path}: not a GGUF v3 file (magic {magic:#x}, version {version})")
        n_tensors, n_kv = struct.unpack("<QQ", self.f.read(16))
        self.meta = {}
        for _ in range(n_kv):
            key = self._str()
            vtype = struct.unpack("<I", self.f.read(4))[0]
            self.meta[key] = self._value(vtype)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            n_dims = struct.unpack("<I", self.f.read(4))[0]
            dims = list(struct.unpack("<" + "Q" * n_dims, self.f.read(8 * n_dims)))
            ttype = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (dims, ttype, off)
        align = int(self.meta.get("general.alignment", 32))
        pos = self.f.tell()
        self.data_start = (pos + align - 1) // align * align

    def _str(self) -> str:
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8")

    def _value(self, vtype):
        if vtype in KV_FIXED:
            raw = self.f.read(KV_FIXED[vtype])
            if vtype == 8:
                return raw.decode("utf-8")
            if vtype == 7:
                return raw != b"\x00"
            return struct.unpack("<" + {1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 10: "Q", 11: "q", 12: "d"}[vtype], raw)[0]
        if vtype == 8:
            return self._str()
        if vtype == 9:
            etype, count = struct.unpack("<IQ", self.f.read(12))
            return [self._value(etype) for _ in range(count)]
        raise SystemExit(f"unknown GGUF KV type {vtype}")

    def tensor(self, name):
        if name not in self.tensors:
            raise SystemExit(f"tensor {name} is not in the GGUF")
        dims, ttype, off = self.tensors[name]
        if ttype == TYPE_NVFP4:
            nbytes = (dims[0] // 64) * 36 * int(np.prod(dims[1:]))
        elif ttype == TYPE_Q8_0:
            nbytes = (dims[0] // 32) * 34 * int(np.prod(dims[1:]))
        elif ttype == TYPE_F32:
            nbytes = int(np.prod(dims)) * 4
        else:
            raise SystemExit(f"{name}: verification does not handle {TYPE_NAMES.get(ttype, ttype)}")
        self.f.seek(self.data_start + off)
        buf = self.f.read(int(nbytes))
        if len(buf) != int(nbytes):
            raise SystemExit(f"{name}: short read")
        return dims, ttype, np.frombuffer(buf, dtype=np.uint8)


def dequant_ggml_nvfp4(raw: np.ndarray, n_out: int, n_in: int, g: float) -> np.ndarray:
    """GGUF block layout: 36 B = 4 E4M3 scales then 32 B; in each 16, lo = j, hi = j+8."""
    nblk = n_in // 64
    blocks = raw.reshape(n_out, nblk, 36)
    out = np.empty((n_out, n_in), dtype=np.float32)
    for b in range(nblk):
        blk = blocks[:, b, :]
        sc = _E4M3[blk[:, :4]]                                   # [O, 4] fp32
        q = blk[:, 4:]                                           # [O, 32]
        lo = (q & 0x0F).astype(np.int64).reshape(n_out, 4, 8)
        hi = (q >> 4).astype(np.int64).reshape(n_out, 4, 8)
        sub = np.empty((n_out, 4, 16), dtype=np.int64)           # j (0..7) then j+8 (8..15)
        sub[:, :, 0:8] = lo
        sub[:, :, 8:16] = hi
        val = np.where(sub & 8, -E2M1[sub & 7], E2M1[sub & 7]).astype(np.float32)
        # (val * scale) / G, elementwise fp32, the same order as dequant_nvfp4_row.
        out[:, b * 64:(b + 1) * 64] = ((val * sc[:, :, None]) / np.float32(g)).reshape(n_out, 64)
    return out


def dequant_spec_a(packed: np.ndarray, scales: np.ndarray, g: float) -> np.ndarray:
    n_out = packed.shape[0]
    n_in = packed.shape[1] * 2
    codes = np.empty((n_out, n_in), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    val = np.where(codes & 8, -E2M1[codes & 7], E2M1[codes & 7]).astype(np.float32)
    sc = np.repeat(_E4M3[scales], 16, axis=1)
    return (val * sc) / np.float32(g)


class Safetensors:
    def __init__(self, model_dir: Path):
        idx_path = model_dir / "model.safetensors.index.json"
        if idx_path.exists():
            weight_map = json.loads(idx_path.read_text())["weight_map"]
        else:
            weight_map = None
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._cache = {}

    def _shard(self, name: str):
        path = self.model_dir / (self.weight_map[name] if self.weight_map else "model.safetensors")
        if path not in self._cache:
            with open(path, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
            self._cache[path] = (header, 8 + hlen)
        return path, *self._cache[path]

    def get(self, name: str):
        path, header, data0 = self._shard(name)
        meta = header[name]
        a, b = meta["data_offsets"]
        dtype = meta["dtype"]
        shape = meta["shape"]
        dt = {"U8": np.uint8, "F32": np.float32, "F8_E4M3": np.uint8, "BF16": np.uint16}[dtype]
        arr = np.memmap(path, dtype=dt, mode="r", offset=data0 + a, shape=(b - a) // np.dtype(dt).itemsize)
        return np.asarray(arr).reshape(shape), dtype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--max-experts", type=int, default=0, help="0: every expert")
    args = ap.parse_args()

    r = Reader(args.gguf)
    src_layers = r.meta.get("strata.source_layers")
    if src_layers is None:
        raise SystemExit("GGUF has no strata.source_layers")
    st = Safetensors(args.model)

    roles = {"ffn_gate_exps": "w1", "ffn_up_exps": "w3", "ffn_down_exps": "w2"}
    blks = sorted({int(n.split(".")[1]) for n in r.tensors if n.endswith("_exps.weight")})
    print(f"gguf {args.gguf.name}: {len(r.tensors)} tensors, block_count {r.meta.get('minimax-m3.block_count')}, "
          f"source layers {src_layers}, expert blocks {blks}")

    checked = 0
    total_elems = 0
    worst = 0.0
    for blk in blks:
        src_layer = int(src_layers[blk])
        for leaf, wkey in roles.items():
            tname = f"blk.{blk}.{leaf}.weight"
            sname = f"blk.{blk}.{leaf}.scale"
            dims, ttype, raw = r.tensor(tname)
            if ttype != TYPE_NVFP4:
                raise SystemExit(f"{tname} is {TYPE_NAMES.get(ttype, ttype)}, not nvfp4")
            if len(dims) != 3:
                raise SystemExit(f"{tname} has dims {dims}, expected [n_in, n_out, n_expert]")
            n_in, n_out, n_exp = dims
            sdims, stype, sraw = r.tensor(sname)
            if stype != TYPE_F32 or sdims != [n_exp]:
                raise SystemExit(f"{sname} is type {stype}, dims {sdims}; expected f32 [{n_exp}]")
            g_all = np.frombuffer(sraw.tobytes(), dtype=np.float32)
            row_bytes = (n_in // 64) * 36
            bytes_per_expert = row_bytes * n_out
            n_ex = n_exp if args.max_experts <= 0 else min(n_exp, args.max_experts)
            for e in range(n_ex):
                base = f"language_model.model.layers.{src_layer}.block_sparse_moe.experts.{e}.{wkey}"
                packed, pdt = st.get(base + ".weight_packed")
                scales, sdt = st.get(base + ".weight_scale")
                gs, gdt = st.get(base + ".weight_global_scale")
                if pdt != "U8" or sdt not in ("F8_E4M3", "F8_E4M3FN") or gdt != "F32":
                    raise SystemExit(f"{base}: unexpected dtypes {pdt}/{sdt}/{gdt}")
                g = float(gs.reshape(-1)[0])
                if np.float32(g) != g_all[e]:
                    raise SystemExit(f"{sname}[{e}] = {g_all[e]!r}, source G = {g!r}")
                mine = dequant_ggml_nvfp4(raw[e * bytes_per_expert:(e + 1) * bytes_per_expert],
                                          n_out, n_in, g_all[e])
                ref = dequant_spec_a(np.asarray(packed), np.asarray(scales), g_all[e])
                if mine.shape != ref.shape:
                    raise SystemExit(f"{base}: shape {mine.shape} vs spec {ref.shape}")
                d = float(np.max(np.abs(mine - ref)))
                worst = max(worst, d)
                total_elems += mine.size
                checked += 1
                if d != 0.0:
                    print(f"FAIL {base}: max |diff| = {d!r}")
                    return 1
            print(f"  {tname:28s} layer {src_layer:2d}  G[{g_all[0]:.6g} .. {g_all[-1]:.6g}]  "
                  f"{n_ex} experts  exact")
    print(f"\n{checked} expert matrices, {total_elems} elements: max |GGUF dequant - spec A dequant| = {worst!r}")
    if worst == 0.0:
        print("PASS: bit-for-bit identical in fp32")
        return 0
    print("FAIL: not bit-for-bit")
    return 1


if __name__ == "__main__":
    sys.exit(main())
