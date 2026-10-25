"""tools/minimax/dspark_pack.py - pack MiniMax-M3.1-Flash's DSpark draft head into the GGUF the engine reads.

The checkpoint dir (models/MiniMax-M3.1-Flash/dspark) holds one safetensors file whose weights are MXFP8
(F8_E4M3 codes with one E8M0 scale per 32 values along the input dim) except the small pieces (fc, the markov and
confidence heads, all the norms), which are BF16.  This script decodes them, drops the
`language_model.model.dspark.` prefix and writes a GGUF v3 with

    fc.weight                     [n_in][n_embd]      the taps of the target's layers -> the drafter's width
    hidden_norm.weight            [n_embd]            the norm after that projection
    layers.N.*                    as in the checkpoint (input_layernorm, self_attn.{q,k,v,o}_proj,
                                  self_attn.{q,k}_norm, post_attention_layernorm, mlp.{gate,up,down}_proj)
    final_norm.weight             [n_embd]            before the head
    markov_w1.weight              [vocab][rank]       the low-rank Markov head: bias[v] = <w1[v], w2[prev]>
    markov_w2.weight              [vocab][rank]
    confidence_head.{weight,bias} [1][n_embd+rank]    sigmoid confidence per draft position

and the geometry as metadata (`dspark.*`), so nothing about the drafter is compiled in.  `--type` picks the payload:
`q8_0` (default: the engine runs it through the device MMVQ path, 1.7 GiB) or `f16` (2x the bytes, plain GEMM);
the 1-D norms, the markov head and the confidence head always stay F16 (they are tiny and read by the host).

    python tools/minimax/dspark_pack.py --src models/MiniMax-M3.1-Flash/dspark \
        --out models/MiniMax-M3.1-Flash/dspark-q8_0.gguf [--type f16]

The GGUF is byte-layout compatible with the engine's header-only reader (src/artifact/gguf_reader.hpp), so
`--minimax-inspect-dspark <file>` prints the same table without a GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

# ---- ggml type ids we write (the reader's table)
GGUF_MAGIC = 0x46554747
T_F32, T_F16, T_Q4_0, T_Q5_0, T_Q8_0 = 0, 1, 2, 6, 8

# metadata value types
M_U32, M_I32, M_F32, M_BOOL, M_STRING, M_ARRAY, M_U64 = 4, 5, 6, 7, 8, 9, 10

PREFIX = "language_model.model.dspark."


def e4m3_table() -> np.ndarray:
    """The 256 F8_E4M3 codes as float32 (the same table tools/ds4/dspark_ref.py decodes with)."""
    t = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        v = np.nan if (e == 15 and m == 7) else (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()


class Checkpoint:
    def __init__(self, d: pathlib.Path):
        self.d = d
        files = sorted(d.glob("*.safetensors"))
        if not files:
            raise SystemExit(f"{d}: no *.safetensors")
        self.shards = []
        self.index = {}
        for f in files:
            raw = np.memmap(f, np.uint8, mode="r")
            n = struct.unpack("<Q", bytes(raw[:8]))[0]
            head = json.loads(bytes(raw[8:8 + n]))
            head.pop("__metadata__", None)
            base = 8 + n
            self.shards.append((f, raw, base))
            for k, e in head.items():
                self.index[k] = (len(self.shards) - 1, base, e)

    def keys(self):
        return self.index.keys()

    def raw(self, name):
        s, base, e = self.index[name]
        a, b = e["data_offsets"]
        raw = self.shards[s][1]
        return raw[base + a:base + b], e["dtype"], tuple(e["shape"])

    def get(self, name) -> np.ndarray:
        w, dt, sh = self.raw(name)
        if dt == "F32":
            v = np.frombuffer(w, np.float32).reshape(sh)
        elif dt == "BF16":
            v = (np.frombuffer(w, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(sh)
        elif dt == "F8_E4M3":
            sname = name[:-len("weight")] + "weight_scale_inv"
            if sname not in self.index:
                raise SystemExit(f"{name}: no {sname}")
            s, sdt, ssh = self.raw(sname)
            if sdt != "U8":
                raise SystemExit(f"{sname}: scale dtype {sdt}, expected U8 (E8M0)")
            rows, cols = sh
            sc = np.asarray(s, np.uint8).reshape(ssh)
            if cols % (sc.shape[1]) != 0 or rows != sc.shape[0]:
                raise SystemExit(f"{sname}: scales {ssh} do not tile {sh}")
            per = cols // sc.shape[1]
            table = np.exp2(sc.astype(np.float32) - 127.0)
            table = np.repeat(table, per, axis=1)[:, :cols]
            v = E4M3[np.asarray(w, np.uint8).reshape(rows, cols)] * table
        else:
            raise SystemExit(f"{name}: dtype {dt} is not handled")
        return np.ascontiguousarray(v, np.float32)


def q8_0(v: np.ndarray) -> bytes:
    """ggml Q8_0: fp16 scale + 32 int8 per 32 values, d = amax / 127."""
    rows, cols = v.shape
    if cols % 32:
        raise SystemExit(f"Q8_0 rows must be whole 32-value blocks (got {cols})")
    x = v.reshape(rows, cols // 32, 32)
    amax = np.abs(x).max(-1)
    d = (amax / 127.0).astype(np.float16)
    # a zero block keeps d = 0 and all-zero codes
    dd = np.where(d == 0, np.float16(1.0), d).astype(np.float32)
    q = np.clip(np.rint(x / dd[..., None]), -127, 127).astype(np.int8)
    q[amax == 0] = 0
    out = np.empty((rows, cols // 32, 34), np.uint8)
    out[..., :2] = d.view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.tobytes()


def _blocks(v: np.ndarray):
    rows, cols = v.shape
    if cols % 32:
        raise SystemExit(f"the 4/5-bit blocks hold 32 values (got a {cols}-wide row)")
    return v.reshape(rows, cols // 32, 32).astype(np.float32)


def _d_and_id(x: np.ndarray, scale: float):
    """ggml's rule: d = max / -scale (the signed element of largest magnitude), which is fp16 in the block."""
    mx = np.take_along_axis(x, np.abs(x).argmax(-1)[..., None], -1)[..., 0]
    d = (mx / -scale).astype(np.float16)
    dd = d.astype(np.float32)
    idv = np.where(dd != 0, 1.0 / np.where(dd != 0, dd, 1.0), 0.0)
    return d, idv


def q4_0(v: np.ndarray) -> bytes:
    """ggml Q4_0: value[j] = (lo nibble of qs[j] - 8) * d, value[j + 16] = (hi nibble - 8) * d (halves, not pairs)."""
    x = _blocks(v)
    d, idv = _d_and_id(x, 8.0)
    xi = np.clip(np.floor(x * idv[..., None] + 8.5), 0, 15).astype(np.uint8)
    out = np.empty(x.shape[:2] + (18,), np.uint8)
    out[..., :2] = d.view(np.uint8).reshape(x.shape[0], x.shape[1], 2)
    out[..., 2:] = xi[..., :16] | (xi[..., 16:] << 4)
    return out.tobytes()


def q5_0(v: np.ndarray) -> bytes:
    """ggml Q5_0: value[j] = (lo nibble | qh bit j << 4) - 16, value[j + 16] = (hi nibble | qh bit j+16) - 16, all * d."""
    x = _blocks(v)
    d, idv = _d_and_id(x, 16.0)
    xi = np.clip(np.floor(x * idv[..., None] + 16.5), 0, 31).astype(np.uint8)
    qh = np.zeros(x.shape[:2], np.uint32)
    for j in range(16):
        qh |= ((xi[..., j] & 0x10) >> 4).astype(np.uint32) << np.uint32(j)
        qh |= ((xi[..., 16 + j] & 0x10) >> 4).astype(np.uint32) << np.uint32(j + 16)
    out = np.empty(x.shape[:2] + (22,), np.uint8)
    out[..., :2] = d.view(np.uint8).reshape(x.shape[0], x.shape[1], 2)
    out[..., 2:6] = qh.view(np.uint8).reshape(x.shape[0], x.shape[1], 4)
    out[..., 6:] = (xi[..., :16] & 15) | ((xi[..., 16:] & 15) << 4)
    return out.tobytes()


def f16(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, np.float16).tobytes()


class Writer:
    def __init__(self):
        self.kv = []
        self.t = []

    def meta(self, key, typ, value):
        self.kv.append((key, typ, value))

    def tensor(self, name, data: bytes, shape, typ):
        # GGUF dims are ne0 first: the contiguous (input) dim is shape[-1] for a row-major 2-D array
        dims = [int(d) for d in reversed(shape)]
        self.t.append((name, dims, typ, data))

    def write(self, path, align=32):
        with open(path, "wb") as f:
            f.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(self.t), len(self.kv)))
            for key, typ, value in self.kv:
                f.write(_str(key))
                f.write(struct.pack("<I", typ))
                f.write(_value(typ, value))
            off = 0
            offsets = []
            for name, dims, typ, data in self.t:
                f.write(_str(name))
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", typ))
                f.write(struct.pack("<Q", off))
                pad = (-len(data)) % align
                offsets.append(off)
                off += len(data) + pad
            pos = f.tell()
            f.write(b"\0" * ((-pos) % align))
            for (name, dims, typ, data), o in zip(self.t, offsets):
                f.write(data)
                f.write(b"\0" * ((-len(data)) % align))
        return path


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _value(typ, v):
    if typ == M_STRING:
        return _str(v)
    if typ == M_U32:
        return struct.pack("<I", v)
    if typ == M_I32:
        return struct.pack("<i", v)
    if typ == M_F32:
        return struct.pack("<f", v)
    if typ == M_BOOL:
        return struct.pack("<B", 1 if v else 0)
    if typ == M_U64:
        return struct.pack("<Q", v)
    if typ == M_ARRAY:
        et, items = v
        out = struct.pack("<IQ", et, len(items))
        for it in items:
            out += _value(et, it)
        return out
    raise SystemExit(f"metadata type {typ} not handled")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the checkpoint's dspark/ directory")
    ap.add_argument("--out", required=True, help="the .gguf to write")
    ap.add_argument("--type", default="q8_0", choices=["q8_0", "q5_0", "q4_0", "f16"],
                    help="payload type of the 2-D matrices (q4_0: ~4 bits, the VRAM the drafter may not take "
                         "from the expert slots)")
    ap.add_argument("--tap-layers", default="",
                    help="override dspark.tap_layers (a truncated model runs the drafter on the layers it has; the "
                         "drafts are then only a test vehicle, not a speed measurement)")
    ap.add_argument("--fc-type", default="", choices=["", "q8_0", "q5_0", "q4_0", "f16"],
                    help="override --type for fc.weight (30720 wide, the widest read)")
    a = ap.parse_args()
    src = pathlib.Path(a.src)
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    ck = Checkpoint(src)

    n_embd = int(tc["hidden_size"])
    n_layers = int(tc["num_hidden_layers"])
    n_head = int(tc["num_attention_heads"])
    n_kv = int(tc["num_key_value_heads"])
    head_dim = int(tc["head_dim"])
    n_rot = int(tc.get("rotary_dim", head_dim))
    n_ff = int(tc.get("dense_intermediate_size", tc.get("dspark_ffn_hidden_size", tc["intermediate_size"])))
    taps = [int(x) for x in tc["dspark_target_layer_ids"]]
    if a.tap_layers:
        taps = [int(x) for x in a.tap_layers.split(',') if x.strip() != '']
    block = int(tc["dspark_block_size"])
    rank = int(tc["dspark_markov_rank"])
    mask_id = int(tc["dspark_noise_token_id"])
    vocab = int(tc["vocab_size"])
    alpha = float(tc.get("swiglu_alpha", 1.702))
    limit = float(tc.get("swiglu_limit", 7.0))
    eps = float(tc.get("rms_norm_eps", 1e-6))
    theta = float(tc.get("rope_theta", 1e7))
    window = int(tc.get("sliding_window", 4096))
    w = Writer()
    w.meta("general.architecture", M_STRING, "minimax-dspark")
    w.meta("general.name", M_STRING, "MiniMax-M3.1-Flash DSpark draft head")
    w.meta("dspark.block_count", M_U32, n_layers)
    w.meta("dspark.embedding_length", M_U32, n_embd)
    w.meta("dspark.attention.head_count", M_U32, n_head)
    w.meta("dspark.attention.head_count_kv", M_U32, n_kv)
    w.meta("dspark.attention.key_length", M_U32, head_dim)
    w.meta("dspark.rope.dimension_count", M_U32, n_rot)
    w.meta("dspark.rope.freq_base", M_F32, theta)
    w.meta("dspark.attention.layer_norm_rms_epsilon", M_F32, eps)
    w.meta("dspark.feed_forward_length", M_U32, n_ff)
    w.meta("dspark.block_size", M_U32, block)
    w.meta("dspark.markov_rank", M_U32, rank)
    w.meta("dspark.mask_token_id", M_U32, mask_id)
    w.meta("dspark.vocab_size", M_U32, vocab)
    w.meta("dspark.attention.sliding_window", M_U32, window)
    w.meta("dspark.swiglu_alpha", M_F32, alpha)
    w.meta("dspark.swiglu_limit", M_F32, limit)
    w.meta("dspark.norm_add1", M_BOOL, bool(tc.get("use_gemma_norm", False)))
    w.meta("dspark.tap_layers", M_ARRAY, (M_I32, taps))

    n_in_fc = len(taps) * n_embd
    mat_type = a.type
    fc_type = a.fc_type or mat_type

    def put_matrix(name, ckname, typ, want_rows=None, want_cols=None):
        v = ck.get(ckname)
        if want_cols is not None and v.shape[-1] != want_cols:
            raise SystemExit(f"{ckname}: shape {v.shape}, expected [.., {want_cols}]")
        if want_rows is not None and v.shape[0] != want_rows:
            raise SystemExit(f"{ckname}: shape {v.shape}, expected [{want_rows}, ..]")
        if v.ndim == 2 and v.shape[0] == 1:
            v = v.reshape(-1)
        if v.ndim == 1:
            w.tensor(name, f16(v), v.shape, T_F16)
            return
        if typ == "f16":
            w.tensor(name, f16(v), v.shape, T_F16)
        elif typ == "q4_0":
            w.tensor(name, q4_0(v), v.shape, T_Q4_0)
        elif typ == "q5_0":
            w.tensor(name, q5_0(v), v.shape, T_Q5_0)
        else:
            w.tensor(name, q8_0(v), v.shape, T_Q8_0)

    put_matrix("fc.weight", PREFIX + "fc.weight", fc_type, want_cols=n_in_fc, want_rows=n_embd)
    put_matrix("hidden_norm.weight", PREFIX + "hidden_norm.weight", "f16", want_rows=n_embd)
    put_matrix("final_norm.weight", PREFIX + "final_norm.weight", "f16", want_rows=n_embd)
    for l in range(n_layers):
        p = PREFIX + f"layers.{l}.decoder_layer."
        q = f"layers.{l}."
        put_matrix(q + "input_layernorm.weight", p + "input_layernorm.weight", "f16", want_rows=n_embd)
        put_matrix(q + "self_attn.q_proj.weight", p + "self_attn.q_proj.weight", mat_type,
                   want_rows=n_head * head_dim, want_cols=n_embd)
        put_matrix(q + "self_attn.k_proj.weight", p + "self_attn.k_proj.weight", mat_type,
                   want_rows=n_kv * head_dim, want_cols=n_embd)
        put_matrix(q + "self_attn.v_proj.weight", p + "self_attn.v_proj.weight", mat_type,
                   want_rows=n_kv * head_dim, want_cols=n_embd)
        put_matrix(q + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight", mat_type,
                   want_rows=n_embd, want_cols=n_head * head_dim)
        put_matrix(q + "self_attn.q_norm.weight", p + "self_attn.q_norm.weight", "f16", want_rows=head_dim)
        put_matrix(q + "self_attn.k_norm.weight", p + "self_attn.k_norm.weight", "f16", want_rows=head_dim)
        put_matrix(q + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight", "f16",
                   want_rows=n_embd)
        put_matrix(q + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight", mat_type, want_rows=n_ff,
                   want_cols=n_embd)
        put_matrix(q + "mlp.up_proj.weight", p + "mlp.up_proj.weight", mat_type, want_rows=n_ff, want_cols=n_embd)
        put_matrix(q + "mlp.down_proj.weight", p + "mlp.down_proj.weight", mat_type, want_rows=n_embd,
                   want_cols=n_ff)
    put_matrix("markov_w1.weight", PREFIX + "markov_head.markov_w1.weight", "f16", want_rows=vocab,
               want_cols=rank)
    put_matrix("markov_w2.weight", PREFIX + "markov_head.markov_w2.weight", "f16", want_rows=vocab,
               want_cols=rank)
    put_matrix("confidence_head.weight", PREFIX + "confidence_head.proj.weight", "f16", want_rows=1,
               want_cols=n_embd + rank)
    put_matrix("confidence_head.bias", PREFIX + "confidence_head.proj.bias", "f16", want_rows=1)

    out = w.write(a.out)
    size = pathlib.Path(out).stat().st_size
    print(f"{out}: {len(w.t)} tensors, {size / 2**30:.2f} GiB, matrices {mat_type}, fc {fc_type}, "
          f"{n_layers} layers, taps {taps}, block {block}, rank {rank}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
