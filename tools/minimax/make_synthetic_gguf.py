"""Build a tiny synthetic `minimax-m3` GGUF.

The default file is the P2a loader fixture: two layers, every real tensor name, all zeros. It is not a model
to run. `--runnable` writes a 4-layer model (dense MLP on layers 0-2, one MoE layer, indexer block 8)
with glorot weights so the forward pass can be checked against tools/minimax/tiny_ref.py.

`--dense-type` stores the dense matrices (not norms or routed experts) as f16, bf16, q8_0, q6_k, or q4_k.
A row whose length is not a multiple of the block stays f32. `--quant-experts q8_0` or `iq2_xs` quantizes
the routed `*_exps.weight` tensors. IQ2_XS uses ggml's quantizer with a uniform importance row of ones
(strata-minimax --minimax-pack-iq2xs); the reference then dequantizes those stored values.

    python tools/minimax/make_synthetic_gguf.py out.gguf [--drop NAME] [--bad-shape]
    python tools/minimax/make_synthetic_gguf.py runnable.gguf --runnable [--dense-type q4_k] [--quant-experts iq2_xs]

Requires gguf-py from the local llama.cpp copy (see llama_baseline()).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE_GGUF_CANDIDATES = [
    Path(__file__).resolve().parents[2] / '_baseline' / 'llama.cpp' / 'gguf-py',
    Path(__file__).resolve().parents[2].parent / 'Strata-MiniMax' / '_baseline' / 'llama.cpp' / 'gguf-py',
]


def llama_baseline(gguf_py=None):
    candidates = ([Path(gguf_py)] if gguf_py else []) + BASELINE_GGUF_CANDIDATES
    for candidate in candidates:
        if (candidate / 'gguf').is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            import gguf
            return gguf
    raise SystemExit('gguf-py not found (tried ' + ', '.join(str(p) for p in candidates) +
                     '); set up _baseline/llama.cpp or pass --gguf-py')


GEOMETRY = dict(layers=2, hidden=256, vocab=320, heads=8, heads_kv=2, head_dim=32, rot=16, dense_ff=512,
                exp_ff=192, experts=8, used=4, shared=1, leading_dense=1, idx_heads=2, idx_dim=32, idx_top_k=2,
                idx_block=8, idx_local=1, idx_first=0, ctx=4096)

# Small enough to run token-by-token, wide enough that 200 tokens span more indexer blocks than top-k.
# leading_dense=3 puts swigluoai dense MLP on layers 0-2 and a routed+shared expert layer on layer 3.
RUNNABLE = dict(layers=4, hidden=256, vocab=128, heads=8, heads_kv=2, head_dim=32, rot=16, dense_ff=64,
                exp_ff=64, experts=8, used=4, shared=1, leading_dense=3, idx_heads=2, idx_dim=32, idx_top_k=16,
                idx_block=8, idx_local=1, idx_first=0, ctx=512)

WEIGHT_SEED = 0xC0FFEE


def _tensor_list(g):
    n, ne, vocab = g['layers'], g['hidden'], g['vocab']
    hq, hkv, d = g['heads'] * g['head_dim'], g['heads_kv'] * g['head_dim'], g['head_dim']
    ff, ffe = g['dense_ff'], g['exp_ff']
    nxp, shexp = g['experts'], g['exp_ff'] * g['shared']
    iq = g['idx_heads'] * g['idx_dim']

    # (gguf name, GGUF shape (ne0 first), numpy dtype).  The writer stores the numpy shape reversed, so each
    # numpy shape here is the reverse of the expected GGUF shape.
    yield 'token_embd.weight', (vocab, ne), np.float16
    yield 'output_norm.weight', (ne,), np.float32
    yield 'output.weight', (vocab, ne), np.float16
    for i in range(n):
        b = f'blk.{i}.'
        yield b + 'attn_norm.weight', (ne,), np.float32
        yield b + 'attn_q.weight', (hq, ne), np.float16
        yield b + 'attn_k.weight', (hkv, ne), np.float16
        yield b + 'attn_v.weight', (hkv, ne), np.float16
        yield b + 'attn_output.weight', (ne, hq), np.float16
        yield b + 'attn_q_norm.weight', (d,), np.float32
        yield b + 'attn_k_norm.weight', (d,), np.float32
        yield b + 'ffn_norm.weight', (ne,), np.float32
        if i < g['leading_dense']:
            yield b + 'ffn_gate.weight', (ff, ne), np.float16
            yield b + 'ffn_up.weight', (ff, ne), np.float16
            yield b + 'ffn_down.weight', (ne, ff), np.float16
        else:
            yield b + 'ffn_gate_inp.weight', (nxp, ne), np.float32
            yield b + 'exp_probs_b.bias', (nxp,), np.float32
            yield b + 'ffn_gate_exps.weight', (nxp, ffe, ne), np.float16
            yield b + 'ffn_down_exps.weight', (nxp, ne, ffe), np.float16
            yield b + 'ffn_up_exps.weight', (nxp, ffe, ne), np.float16
            if shexp:
                yield b + 'ffn_gate_shexp.weight', (shexp, ne), np.float16
                yield b + 'ffn_down_shexp.weight', (ne, shexp), np.float16
                yield b + 'ffn_up_shexp.weight', (shexp, ne), np.float16
        if i >= g['idx_first']:
            yield b + 'indexer.q_proj.weight', (iq, ne), np.float32
            yield b + 'indexer.k_proj.weight', (g['idx_dim'], ne), np.float32
            yield b + 'indexer.q_norm.weight', (g['idx_dim'],), np.float32
            yield b + 'indexer.k_norm.weight', (g['idx_dim'],), np.float32


def _glorot(name, shape, rng):
    shape = tuple(int(x) for x in shape)
    if name.endswith('norm.weight'):
        # Already includes the +1 the llama.cpp converter bakes in (spec F.2). Metadata says not to add it again.
        return (1.0 + rng.normal(0.0, 0.02, size=shape)).astype(np.float32)
    if name.endswith('exp_probs_b.bias'):
        return rng.normal(0.0, 0.05, size=shape).astype(np.float32)
    fan = max(int(shape[-1]), 1)
    return rng.normal(0.0, 1.0 / np.sqrt(fan), size=shape).astype(np.float32)


# Names match GGMLQuantizationType. IQ2_XS is not implemented in gguf-py; the engine packs it.  The five
# formats ported from ik_llama.cpp have no ggml-py ids either: the engine packs them and the writer below
# stores them under the engine's own ids (160..164, see include/strata/kernels/iqk_types.hpp).
_QUANT = {
    'f16': 'F16',
    'bf16': 'BF16',
    'q8_0': 'Q8_0',
    'q6_k': 'Q6_K',
    'q4_k': 'Q4_K',
    'iq2_xs': 'IQ2_XS',
    'iq2_kt': 160,
    'iq3_k': 161,
    'iq4_ks': 162,
    'iq3_kt': 163,
    'iq2_kl': 164,
}
IKQ_TYPES = {160, 161, 162, 163, 164}


def _minimax_exe():
    env = os.environ.get('STRATA_MINIMAX_EXE')
    candidates = [env, str(ROOT / 'build' / 'strata-minimax.exe'), str(ROOT / 'build' / 'strata-minimax')]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return None


def _is_dense_matrix(name):
    if name.endswith('norm.weight') or name.endswith('.bias'):
        return False
    if name.endswith('_exps.weight'):
        return False
    return name.endswith('.weight')


def _pack_ggml(arr, kind, gguf_mod, exe):
    # Q4_K and Q6_K have no quantize_blocks in this gguf-py. IQ2_XS has none at all. ggml writes both.
    if exe is None:
        raise SystemExit(f'{kind} needs strata-minimax (tools/minimax/build_windows.ps1) for ggml quantize')
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
    rows, cols = flat.shape
    raw_id = _QUANT[kind]
    ikq = isinstance(raw_id, int)
    qtype = raw_id if ikq else getattr(gguf_mod.GGMLQuantizationType, raw_id)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'w.f32'
        dst = Path(td) / 'w.bin'
        src.write_bytes(flat.tobytes())
        proc = subprocess.run(
            [exe, '--minimax-pack-quant', kind, str(rows), str(cols), str(src), str(dst)],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            raise SystemExit(proc.stderr.strip() or f'{kind} pack failed')
        raw = np.fromfile(dst, dtype=np.uint8)
    if ikq:
        # No gguf-py block table for these: one row per ne1*ne2, each of the engine's own row size.
        return np.ascontiguousarray(raw.reshape(-1)), qtype
    shape = gguf_mod.quants.quant_shape_to_byte_shape(arr.shape, qtype)
    if raw.size != int(np.prod(shape)):
        raise SystemExit(f'{kind} pack wrote {raw.size} bytes, expected {int(np.prod(shape))}')
    return np.ascontiguousarray(raw.reshape(shape)), qtype


def _quantize(arr, kind, gguf_mod, exe):
    raw_id = _QUANT[kind]
    ikq = isinstance(raw_id, int)
    if ikq:
        block = 256
    else:
        qtype = getattr(gguf_mod.GGMLQuantizationType, raw_id)
        block = gguf_mod.GGML_QUANT_SIZES[qtype][0]
    if int(arr.shape[-1]) % int(block) != 0:
        return None, None
    if ikq or kind in ('q4_k', 'q6_k', 'iq2_xs'):
        return _pack_ggml(arr, kind, gguf_mod, exe)
    raw = gguf_mod.quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), qtype)
    raw = np.ascontiguousarray(raw)
    if kind == 'f16':
        return raw, None
    return raw, qtype


def _add_meta(w, g, weights, gguf):
    w.add_block_count(g['layers'])
    w.add_context_length(g['ctx'])
    w.add_embedding_length(g['hidden'])
    w.add_vocab_size(g['vocab'])
    w.add_feed_forward_length(g['dense_ff'])
    w.add_head_count(g['heads'])
    w.add_head_count_kv(g['heads_kv'])
    w.add_key_length(g['head_dim'])
    w.add_value_length(g['head_dim'])
    w.add_rope_dimension_count(g['rot'])
    w.add_rope_freq_base(1.0e7)
    w.add_layer_norm_rms_eps(1.0e-6)
    w.add_expert_count(g['experts'])
    w.add_expert_used_count(g['used'])
    w.add_expert_feed_forward_length(g['exp_ff'])
    w.add_expert_shared_count(g['shared'])
    w.add_expert_weights_scale(2.0)
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_leading_dense_block_count(g['leading_dense'])
    w.add_indexer_head_count(g['idx_heads'])
    w.add_indexer_key_length(g['idx_dim'])
    w.add_indexer_top_k(g['idx_top_k'])
    w.add_indexer_block_size(g['idx_block'])
    w.add_indexer_local_blocks(g['idx_local'])
    # Upstream's gguf-py (what CI checks out) has the indexer keys above but not this writer method yet; the key it
    # writes is the one the engine reads (src/minimax/config.cpp), so write it directly when the method is absent.
    if hasattr(w, "add_indexer_first_layer"):
        w.add_indexer_first_layer(g['idx_first'])
    else:
        w.add_uint32(f"{w.arch}.attention.indexer.first_layer", g['idx_first'])
    if weights == 'glorot':
        # Authoritative: the norm tensors below already contain +1. The engine must not add it again.
        w.add_bool('minimax-m3.norm.add1', False)


def _unpack_ikq(raw, kind, shape, exe):
    """The stored bytes -> f32 with the engine's dequantizer (gguf-py has no codec for these types)."""
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    cols = int(shape[-1])
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'w.bin'
        dst = Path(td) / 'w.f32'
        src.write_bytes(np.ascontiguousarray(raw, dtype=np.uint8).tobytes())
        proc = subprocess.run(
            [exe, '--minimax-pack-dequant', kind, str(rows), str(cols), str(src), str(dst)],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            raise SystemExit(proc.stderr.strip() or f'{kind} dequant failed')
        out = np.fromfile(dst, dtype=np.float32)
    return out.reshape(shape)


def build(path, *, drop=None, bad_shape=False, geometry=None, gguf_py=None, weights=None, quant_experts=None,
          dense_type=None, pack_exe=None, deq_copy=None):
    g = {**GEOMETRY, **(geometry or {})}
    gguf = llama_baseline(gguf_py)
    w = gguf.GGUFWriter(str(path), 'minimax-m3')
    _add_meta(w, g, weights, gguf)
    rng = np.random.default_rng(WEIGHT_SEED)

    if dense_type not in (None, 'f32', 'f16', 'bf16', 'q8_0', 'q6_k', 'q4_k'):
        raise SystemExit('dense_type must be f16, bf16, q8_0, q6_k, or q4_k')
    ikq_kinds = ('iq2_kt', 'iq3_k', 'iq4_ks', 'iq3_kt', 'iq2_kl')
    if quant_experts not in ((None, 'q8_0', 'iq2_xs') + ikq_kinds):
        raise SystemExit('quant_experts must be q8_0, iq2_xs, ' + ', '.join(ikq_kinds))
    exe = pack_exe or _minimax_exe()
    deq_tensors = {}

    for name, shape, dtype in _tensor_list(g):
        if name == drop:
            continue
        if bad_shape and name == 'blk.1.ffn_gate_exps.weight':
            shape = (shape[0], shape[1], shape[2] + 1)
        if weights == 'glorot':
            arr = _glorot(name, shape, rng)
        else:
            arr = np.zeros(shape, dtype=dtype)
        kind = None
        if quant_experts and name.endswith('_exps.weight'):
            kind = quant_experts
        elif dense_type and dense_type != 'f32' and _is_dense_matrix(name):
            kind = dense_type
        raw, raw_dtype = (None, None)
        deq = None
        if kind:
            raw, raw_dtype = _quantize(np.ascontiguousarray(arr, dtype=np.float32), kind, gguf, exe)
        if raw is None:
            w.add_tensor(name, arr)
        elif raw_dtype is None:
            w.add_tensor(name, raw)
        elif isinstance(raw_dtype, int) and raw_dtype in IKQ_TYPES:
            # Bytes only: gguf-py has no block geometry for these, so the shape is the logical one in numpy
            # order (the writer reverses it, as it does for every tensor) and the payload is the engine's row
            # layout.
            w.add_tensor(name, raw.view(np.int8), raw_shape=list(arr.shape), raw_dtype=raw_dtype)
            if deq_copy:
                deq = _unpack_ikq(raw, kind, arr.shape, exe)
        else:
            w.add_tensor(name, raw, raw_dtype=raw_dtype)
        if deq is not None:
            deq_tensors[name] = deq
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    if deq_copy and deq_tensors:
        # The reference model: the same weights, with the ported expert formats expanded to f32.  A Python
        # forward pass can read this one (gguf-py has no codec for the ported ids).
        w2 = gguf.GGUFWriter(str(deq_copy), 'minimax-m3')
        _add_meta(w2, g, weights, gguf)
        rng2 = np.random.default_rng(WEIGHT_SEED)
        for name, shape, dtype in _tensor_list(g):
            if name == drop:
                continue
            if bad_shape and name == 'blk.1.ffn_gate_exps.weight':
                shape = (shape[0], shape[1], shape[2] + 1)
            # Draw for every tensor, in the same order as the main loop, so the two files hold the same
            # non-expert weights (`_glorot` pulls from the shared RNG).
            arr = _glorot(name, shape, rng2) if weights == 'glorot' else np.zeros(shape, dtype=dtype)
            if name in deq_tensors:
                w2.add_tensor(name, np.ascontiguousarray(deq_tensors[name], dtype=np.float32))
                continue
            kind = None
            if quant_experts and name.endswith('_exps.weight'):
                kind = quant_experts
            elif dense_type and dense_type != 'f32' and _is_dense_matrix(name):
                kind = dense_type
            if not kind:
                w2.add_tensor(name, arr)
                continue
            raw, raw_dtype = _quantize(np.ascontiguousarray(arr, dtype=np.float32), kind, gguf, exe)
            if raw is None:
                w2.add_tensor(name, arr)
            elif isinstance(raw_dtype, int) and raw_dtype in IKQ_TYPES:
                w2.add_tensor(name, _unpack_ikq(raw, kind, arr.shape, exe))
            else:
                w2.add_tensor(name, raw, raw_dtype=raw_dtype)
        w2.write_header_to_file()
        w2.write_kv_data_to_file()
        w2.write_tensors_to_file()
        w2.close()
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--drop', help='leave this tensor out (a malformed-file test)')
    ap.add_argument('--bad-shape', action='store_true', help='write blk.1.ffn_gate_exps.weight one element too wide')
    ap.add_argument('--runnable', action='store_true', help='4-layer model with glorot weights (RUNNABLE geometry)')
    ap.add_argument('--dense-type', choices=('f32', 'f16', 'bf16', 'q8_0', 'q6_k', 'q4_k'),
                    help='store dense matrices as this type (norms and routed experts stay put)')
    ap.add_argument('--quant-experts', choices=('q8_0', 'iq2_xs', 'iq2_kt', 'iq3_k', 'iq4_ks', 'iq3_kt', 'iq2_kl'),
                    help='quantize routed *_exps.weight (iq2_xs and the ik_llama.cpp formats use a uniform imatrix)')
    ap.add_argument('--gguf-py', type=Path, help='path to llama.cpp/gguf-py')
    a = ap.parse_args(argv)
    build(a.out, drop=a.drop, bad_shape=a.bad_shape, gguf_py=a.gguf_py,
          geometry=RUNNABLE if a.runnable else None,
          weights='glorot' if a.runnable else None,
          quant_experts=a.quant_experts,
          dense_type=a.dense_type)
    print(a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
