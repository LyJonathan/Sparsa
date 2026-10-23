"""Build a tiny synthetic `minimax-m3` GGUF.

The default file is the P2a loader fixture: two layers, every real tensor name, all zeros. It is not a model
to run. `--runnable` writes a 4-layer F32 model (dense MLP on layers 0-2, one MoE layer, indexer block 8)
with glorot weights so the forward pass can be checked against tools/minimax/tiny_ref.py. `--quant-experts q8_0`
quantizes the routed `*_exps.weight` tensors; the reference then dequantizes those stored values.

    python tools/minimax/make_synthetic_gguf.py out.gguf [--drop NAME] [--bad-shape]
    python tools/minimax/make_synthetic_gguf.py runnable.gguf --runnable [--quant-experts q8_0]

Requires gguf-py from the local llama.cpp copy (see llama_baseline()).
"""
from __future__ import annotations

import argparse
import sys
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


def build(path, *, drop=None, bad_shape=False, geometry=None, gguf_py=None, weights=None, quant_experts=None):
    g = {**GEOMETRY, **(geometry or {})}
    gguf = llama_baseline(gguf_py)
    w = gguf.GGUFWriter(str(path), 'minimax-m3')
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
    w.add_indexer_first_layer(g['idx_first'])
    rng = np.random.default_rng(WEIGHT_SEED)
    if weights == 'glorot':
        # Authoritative: the norm tensors below already contain +1. The engine must not add it again.
        w.add_bool('minimax-m3.norm.add1', False)

    for name, shape, dtype in _tensor_list(g):
        if name == drop:
            continue
        if bad_shape and name == 'blk.1.ffn_gate_exps.weight':
            shape = (shape[0], shape[1], shape[2] + 1)
        if weights == 'glorot':
            arr = _glorot(name, shape, rng)
        else:
            arr = np.zeros(shape, dtype=dtype)
        if quant_experts == 'q8_0' and name.endswith('_exps.weight'):
            raw = gguf.quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--drop', help='leave this tensor out (a malformed-file test)')
    ap.add_argument('--bad-shape', action='store_true', help='write blk.1.ffn_gate_exps.weight one element too wide')
    ap.add_argument('--runnable', action='store_true', help='4-layer F32 model with glorot weights (RUNNABLE geometry)')
    ap.add_argument('--quant-experts', choices=('q8_0',), help='quantize routed *_exps.weight (with --runnable)')
    ap.add_argument('--gguf-py', type=Path, help='path to llama.cpp/gguf-py')
    a = ap.parse_args(argv)
    build(a.out, drop=a.drop, bad_shape=a.bad_shape, gguf_py=a.gguf_py,
          geometry=RUNNABLE if a.runnable else None,
          weights='glorot' if a.runnable else None,
          quant_experts=a.quant_experts)
    print(a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
