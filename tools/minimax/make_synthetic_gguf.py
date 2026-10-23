"""Build a tiny synthetic `minimax-m3` GGUF for the P2a loader/inspect test.

The shapes and names are the real ones from spec F.2, with every dimension reduced; the model is deliberately NOT
MiniMax-M3.1-Flash's fingerprint (2 layers, 256-wide), which exercises the inspect-only path.  There is no such
model: this is a directory test, not weights to run.

    python tools/minimax/make_synthetic_gguf.py out.gguf [--drop NAME] [--bad-shape]

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


def build(path, *, drop=None, bad_shape=False, geometry=None, gguf_py=None):
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

    for name, shape, dtype in _tensor_list(g):
        if name == drop:
            continue
        if bad_shape and name == 'blk.1.ffn_gate_exps.weight':
            shape = (shape[0], shape[1], shape[2] + 1)
        w.add_tensor(name, np.zeros(shape, dtype=dtype))
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
    ap.add_argument('--gguf-py', type=Path, help='path to llama.cpp/gguf-py')
    a = ap.parse_args(argv)
    build(a.out, drop=a.drop, bad_shape=a.bad_shape, gguf_py=a.gguf_py)
    print(a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
