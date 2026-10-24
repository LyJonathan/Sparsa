"""Independent MiniMax forward, written from the reference spec (sections C, D, F).

Reads a GGUF (the synthetic runnable file, or a Q8_0-expert variant of it) and runs one token at a time in
fp32. Matrix-vector products accumulate in float64 left to right, then cast back to float32, which is what
the engine's CPU path does. KV and indexer-K round-trip through float16 before the dots. Norm weights are
used as stored unless `minimax-m3.norm.add1` is true.

This is not the engine and it does not import the engine's weight formulas. It only reads the file.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from make_synthetic_gguf import llama_baseline


def _field(reader, key):
    f = reader.get_field(key)
    if f is None:
        return None
    return f.contents()


def _need(reader, suffix):
    v = _field(reader, 'minimax-m3.' + suffix)
    if v is None:
        raise SystemExit('missing minimax-m3.' + suffix)
    if isinstance(v, list):
        if not v:
            raise SystemExit('empty minimax-m3.' + suffix)
        return v[0]
    return v


def _opt(reader, suffix, default):
    v = _field(reader, 'minimax-m3.' + suffix)
    if v is None:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _f32(reader, name, tensors, override=None):
    if override is not None and name in override:
        # Dequantized outside: the formats ported from ik_llama.cpp have no gguf-py codec, so the test
        # dequantizes the stored bytes with the engine's own kernel and hands them in.
        return np.ascontiguousarray(override[name], dtype=np.float32)
    t = tensors[name]
    gguf = reader  # unused; type dispatch is on the tensor
    del gguf
    kind = t.tensor_type.name
    if kind == 'F32':
        return np.ascontiguousarray(t.data, dtype=np.float32)
    if kind == 'F16':
        return np.ascontiguousarray(t.data.astype(np.float32))
    import gguf as gguf_mod
    return np.ascontiguousarray(gguf_mod.quants.dequantize(t.data, t.tensor_type), dtype=np.float32)


def load(path, gguf_py=None, deq=None):
    gguf = llama_baseline(gguf_py)
    reader = gguf.GGUFReader(str(path))
    tensors = {t.name: t for t in reader.tensors}
    n_layers = int(_need(reader, 'block_count'))
    hidden = int(_need(reader, 'embedding_length'))
    heads = int(_need(reader, 'attention.head_count'))
    heads_kv = int(_need(reader, 'attention.head_count_kv'))
    head_dim = int(_need(reader, 'attention.key_length'))
    rot = int(_opt(reader, 'rope.dimension_count', head_dim))
    theta = float(_opt(reader, 'rope.freq_base', 1.0e7))
    eps = float(_need(reader, 'attention.layer_norm_rms_epsilon'))
    dense_ff = int(_need(reader, 'feed_forward_length'))
    exp_ff = int(_need(reader, 'expert_feed_forward_length'))
    n_expert = int(_need(reader, 'expert_count'))
    used = int(_need(reader, 'expert_used_count'))
    shared = int(_opt(reader, 'expert_shared_count', 0))
    leading = int(_opt(reader, 'leading_dense_block_count', 0))
    scale = float(_opt(reader, 'expert_weights_scale', 1.0))
    renorm = bool(int(_opt(reader, 'expert_weights_norm', 0)))
    alpha = float(_opt(reader, 'swiglu.alpha', 1.702))
    limit = float(_opt(reader, 'swiglu.limit', 7.0))
    idx_heads = int(_need(reader, 'attention.indexer.head_count'))
    idx_dim = int(_need(reader, 'attention.indexer.key_length'))
    idx_top_k = int(_need(reader, 'attention.indexer.top_k'))
    idx_block = int(_need(reader, 'attention.indexer.block_size'))
    idx_local = int(_need(reader, 'attention.indexer.local_blocks'))
    idx_first = int(_opt(reader, 'attention.indexer.first_layer', leading))
    add1 = _field(reader, 'minimax-m3.norm.add1')
    norm_add1 = bool(int(add1)) if add1 is not None else False

    def norm(name):
        w = _f32(reader, name, tensors, deq)
        if norm_add1:
            w = w + np.float32(1)
        return w

    layers = []
    for i in range(n_layers):
        b = f'blk.{i}.'
        L = {
            'attn_norm': norm(b + 'attn_norm.weight'),
            'ffn_norm': norm(b + 'ffn_norm.weight'),
            'q_norm': norm(b + 'attn_q_norm.weight'),
            'k_norm': norm(b + 'attn_k_norm.weight'),
            'wq': _f32(reader, b + 'attn_q.weight', tensors, deq),
            'wk': _f32(reader, b + 'attn_k.weight', tensors, deq),
            'wv': _f32(reader, b + 'attn_v.weight', tensors, deq),
            'wo': _f32(reader, b + 'attn_output.weight', tensors, deq),
            'moe': i >= leading,
            'indexer': i >= idx_first,
        }
        if not L['moe']:
            L['gate'] = _f32(reader, b + 'ffn_gate.weight', tensors, deq)
            L['up'] = _f32(reader, b + 'ffn_up.weight', tensors, deq)
            L['down'] = _f32(reader, b + 'ffn_down.weight', tensors, deq)
        else:
            L['router'] = _f32(reader, b + 'ffn_gate_inp.weight', tensors, deq)
            L['bias'] = _f32(reader, b + 'exp_probs_b.bias', tensors, deq)
            L['egate'] = _f32(reader, b + 'ffn_gate_exps.weight', tensors, deq)
            L['eup'] = _f32(reader, b + 'ffn_up_exps.weight', tensors, deq)
            L['edown'] = _f32(reader, b + 'ffn_down_exps.weight', tensors, deq)
            if shared:
                L['sgate'] = _f32(reader, b + 'ffn_gate_shexp.weight', tensors, deq)
                L['sup'] = _f32(reader, b + 'ffn_up_shexp.weight', tensors, deq)
                L['sdown'] = _f32(reader, b + 'ffn_down_shexp.weight', tensors, deq)
        if L['indexer']:
            L['iq'] = _f32(reader, b + 'indexer.q_proj.weight', tensors, deq)
            L['ik'] = _f32(reader, b + 'indexer.k_proj.weight', tensors, deq)
            L['iq_norm'] = norm(b + 'indexer.q_norm.weight')
            L['ik_norm'] = norm(b + 'indexer.k_norm.weight')
        layers.append(L)

    return {
        'embd': _f32(reader, 'token_embd.weight', tensors, deq),
        'output': _f32(reader, 'output.weight', tensors, deq),
        'out_norm': norm('output_norm.weight'),
        'layers': layers,
        'hidden': hidden,
        'heads': heads,
        'heads_kv': heads_kv,
        'head_dim': head_dim,
        'rot': rot,
        'theta': theta,
        'eps': eps,
        'dense_ff': dense_ff,
        'exp_ff': exp_ff,
        'n_expert': n_expert,
        'used': used,
        'shared': shared,
        'scale': scale,
        'renorm': renorm,
        'alpha': alpha,
        'limit': limit,
        'idx_heads': idx_heads,
        'idx_dim': idx_dim,
        'idx_top_k': idx_top_k,
        'idx_block': idx_block,
        'idx_local': idx_local,
        'norm_add1': norm_add1,
    }


def gemv(W, x):
    """Left-to-right float64 dot of each row, cast back to float32."""
    W = np.ascontiguousarray(W, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    acc = np.zeros(W.shape[0], dtype=np.float64)
    for i in range(W.shape[1]):
        acc += W[:, i] * x[i]
    return acc.astype(np.float32)


def rms(x, w, eps):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    ss = np.float32(0)
    for v in x:
        ss = np.float32(ss + np.float32(v * v))
    inv = np.float32(1.0) / np.sqrt(np.float32(ss / np.float32(x.size) + np.float32(eps)))
    return (x * inv * w).astype(np.float32)


def sigmoid(x):
    return np.float32(1.0 / (1.0 + math.exp(-float(x))))


def swigluoai(gate, up, alpha, limit):
    g = np.minimum(np.asarray(gate, dtype=np.float32), np.float32(limit))
    u = np.clip(np.asarray(up, dtype=np.float32), -np.float32(limit), np.float32(limit))
    # sigmoid(alpha * gate) per element, double exp, matching the engine.
    s = np.empty_like(g)
    a = float(alpha)
    for i in range(g.size):
        s[i] = sigmoid(np.float32(a) * g[i])
    return ((u + np.float32(1)) * g * s).astype(np.float32)


def rope(x, n_rot, pos, theta):
    """x is (heads, dim). NeoX pairs (i, i + n_rot/2) on the first n_rot dims."""
    y = np.array(x, dtype=np.float32, copy=True)
    half = n_rot // 2
    for i in range(half):
        freq = float(pos) / (float(theta) ** ((2.0 * i) / float(n_rot)))
        c = np.float32(math.cos(freq))
        s = np.float32(math.sin(freq))
        a = y[:, i].copy()
        b = y[:, i + half].copy()
        y[:, i] = a * c - b * s
        y[:, i + half] = b * c + a * s
    return y


def softmax(scores):
    s = np.asarray(scores, dtype=np.float32)
    m = np.float32(-np.inf)
    for v in s:
        if v > m:
            m = v
    out = np.zeros_like(s)
    if not np.isfinite(m):
        return out
    for i, v in enumerate(s):
        if v == np.float32(-np.inf) or v == -np.inf:
            out[i] = np.float32(0)
        else:
            out[i] = np.exp(np.float32(v - m))
    total = np.float32(0)
    for v in out:
        total = np.float32(total + v)
    if total > 0:
        out *= np.float32(1) / total
    return out


def _f16(x):
    return np.ascontiguousarray(x, dtype=np.float16)


def _from_f16(h):
    return h.astype(np.float32)


def mlp(gate, up, down, x, alpha, limit):
    g = gemv(gate, x)
    u = gemv(up, x)
    h = swigluoai(g, u, alpha, limit)
    return gemv(down, h)


def forward(model, tokens):
    """Logits after each token, shape (len(tokens), vocab), float32. The row predicts the next id."""
    m = model
    tokens = [int(t) for t in tokens]
    n = len(tokens)
    H, HKV, D = m['heads'], m['heads_kv'], m['head_dim']
    q_per_kv = H // HKV
    q_per_idx = H // m['idx_heads']
    scale = np.float32(1.0 / math.sqrt(D))
    k_cache = [[] for _ in m['layers']]
    v_cache = [[] for _ in m['layers']]
    ik_cache = [[] for _ in m['layers']]
    logits = np.zeros((n, m['output'].shape[0]), dtype=np.float32)
    x = None
    for pos, tok in enumerate(tokens):
        x = m['embd'][tok].astype(np.float32).copy()
        for li, L in enumerate(m['layers']):
            h = rms(x, L['attn_norm'], m['eps'])
            q = gemv(L['wq'], h).reshape(H, D)
            k = gemv(L['wk'], h).reshape(HKV, D)
            v = gemv(L['wv'], h).reshape(HKV, D)
            for head in range(H):
                q[head] = rms(q[head], L['q_norm'], m['eps'])
            for head in range(HKV):
                k[head] = rms(k[head], L['k_norm'], m['eps'])
            q = rope(q, m['rot'], pos, m['theta'])
            k = rope(k, m['rot'], pos, m['theta'])
            k_cache[li].append(_f16(k))
            v_cache[li].append(_f16(v))
            sel = None
            if L['indexer']:
                iq = gemv(L['iq'], h).reshape(m['idx_heads'], m['idx_dim'])
                ik = gemv(L['ik'], h)
                for head in range(m['idx_heads']):
                    iq[head] = rms(iq[head], L['iq_norm'], m['eps'])
                ik = rms(ik, L['ik_norm'], m['eps'])
                iq = rope(iq, m['rot'], pos, m['theta'])
                ik = rope(ik.reshape(1, -1), m['rot'], pos, m['theta'])[0]
                ik_cache[li].append(_f16(ik))
                Tk = pos + 1
                BLK = m['idx_block']
                nb = (Tk + BLK - 1) // BLK
                bs = np.full((m['idx_heads'], nb), -np.inf, dtype=np.float32)
                cached = np.stack([_from_f16(t) for t in ik_cache[li]], axis=0)
                for head in range(m['idx_heads']):
                    dots = _row_dots(cached, iq[head])
                    for b in range(nb):
                        begin = b * BLK
                        end = min(begin + BLK, Tk)
                        bs[head, b] = np.max(dots[begin:end])
                    qb = pos // BLK
                    for local in range(m['idx_local']):
                        bi = qb - local
                        if bi < 0:
                            bi = 0
                        if bi < nb:
                            bs[head, bi] = np.float32(np.inf)
                kk = min(m['idx_top_k'], nb)
                sel = np.zeros((m['idx_heads'], nb), dtype=np.int8)
                for head in range(m['idx_heads']):
                    order = sorted(range(nb), key=lambda b, head=head: (-float(bs[head, b]), b))
                    for b in order[:kk]:
                        if bs[head, b] == -np.inf:
                            continue
                        sel[head, b] = 1
            Tk = pos + 1
            Ks = np.stack([_from_f16(t) for t in k_cache[li]], axis=0)   # (T, HKV, D)
            Vs = np.stack([_from_f16(t) for t in v_cache[li]], axis=0)
            att = np.zeros((H, D), dtype=np.float32)
            for qh in range(H):
                kv = qh // q_per_kv
                dots = _row_dots(Ks[:, kv, :], q[qh]) * scale
                prob = np.empty(Tk, dtype=np.float32)
                for t in range(Tk):
                    keep = True
                    if sel is not None:
                        keep = sel[qh // q_per_idx, t // m['idx_block']] != 0
                    prob[t] = dots[t] if keep else np.float32(-np.inf)
                prob = softmax(prob)
                acc = np.zeros(D, dtype=np.float32)
                for t in range(Tk):
                    if prob[t] == 0:
                        continue
                    acc += prob[t] * Vs[t, kv]
                att[qh] = acc
            x = x + gemv(L['wo'], att.reshape(-1))
            h = rms(x, L['ffn_norm'], m['eps'])
            if not L['moe']:
                y = mlp(L['gate'], L['up'], L['down'], h, m['alpha'], m['limit'])
            else:
                logits_r = gemv(L['router'], h)
                prob_e = np.array([sigmoid(v) for v in logits_r], dtype=np.float32)
                score = prob_e + L['bias']
                order = sorted(range(m['n_expert']), key=lambda i: (-float(score[i]), i))[:m['used']]
                w = prob_e[order].astype(np.float32).copy()
                if m['renorm']:
                    s = float(w.sum())
                    if s > 0:
                        w /= np.float32(s)
                y = np.zeros(m['hidden'], dtype=np.float32)
                for j, e in enumerate(order):
                    ye = mlp(L['egate'][e], L['eup'][e], L['edown'][e], h, m['alpha'], m['limit'])
                    y += w[j] * ye
                y *= np.float32(m['scale'])
                if m['shared']:
                    y += mlp(L['sgate'], L['sup'], L['sdown'], h, m['alpha'], m['limit'])
            x = x + y
        h = rms(x, m['out_norm'], m['eps'])
        logits[pos] = gemv(m['output'], h)
    return logits


def _row_dots(rows, vec):
    """rows (T, D) float32, vec (D,) float32. Sequential float64 dot per row."""
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    vec = np.ascontiguousarray(vec, dtype=np.float64)
    acc = np.zeros(rows.shape[0], dtype=np.float64)
    for i in range(rows.shape[1]):
        acc += rows[:, i] * vec[i]
    return acc.astype(np.float32)


def compare(reference, got):
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(got, dtype=np.float32)
    diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    def sm(x):
        z = x.astype(np.float64)
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    p, q = sm(a), sm(b)
    kl = float(np.mean(np.sum(p * np.log((p + 1e-12) / (q + 1e-12)), axis=-1)))
    top = float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))
    return {'max_abs': float(diff), 'kl': kl, 'top1': top}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('gguf', type=Path)
    ap.add_argument('tokens', type=Path, help='whitespace/comma token ids, # comments')
    ap.add_argument('--out', type=Path, help='write fp32 logits as .npy')
    ap.add_argument('--last-k', type=int, default=0)
    args = ap.parse_args(argv)
    text = args.tokens.read_text(encoding='utf-8')
    ids = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].replace(',', ' ')
        ids.extend(int(p) for p in line.split())
    logits = forward(load(args.gguf), ids)
    if args.last_k > 0:
        logits = logits[-args.last_k:]
    if args.out:
        np.save(args.out, logits)
    print(f'rows={logits.shape[0]} cols={logits.shape[1]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
