"""`output_rel` for a published pack's expert tensor, on the real routed tokens of one layer.

`strata-mm-quant --exp-layer` measures the output error of *its own* quantization; this measures the output error of
the **stored bytes of a published pack** against the source checkpoint, on the tokens the router actually sent to each
expert:

    output_rel = || x W^T - x Q^T || / || x W^T ||

with `x` the layer's routed-expert input (rebuilt from a per-op reference dump by `quant/exp_input.py`'s
reconstruction) for gate/up, and the expert's own hidden state for down.  Nothing here starts the engine.

Two dump facts this has to respect: the dumps exist only for layers 0, 3 and 40 (a per-op dump is a reference
forward pass, so this cannot cover all 57 MoE layers today), and `input_layernorm.weight` of this checkpoint contains
a channel at exactly -1, i.e. a norm weight of 0 after the +1 bake, which is what made `exp_input.py` divide by zero
and return NaN.  That channel carries no information (its normalised value is 0 regardless of the residual), so it is
masked out rather than approximated.

    python tools/minimax/output_rel_probe.py --layer 3 \
        --dump _local/ref_out/dumps/short/dump-L03 \
        --pack d3=D:\\hf-d3\\... --pack iq1m=... --pack iq2xxs=... --experts 12
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'minimax' / 'quant'))
from exp_input import load_layer_tensors, rms_norm  # noqa: E402
from verify_nvfp4_gguf import Safetensors, dequant_spec_a  # noqa: E402

ROLES = {'ffn_gate_exps.weight': ('w1', 'gate'), 'ffn_up_exps.weight': ('w3', 'up'),
         'ffn_down_exps.weight': ('w2', 'down')}


def reconstruct(model: str, dump: Path, layer: int, eps: float = 1e-6):
    """The layer's routed-expert input for every token, plus the fitted residual scale."""
    t = load_layer_tensors(model, layer)
    ai = np.load(dump / 'attn_input.npy').reshape(-1, t['in'].shape[0]).astype(np.float64)
    ao = np.load(dump / 'attn_out.npy').reshape(-1, ai.shape[1]).astype(np.float64)
    logits = np.load(dump / 'router_logits.npy').reshape(ai.shape[0], -1).astype(np.float64)
    denom = t['in'] + 1.0
    bad = np.abs(denom) < 1e-3
    denom = np.where(bad, 1.0, denom)
    u = ai / denom                       # ai is 0 where the weight is 0, so masking keeps u = 0 there
    post_w = t['post'] + 1.0
    gate_w = t['gate']

    def cost(tk: int, s: float) -> float:
        x = rms_norm(s * u[tk] + ao[tk], post_w, eps)
        p = x @ gate_w.T
        return float(np.linalg.norm(p - logits[tk]) / np.linalg.norm(logits[tk]))

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = np.log(1e-2), np.log(2e3)
    scales = np.empty(ai.shape[0])
    for tk in range(ai.shape[0]):
        a, b = lo, hi
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = cost(tk, np.exp(c)), cost(tk, np.exp(d))
        for _ in range(28):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = cost(tk, np.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = cost(tk, np.exp(d))
        scales[tk] = np.exp((a + b) / 2.0)
    x = np.stack([rms_norm(scales[tk] * u[tk] + ao[tk], post_w, eps) for tk in range(ai.shape[0])])
    return x.astype(np.float64), scales, int(bad.sum())


def pack_tensor(of: str, pattern: str, name: str):
    from gguf import GGUFReader, quants
    for f in sorted(glob.glob(pattern)):
        r = GGUFReader(f)
        for t in r.tensors:
            if t.name != name:
                continue
            with open(f, 'rb') as fh:
                fh.seek(int(t.data_offset))
                raw = fh.read(int(t.n_bytes))
            if len(raw) != int(t.n_bytes):
                raise SystemExit(f'{f}: short read for {name}')
            n_exp = int(t.shape[2]) if len(t.shape) == 3 else 1
            n_out, n_in = int(t.shape[1]), int(t.shape[0])
            return np.frombuffer(raw, dtype=np.uint8), n_exp, n_out, n_in, t.tensor_type
    raise SystemExit(f'{of}: no tensor {name}')


def expert_matrix(raw, qtype, e: int, n_exp: int, n_out: int, n_in: int):
    from gguf import quants
    row_bytes = len(raw) // n_exp
    q = quants.dequantize(raw[e * row_bytes:(e + 1) * row_bytes], qtype)
    return np.asarray(q, dtype=np.float64).reshape(n_out, n_in)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash")
    ap.add_argument('--dump', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--pack', action='append', required=True, help='label=glob')
    ap.add_argument('--experts', type=int, default=12)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    dump = Path(args.dump)
    x, scales, masked = reconstruct(args.model, dump, args.layer)
    print(f'layer {args.layer}: {x.shape[0]} tokens, residual scale median {np.median(scales):.2f}, '
          f'{masked} norm channels masked', flush=True)
    st = Safetensors(Path(args.model))
    rows = []
    for leaf, (wkey, role) in ROLES.items():
        name = f'blk.{args.layer}.{leaf}'
        out = {'layer': args.layer, 'role': role, 'tensor': name, 'packs': {}}
        # the tokens each expert was routed; `down` uses the expert's own hidden state
        probs = sorted(p for p in dump.iterdir() if p.name.startswith('expert_') and p.name.endswith('_token_idx.npy'))
        routed = {int(p.name[len('expert_'):-len('_token_idx.npy')]): np.load(p) for p in probs}
        for spec in args.pack:
            label, _, pattern = spec.partition('=')
            raw, n_exp, n_out, n_in, qtype = pack_tensor(label, pattern, name)
            idx = sorted(routed)[: args.experts]
            num = den = 0.0
            for e in idx:
                q = expert_matrix(raw, qtype, e, n_exp, n_out, n_in)
                w = source_expert(st, args.layer, e, wkey)
                if w.shape != q.shape:
                    raise SystemExit(f'{name} {label}: {q.shape} vs source {w.shape}')
                if role == 'down':
                    hid = np.load(dump / f'expert_{e:03d}_hidden.npy').astype(np.float64)
                    xr = hid
                else:
                    xr = x[routed[e]]
                a = xr @ w.T
                b = xr @ q.T
                num += float(np.linalg.norm(a - b) ** 2)
                den += float(np.linalg.norm(a) ** 2)
            out['packs'][label] = {'output_rel': round((num / den) ** 0.5, 5) if den else None,
                                   'experts': len(idx), 'bytes': len(raw)}
        rows.append(out)
        print(f'  {role}: ' + '  '.join(f'{k} {v["output_rel"]:.4f}' for k, v in out['packs'].items()), flush=True)
    Path(args.out).write_text(json.dumps({'layer': args.layer, 'dump': str(dump), 'rows': rows}, indent=1),
                              encoding='utf-8')
    print(f'wrote {args.out}')
    return 0


def source_expert(st: Safetensors, layer: int, e: int, wkey: str):
    base = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{wkey}'
    packed, _ = st.get(base + '.weight_packed')
    scales, _ = st.get(base + '.weight_scale')
    gs, _ = st.get(base + '.weight_global_scale')
    g = float(np.asarray(gs).reshape(-1)[0])
    return dequant_spec_a(np.asarray(packed), np.asarray(scales), g).astype(np.float64)


if __name__ == '__main__':
    sys.exit(main())
