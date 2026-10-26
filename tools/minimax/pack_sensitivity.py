"""Per-tensor sensitivity of every MoE expert tensor, measured against the source checkpoint.

The A1 recipe needs one number per (layer, projection, candidate type): how much
error that choice adds.  RCO's `--init sensitivity` is `delta^2/n_params` on a
dequantized DB; we can do better with what is on this machine - the FP32/NVFP4
checkpoint the packs were quantized from (`models/MiniMax-M3.1-Flash`, per-expert
`weight_packed` + `weight_scale` + `weight_global_scale`) - and compute, for a
published pack's actual stored bytes, the relative Frobenius error

    relF = ||W - Q||_F / ||W||_F        over sampled experts of one tensor

i.e. the same `plain_rel` column `strata-mm-quant --exp-layer` reports, but read
straight from the pack so it describes the bytes we would actually ship, not a
re-quantization with our own imatrix (which the published tiers do not record).

Why sampled experts: the full grid is 171 expert tensors x 3 packs and dequantizing
both sides is ~790 GB of reads with the array work on top.  `--experts 8` takes 8
evenly spaced experts per tensor - the ranking is per tensor, and an 8-expert mean
of a 128-expert tensor is a stable ranking statistic (shown to agree with the block
KL64 deltas in `README.md`, section "A1: the sensitivity table").

Sizes come from the GGUF tensor tables themselves (`n_bytes`), not from a formula:
a group's cost is what the pack really stores.

    python tools/minimax/pack_sensitivity.py \
        --pack d3=D:\\hf-d3\\IQ2_XS-D3\\*.gguf \
        --pack iq1m=D:\\hf-iq1m\\...\\*.gguf --pack iq2xxs=D:\\hf-iq2xxs\\...\\*.gguf \
        --experts 8 --out _local/a1/sensitivity.json

Engine-free and lock-free (CPU and disk only, low RAM: one expert at a time), so it
runs in the gaps between PC turns and never beside a gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'quant'))
from verify_nvfp4_gguf import Safetensors, dequant_spec_a  # noqa: E402

ROLES = {'ffn_gate_exps.weight': ('w1', 'gate'), 'ffn_up_exps.weight': ('w3', 'up'),
         'ffn_down_exps.weight': ('w2', 'down')}


def pack_tensors(pattern: str) -> dict[str, list[tuple[Path, object]]]:
    """tensor name -> [(shard path, tensor info)] for one split pack."""
    from gguf import GGUFReader
    out: dict[str, list[tuple[Path, object]]] = {}
    for f in sorted(glob.glob(pattern)):
        r = GGUFReader(f)
        for t in r.tensors:
            out.setdefault(t.name, []).append((Path(f), t))
    return out


def raw_of(path: Path, tensor) -> np.ndarray:
    """The tensor's stored bytes, read straight from the shard."""
    r = tensor  # gguf reader's TensorInfo, with data offsets already resolved
    start = int(r.data_offset)
    n = int(r.n_bytes)
    with open(path, 'rb') as f:
        f.seek(start)
        buf = f.read(n)
    if len(buf) != n:
        raise SystemExit(f'{path.name}: short read for {r.name}')
    return np.frombuffer(buf, dtype=np.uint8)


def source_expert(st: Safetensors, layer: int, e: int, wkey: str):
    base = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{wkey}'
    packed, _ = st.get(base + '.weight_packed')
    scales, _ = st.get(base + '.weight_scale')
    gs, _ = st.get(base + '.weight_global_scale')
    g = float(np.asarray(gs).reshape(-1)[0])
    return dequant_spec_a(np.asarray(packed), np.asarray(scales), g)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', action='append', required=True, help='label=glob, repeatable (first is the base)')
    ap.add_argument('--model', default=r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash")
    ap.add_argument('--experts', type=int, default=8, help='evenly spaced experts per tensor (0 = all 128)')
    ap.add_argument('--layers', default='3-59')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nbytes-only', action='store_true',
                    help='only tabulate the sizes (no dequantization, seconds not hours)')
    args = ap.parse_args(argv)

    from gguf import quants
    packs = []
    for spec in args.pack:
        label, _, pattern = spec.partition('=')
        packs.append((label, pack_tensors(pattern)))
        print(f'{label}: {len(packs[-1][1])} tensors', flush=True)
    lo, hi = (int(x) for x in args.layers.split('-'))
    layers = [L for L in sorted({int(n.split('.')[1]) for n in packs[0][1] if n.startswith('blk.')})
              if lo <= L <= hi]
    st = None if args.nbytes_only else Safetensors(Path(args.model))

    rows = []
    t0 = time.time()
    for L in layers:
        for leaf, (wkey, role) in ROLES.items():
            name = f'blk.{L}.{leaf}'
            row = {'layer': L, 'role': role, 'tensor': name, 'packs': {}}
            for label, table in packs:
                if name not in table:
                    raise SystemExit(f'{label}: no {name}')
                path, info = table[name][0]
                row['packs'][label] = {'type': str(info.tensor_type), 'bytes': int(info.n_bytes)}
            if not args.nbytes_only:
                n_exp = int(info.shape[2]) if len(info.shape) == 3 else 1
                n_out, n_in = int(info.shape[1]), int(info.shape[0])
                idx = (np.linspace(0, n_exp - 1, args.experts).round().astype(int) if args.experts
                       else np.arange(n_exp))
                for label, table in packs:
                    path, info = table[name][0]
                    raw = raw_of(path, info)
                    row_bytes = len(raw) // n_exp
                    num = den = 0.0
                    for e in idx:
                        q = quants.dequantize(raw[e * row_bytes:(e + 1) * row_bytes], info.tensor_type)
                        q = np.asarray(q, dtype=np.float32).reshape(n_out, n_in)
                        w = source_expert(st, L, int(e), wkey)
                        if w.shape != q.shape:
                            raise SystemExit(f'{name} {label}: {q.shape} vs source {w.shape}')
                        d = q - w
                        num += float(np.dot(d.ravel(), d.ravel()))
                        den += float(np.dot(w.ravel(), w.ravel()))
                    row['packs'][label]['relf'] = round((num / den) ** 0.5, 5) if den else None
                    row['packs'][label]['experts'] = len(idx)
            rows.append(row)
        if L % 4 == 0 or L == layers[-1]:
            print(f'  layer {L} done, {time.time() - t0:.0f}s', flush=True)

    out = {'model': args.model, 'experts': args.experts, 'rows': rows,
           'seconds': round(time.time() - t0, 1), 'labels': [p[0] for p in packs]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'wrote {args.out}: {len(rows)} groups in {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
