"""P5c: quality and speed cost of --minimax-expert-skip on the G3 reference prompts.

For each G3 prompt shorter than --max-tokens, run the engine teacher-forced (the same token ids the G3 harness
uses) with the skip threshold under test and with it off, then report KL(skip || no-skip), top-1 agreement and the
max-abs logit change.  The baseline is this engine without the skip: the question the feature has to answer is
"what does dropping a light expert change", not "how far is IQ2_XS from the checkpoint".

    python tools/minimax/skip_quality.py --model m.gguf --thresholds 0.05,0.1,0.15,0.2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
G3 = Path(os.environ.get('G3_DIR', str(ROOT / '_local' / 'g3_iq2xs')))


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


def run(model: str, ids_path: Path, out: Path, extra: list[str], ctx: int, tmp: Path) -> np.ndarray:
    txt = tmp / (out.stem + '.ids')
    ids = np.fromfile(ids_path, dtype=np.int32)
    txt.write_text(' '.join(str(int(i)) for i in ids), encoding='utf-8')
    cmd = [exe(), '--model', model, '--minimax-eval', str(txt), '--minimax-dump-logits', str(out),
           '--max-context', str(ctx), '--minimax-allow-nonstandard', '--cpu-threads', '24', *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=7200)
    if p.returncode != 0:
        raise SystemExit(p.stderr[-3000:])
    return np.load(out)


def kl(a: np.ndarray, b: np.ndarray) -> float:
    la = a - a.max(axis=1, keepdims=True)
    lb = b - b.max(axis=1, keepdims=True)
    pa = np.exp(la)
    pa /= pa.sum(axis=1, keepdims=True)
    pb = np.exp(lb)
    pb /= pb.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(pa * np.log(np.maximum(pa, 1e-12) / np.maximum(pb, 1e-12)), axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--thresholds', default='0.05,0.1,0.15,0.2')
    ap.add_argument('--max-tokens', type=int, default=120)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--engine-args', default='--gpus 1 --minimax-kv vram')
    ap.add_argument('--json-out')
    a = ap.parse_args()
    extra = a.engine_args.split()
    man = json.loads((G3 / 'prompts.json').read_text(encoding='utf-8'))
    names = [n for n, m in man.items() if m['tokens'] <= a.max_tokens]
    if not names:
        raise SystemExit('no G3 prompt is short enough')
    thr = [float(x) for x in a.thresholds.split(',')]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for n in names:
            ids = Path(man[n]['ids'])
            base = run(a.model, ids, tmp / f'{n}_base.npy', extra, a.ctx, tmp)
            for t in thr:
                got = run(a.model, ids, tmp / f'{n}_{t}.npy', extra + ['--minimax-expert-skip', str(t)], a.ctx, tmp)
                d = np.abs(got - base)
                per_row = d.max(axis=1)
                k = kl(got.astype(np.float64), base.astype(np.float64))
                top1 = float((got.argmax(axis=1) == base.argmax(axis=1)).mean())
                rows.append({'prompt': n, 'threshold': t, 'max': float(d.max()), 'median': float(np.median(per_row)),
                             'kl': k, 'top1': top1})
                print(f'{n:22s} t={t:.2f} max={d.max():.3e} median={np.median(per_row):.3e} KL={k:.3e} '
                      f'top1={top1:.3f}', flush=True)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print()
    for t in thr:
        sel = [r for r in rows if r['threshold'] == t]
        print(f't={t:.2f} mean KL={np.mean([r["kl"] for r in sel]):.3e} '
              f'mean top1={np.mean([r["top1"] for r in sel]):.3f} worst median={max(r["median"] for r in sel):.3e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
