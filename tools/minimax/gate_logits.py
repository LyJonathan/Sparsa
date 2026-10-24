"""Correctness gate: teacher-forced logits of two engine builds (or one build with two arg sets) on the same
token ids, reported as max-abs, KL and top-1 agreement.

The gate every decode optimisation in P6/P7 has to pass is "the bytes are the same bytes": if a change only
moves where an expert is read from, the logits must be identical, and this script fails (exit 1) unless
--allow differ lets them differ.  For a lossy opt-in flag, pass --allow and the numbers are the report.

    python tools/minimax/gate_logits.py --exe A/strata-minimax.exe --exe B/strata-minimax.exe \
        --model slice.gguf --ids ids.bin --engine-args "..."

Each arm gets a fresh process.  The ids file is int32 (the G3 dumps' format) or text.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(exe: str, model: str, ids: list[int], out: Path, extra: list[str], ctx: int) -> np.ndarray:
    txt = out.with_suffix('.ids')
    txt.write_text(' '.join(str(i) for i in ids), encoding='utf-8')
    cmd = [exe, '--model', model, '--minimax-eval', str(txt), '--minimax-dump-logits', str(out),
           '--max-context', str(ctx), '--minimax-allow-nonstandard', *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=7200)
    if p.returncode != 0:
        raise SystemExit(f'{exe} failed rc={p.returncode}:\n' + p.stderr[-3000:])
    return np.load(out)


def kl(a: np.ndarray, b: np.ndarray) -> float:
    la = a - a.max(axis=1, keepdims=True)
    lb = b - b.max(axis=1, keepdims=True)
    pa = np.exp(la)
    pa /= pa.sum(axis=1, keepdims=True)
    pb = np.exp(lb)
    pb /= pb.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(pa * np.log(np.maximum(pa, 1e-12) / np.maximum(pb, 1e-12)), axis=1)))


def read_ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    if path.suffix.lower() == '.bin' and len(raw) % 4 == 0:
        return [int(x) for x in np.frombuffer(raw, dtype=np.int32)]
    return [int(x) for x in raw.decode('utf-8', 'replace').replace(',', ' ').split() if x.lstrip('-').isdigit()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', action='append', required=True)
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--arm-args', action='append', default=[],
                    help='extra engine args for one arm, in --exe order (repeatable, spaces allowed)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--allow', action='store_true', help='the change is opt-in lossy: report, do not fail')
    ap.add_argument('--json-out')
    a = ap.parse_args()
    if not a.label:
        a.label = [f'arm{i}' for i in range(len(a.exe))]
    while len(a.arm_args) < len(a.exe):
        a.arm_args.append('')
    ids = read_ids(Path(a.ids))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logs = []
        for i, exe in enumerate(a.exe):
            print(f'arm {a.label[i]}: {exe} {a.arm_args[i]}', flush=True)
            logs.append(run(exe, a.model, ids, tmp / f'l{i}.npy', a.arm_args[i].split(), a.ctx))
    base = logs[0]
    for i, got in enumerate(logs):
        d = np.abs(got.astype(np.float64) - base.astype(np.float64))
        k = kl(got.astype(np.float64), base.astype(np.float64))
        top1 = float((got.argmax(axis=1) == base.argmax(axis=1)).mean())
        same = bool(np.array_equal(got, base))
        rows.append({'arm': a.label[i], 'max_abs': float(d.max()), 'mean_abs': float(d.mean()), 'kl': k,
                     'top1': top1, 'identical': same})
        print(f'{a.label[i]:>12}  max {d.max():.3e}  KL {k:.3e}  top1 {top1:.3f}  '
              f'{"identical" if same else "DIFFERS"}', flush=True)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, indent=1), encoding='utf-8')
    ok = a.allow or all(r['identical'] for r in rows[1:])
    print('GATE', 'pass' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
