"""P5a measurement harness: prefill / decode / TTFT for strata-minimax on one GGUF.

Runs the engine's serve protocol and reads the engine's own PP/DONE numbers (no wall-clock guessing), once per
prompt length and once with each engine configuration. Prints a table. Nothing here is MiniMax-specific except the
token ids: they are random within the file's vocabulary, because the model is measured, not its answers.

    python tools/minimax/bench_p5a.py --model m.gguf --id-range 200000 --lengths 512,4096 \
        -- --gpus 1 --minimax-kv vram --minimax-prefill-batch 4096

Everything after `--` is passed to the engine.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


def ids(n: int, vocab: int, seed: int) -> list[int]:
    x = seed
    out = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % vocab)
    return out


def run(model: str, prompt: list[int], max_new: int, extra: list[str], ctx: int):
    cmd = [exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra]
    text = f'GEN {max_new} temperature=0 ' + ' '.join(map(str, prompt)) + '\nSTOP\n'
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=7200)
    if p.returncode != 0:
        raise SystemExit(p.stderr[-2000:])
    done = [l for l in p.stdout.splitlines() if l.startswith('DONE ')]
    if not done:
        raise SystemExit('no DONE line:\n' + p.stdout[-1000:] + p.stderr[-2000:])
    f = done[0].split()
    # DONE generated prompt prompt_ms decode_ms finish 0 0 reused
    gen, pr, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
    return gen, pr, pms, dms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--id-range', type=int, default=200000)
    ap.add_argument('--lengths', default='512,4096')
    ap.add_argument('--decode', type=int, default=32)
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('engine_args', nargs='*')
    a = ap.parse_args()
    extra = [x for x in a.engine_args if x != '--']
    if '--minimax-allow-nonstandard' not in extra:
        pass
    print(f'engine argv: {" ".join(extra)}')
    print(f'{"prompt":>8} {"prefill tok/s":>14} {"TTFT ms":>10} {"decode tok/s":>13} {"ms/token":>9}')
    for n in [int(x) for x in a.lengths.split(',')]:
        prompt = ids(n, a.id_range, a.seed)
        gen, pr, pms, dms = run(a.model, prompt, a.decode, extra, a.ctx)
        pre = pr / (pms / 1000.0) if pms > 0 else float('nan')
        dec = gen / (dms / 1000.0) if dms > 0 else float('nan')
        print(f'{n:8d} {pre:14.1f} {pms:10.1f} {dec:13.2f} {(dms / max(gen, 1)):9.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
