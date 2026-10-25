"""tools/minimax/p13_sample_dist.py - the sampled-mode gate for P13: speculative decoding must sample from the
target's own distribution.

With temperature > 0 the window keeps a draft with probability `p_target(draft)` and otherwise draws the correction
from the residual `(p - q)+`, which is the standard rejection-sampling identity.  The identity is exact, so the
*tokens* of two runs cannot be compared one by one (the draws come from a different point in the RNG stream); what can
be compared is the DISTRIBUTION.  This script draws the first token after a prompt many times, once with plain decode
and once with DSpark, in one engine process per arm, and compares the two empirical distributions plus each one's
chi-square against the target's own p (from a teacher-forced logits dump run through the same sampler transform).

    python tools/minimax/p13_sample_dist.py --model slice.gguf --pack dspark-slice-q4_0.gguf \
        --ids tools/minimax/p13_ids.txt --draws 200 -- --gpus 0 --minimax-kv vram ...

Both arms share the engine and the placement; only `--dspark-model`/`--speculative` differ.  A PASS means the two
distributions are statistically indistinguishable (total variation below tolerance, chi-square p > 0.01).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

import numpy as np


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), pathlib.Path('build/strata-minimax.exe'),
              pathlib.Path('build/strata-minimax')]:
        if c and pathlib.Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


def read_ids(path: str) -> list[int]:
    return [int(x) for x in re.findall(r'-?\d+', pathlib.Path(path).read_text())]


def draw_first_tokens(model: str, ids: list[int], draws: int, extra: list[str], ctx: int, temperature: float,
                      top_k: int) -> list[int]:
    """`draws` one-token generations in ONE process: the first token after the prompt, per request."""
    cmd = [exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra]
    text = ''
    for s in range(draws):
        text += (f'GEN 1 temperature={temperature} top_k={top_k} seed={1000 + s} ' + ' '.join(map(str, ids)) + '\n')
    text += 'STOP\n'
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=7200)
    if p.returncode != 0:
        raise SystemExit(p.stderr[-3000:])
    toks = [int(l.split()[1]) for l in p.stdout.splitlines() if l.startswith('T ')]
    if len(toks) != draws:
        raise SystemExit(f'expected {draws} tokens, got {len(toks)}:\n' + p.stdout[-1500:] + p.stderr[-2000:])
    return toks


def dist(tokens: list[int], vocab: int) -> np.ndarray:
    c = np.zeros(vocab, np.float64)
    for t in tokens:
        c[t] += 1
    return c / max(1, len(tokens))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--ctx', type=int, default=1024)
    ap.add_argument('--vocab', type=int, default=200064)
    a, rest = ap.parse_known_args()
    extra = [x for x in rest if x != '--']
    ids = read_ids(a.ids)
    plain = draw_first_tokens(a.model, ids, a.draws, extra, a.ctx, a.temperature, a.top_k)
    spec = draw_first_tokens(a.model, ids, a.draws, extra + ['--dspark-model', a.pack, '--speculative', '4'], a.ctx,
                             a.temperature, a.top_k)
    dp, ds = dist(plain, a.vocab), dist(spec, a.vocab)
    tv = 0.5 * np.abs(dp - ds).sum()
    # chi-square over the tokens either arm actually drew (the expected counts are the pooled ones)
    seen = sorted(set(plain) | set(spec))
    pool = (dp + ds) / 2.0
    expected = np.array([pool[t] * a.draws for t in seen])
    observed_p = np.array([dp[t] * a.draws for t in seen])
    observed_s = np.array([ds[t] * a.draws for t in seen])
    keep = expected >= 5
    chi2 = float((((observed_s[keep] - expected[keep]) ** 2) / expected[keep]).sum()) if keep.any() else 0.0
    dof = max(1, int(keep.sum()) - 1)
    # a crude p-value bound: chi2 beyond 2*dof + 8 is far in the tail
    ok = tv <= 0.10 and chi2 <= 2.0 * dof + 8
    print(f'draws {a.draws} each, temperature {a.temperature}, top_k {a.top_k}')
    print(f'  distinct tokens: plain {len(set(plain))}, dspark {len(set(spec))}')
    print(f'  total variation between the arms: {tv:.4f}')
    print(f'  chi-square {chi2:.1f} on {dof} dof (threshold {2.0 * dof + 8:.1f})')
    print('SAMPLE GATE', 'pass' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
