"""tools/minimax/bench_dspark.py - plain decode against DSpark speculative decode, on one model.

Both arms run the same binary and the same placement; the second adds --dspark-model and --speculative.  The engine
reports its own numbers (the DONE line's decode_ms, the `INFO minimax-spec` line's draft/verify split and the acceptance
counts, the tier line's SSD bytes, `--minimax-profile-time`'s per-token phases), and the tokens of the two arms are
compared, so an arm that is faster but not identical is reported as such instead of as a win.

Two prompt shapes:
  --ids FILE           a token-id file (the quick A/B on a short prompt)
  --doc FILE           the sustained test: the first --doc-tokens tokens of a real document plus a question, one long
                       generation of --decode tokens (Maya's calibration shape: ~8k-token document, ~1,500 tokens)

    python tools/minimax/bench_dspark.py --model <full model.gguf> --pack <dspark-q4_0.gguf> \
        --doc docs/DETAILS.md --doc-tokens 8000 --decode 1500 --temperature 0.7 --reps 3 --spec 4 -- \
        --gpus 0,1 --layer-split auto --minimax-profile mm3-d3.bin --memory-limit-mib 86000 \
        --minimax-prefill-batch 4096 --minimax-profile-time --minimax-stats

Every arm gets a fresh engine process (a fresh tier plan and page cache).  With --warmup the arm first answers once
(not counted), which is where the caches settle.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


LAUNCH = HERE / 'pc_launch.py'


def via_launcher(cmd: list[str], owner: str, est_gb: float | None) -> list[str]:
    """Every engine run on the shared PC goes through pc_launch.py (the lock, the RAM floor, the watchdog)."""
    if owner == 'none':
        print('[bench] --launch-owner none: running the engine WITHOUT the shared-PC lock', file=sys.stderr)
        return cmd
    if not LAUNCH.exists():
        raise SystemExit('tools/minimax/pc_launch.py is missing: every run on the shared PC goes through it '
                         '(pass --launch-owner none to override, which skips the lock and the RAM floor)')
    args = [sys.executable, str(LAUNCH), '--owner', owner]
    if est_gb is not None:
        args += ['--est-ram-gb', f'{est_gb:g}']
    return args + ['--'] + cmd


def exe() -> str:
    for c in [os.environ.get('STRATA_MINIMAX_EXE'), ROOT / 'build' / 'strata-minimax.exe',
              ROOT / 'build' / 'strata-minimax']:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('strata-minimax is not built')


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.environ.get('STRATA_DSPARK_TOKENIZER',
                                              'D:/hf-d3/IQ2_XS-D3/tokenizer/tokenizer.json'))


def read_ids(path: str) -> list[int]:
    txt = Path(path).read_text(encoding='utf-8', errors='replace')
    return [int(x) for x in re.findall(r'-?\d+', txt)]


def doc_prompt(path: str, n_tokens: int, question: str) -> list[int]:
    """The sustained test's prompt: `n_tokens` of a real document, then a question about it."""
    tok = tokenizer()
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    ids = tok.encode(text).ids[:n_tokens]
    return ids + tok.encode('\n\n' + question + '\n\nAnswer:').ids


def run(model: str, prompt: list[int], decode: int, extra: list[str], ctx: int, temperature: float = 0.0,
        seed: int = 1234, warmup: int = 0, owner: str = 'dspark', est_gb: float | None = None):
    cmd = via_launcher([exe(), '--model', model, '--serve', '--max-context', str(ctx), *extra], owner, est_gb)
    gen = f'GEN {decode} temperature={temperature} seed={seed} ' + ' '.join(map(str, prompt))
    text = ''
    if warmup:
        text += f'GEN {warmup} temperature={temperature} seed={seed} ' + ' '.join(map(str, prompt)) + '\n'
    text += gen + '\nSTOP\n'
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=10800)
    if p.returncode != 0:
        raise SystemExit(p.stderr[-3000:])
    out_lines = p.stdout.splitlines()
    dones = [i for i, l in enumerate(out_lines) if l.startswith('DONE ')]
    if not dones:
        raise SystemExit('no DONE line:\n' + p.stdout[-1500:] + p.stderr[-2500:])
    seg = out_lines[dones[0] + 1:] if len(dones) > 1 else out_lines[:]
    toks = [int(l.split()[1]) for l in seg if l.startswith('T ')]
    f = out_lines[dones[-1]].split()
    out = {'gen': int(f[1]), 'prompt': int(f[2]), 'prompt_ms': float(f[3]), 'decode_ms': float(f[4]),
           'tokens': toks, 'spec': {}, 'ssd_bytes': 0, 'phase_ms': {}, 'rounds': 0}
    for line in p.stderr.splitlines():
        if line.startswith('INFO minimax-spec '):
            out['spec'] = {}
            for k, v in re.findall(r'(\w+)=([-\d.]+)', line):
                out['spec'][k] = float(v) if ('.' in v or 'e' in v) else int(v)
        elif line.startswith('INFO minimax-stats token='):
            m = re.search(r'ssd_bytes=(\d+)', line)
            if m:
                out['ssd_bytes'] = max(out['ssd_bytes'], int(m.group(1)))
        elif line.startswith('INFO minimax-profile '):
            for k, v in re.findall(r'(\w+)=([-\d.]+)', line):
                out['phase_ms'].setdefault(k, []).append(float(v))
    return out


def arm_report(name: str, runs: list[dict]) -> list[str]:
    rate = [r['gen'] / r['decode_ms'] * 1000.0 for r in runs]
    ms = [r['decode_ms'] / max(1, r['gen']) for r in runs]
    lines = [f'{name}: {statistics.median(rate):.2f} tok/s (median of ' + ', '.join(f'{v:.2f}' for v in rate) +
             f'), {statistics.median(ms):.0f} ms/token, ' +
             f'{statistics.median([r["ssd_bytes"] / 2**20 / max(1, r["gen"]) for r in runs]):.1f} MB SSD/token']
    spec = [r for r in runs if r['spec']]
    if spec:
        s = spec[-1]['spec']
        rounds = max(1, s.get('rounds', 0))
        lines.append(f'  drafts {s.get("drafts", 0)} accepted {s.get("accepted", 0)} '
                     f'rate {s.get("rate", 0):.3f} per-round {s.get("per_round", 0):.2f} tokens | '
                     f'draft {s.get("draft_ms", 0) / rounds:.1f} ms + verify {s.get("verify_ms", 0) / rounds:.1f} ms '
                     f'per round')
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', default='')
    ap.add_argument('--ids')
    ap.add_argument('--doc')
    ap.add_argument('--doc-tokens', type=int, default=8000)
    ap.add_argument('--question', default='Summarise the document in a few paragraphs, then list its main sections.')
    ap.add_argument('--decode', type=int, default=64)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--spec', default='4')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--ctx', type=int, default=16384)
    ap.add_argument('--plain-only', action='store_true')
    ap.add_argument('--launch-owner', default='dspark',
                    help="the shared-PC lock owner for pc_launch.py (default dspark; 'none' skips the launcher)")
    ap.add_argument('--est-ram-gb', type=float, default=None,
                    help="the run's host-RAM estimate for the launcher (default: pc_launch estimates it)")
    a, rest = ap.parse_known_args()
    extra = [x for x in rest if x != '--']
    base_ids = read_ids(a.ids) if a.ids else None
    arms = [('plain', [])]
    if not a.plain_only:
        if not a.pack:
            raise SystemExit('--pack is needed for the speculative arm')
        arms.append(('dspark', ['--dspark-model', a.pack, '--speculative', str(a.spec)]))
    res = {n: [] for n, _ in arms}
    for rep in range(a.reps):
        for name, args in arms:
            if base_ids:
                prompt = list(base_ids)
                prompt[-1] = (prompt[-1] + rep + 1) % 200000
            else:
                prompt = doc_prompt(a.doc, a.doc_tokens, a.question)
            r = run(a.model, prompt, a.decode, extra + args, a.ctx, a.temperature, a.seed + rep, a.warmup,
                    a.launch_owner, a.est_ram_gb)
            r['tok_s'] = r['gen'] / r['decode_ms'] * 1000.0
            r['ssd_mb'] = r['ssd_bytes'] / 2 ** 20
            res[name].append(r)
            print(f'rep {rep} {name}: {r["tok_s"]:.2f} tok/s, {r["decode_ms"]:.0f} ms decode, '
                  f'{r["prompt"]} prompt tokens in {r["prompt_ms"]:.0f} ms, '
                  f'ssd {r["ssd_mb"] / max(1, r["gen"]):.1f} MB/token' +
                  (f', spec {r["spec"]}' if r['spec'] else ''), flush=True)
    print()
    for name, _ in arms:
        for line in arm_report(name, res[name]):
            print(line)
    for name, _ in arms:
        runs = [r for r in res[name] if r['phase_ms']]
        if not runs:
            continue
        keys = list(runs[0]['phase_ms'].keys())
        print(f'\n{name} phase ms/token (median over {len(runs)} runs):')
        for k in keys:
            vals = [statistics.median(r['phase_ms'][k]) for r in runs if k in r['phase_ms']]
            if vals:
                print(f'  {k:12s} {statistics.median(vals):7.2f}')
    if len(arms) == 2:
        same = all(res['plain'][i]['tokens'] == res['dspark'][i]['tokens'] for i in range(a.reps))
        print(f'\ntokens: {"IDENTICAL" if same else "DIFFER"} ({len(res["plain"][0]["tokens"])} per run)')
        for i in range(a.reps):
            if res['plain'][i]['tokens'] != res['dspark'][i]['tokens']:
                x, y = res['plain'][i]['tokens'], res['dspark'][i]['tokens']
                first = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), None)
                print(f'  rep {i}: first difference at {first} of {len(x)}')
        rate = [res['dspark'][i]['tok_s'] / res['plain'][i]['tok_s'] - 1.0 for i in range(a.reps)]
        print(f'gain: {" ".join(f"{100 * v:+.1f}%" for v in rate)} (median {100 * statistics.median(rate):+.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
