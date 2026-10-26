"""Run several G3 arms back to back inside ONE PC turn, and rank them.

The sweep's whole point is that a block's KL64 delta is a statement about *that*
prompt set, measured in the *same session* as the baseline.  Two engines started
minutes apart on a shared box are two different sessions: the page cache, the
tier admission and the CPU contention all move.  So the arms of one comparison
run under a single lock, as one child of `pc_launch.py`:

    python tools/minimax/pc_launch.py --owner maya --est-ram-gb 74 --max-minutes 45 -- \
        python tools/minimax/arm_session.py --arms "d3:;b3-7:3-7" --out <dir>

An arm is `name=spec`:

    d3:        the base pack, no override
    b3-7:3-7   the base pack with layers 3-7's experts read from --cold
               (`--expert-source 3-7=<cold>`, no file written)
    cand:@path an arm that *is* a file (a mixed pack); the base path still names the profile

Each arm gets its own output directory, so `g3_sparsa.py`'s per-prompt dumps
resume: a turn that hits `pc_launch`'s time limit leaves the completed prompts
behind and the next invocation continues them instead of starting over.  For the
same reason this driver never passes `--force` - the dumps in an arm's own
directory are that arm's own bytes.

At the end it reads every arm's `results.json` and prints one table: positions,
top-1, KL64, the delta against the first arm, GB saved, and KL64 delta per GB.
A block's rows are what the recipe is chosen from; per-block deltas are never
summed into a candidate's expected quality (the candidate is measured whole).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G3 = ROOT / 'tools' / 'minimax' / 'g3_sparsa.py'
TMP = Path(os.environ.get('TEMP', r'C:\Users\admin\AppData\Local\Temp'))

# the sweep's settings; every arm in a session shares them, which is the point
SWEEP = {'gpus': '0,1', 'vram_slots': '0', 'layer_split': 'auto', 'cpu_threads': '12',
         'max_context': '8192'}
# measured from the two packs' own tensor tables: D3 2,321,547,264 B a layer of experts,
# IQ1_M 1,679,818,752 B, so a layer moved to IQ1_M saves 641,728,512 B = 0.6417 GB
GB_PER_LAYER = 0.6417


def parse_arms(spec: str) -> list[tuple[str, str]]:
    arms = []
    for part in spec.split(';'):
        part = part.strip()
        if not part:
            continue
        name, _, what = part.partition(':')
        name = name.strip()
        if not name:
            raise SystemExit(f'arm without a name in {spec!r}')
        arms.append((name, what.strip()))
    if not arms:
        raise SystemExit('no arms')
    return arms


def arm_cmd(a, base: Path, cold: Path | None, arm_dir: Path, prompts: str) -> list[str]:
    _, what = a
    model = base
    extra: list[str] = []
    if what.startswith('@'):
        model = Path(what[1:])
    elif what:
        if cold is None:
            raise SystemExit('this arm needs --cold')
        # the engine parses `--expert-source LO-HI=<first shard>` as two tokens
        extra += ['--engine-arg=--expert-source', f'--engine-arg={what}={cold}']
    cmd = [sys.executable, '-u', str(G3), '--model', str(model), '--out', str(arm_dir),
           '--tier', a[0], '--ref', a_ref, '--ids', a_ids, '--no-ppl',
           '--memory-limit-mib', a_mem, '--prompts', prompts,
           '--gpus', SWEEP['gpus'], '--vram-slots', SWEEP['vram_slots'],
           '--layer-split', SWEEP['layer_split'], '--cpu-threads', SWEEP['cpu_threads'],
           '--max-context', SWEEP['max_context']]
    return cmd + extra


def scored(arm_dir: Path, wanted: set[str]) -> dict | None:
    f = arm_dir / 'results.json'
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    got = {r['prompt'] for r in d.get('per_prompt') or []}
    if d.get('overall') and got == wanted:
        return d
    return None


def main(argv=None) -> int:
    global a_ref, a_ids, a_mem
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True, help='name:spec;... (spec "" = base, "3-7" = --expert-source, '
                                                 '"@path" = a file arm)')
    ap.add_argument('--base', required=True, help='the base pack, first shard (D3)')
    ap.add_argument('--cold', default=None, help='the donor pack, first shard (IQ1_M)')
    ap.add_argument('--out', required=True, help='root output dir; each arm gets <out>/<name>')
    ap.add_argument('--prompts', default='', help='comma-separated subset; empty = the six short prompts')
    ap.add_argument('--ref', default=r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\ref_out\logits")
    ap.add_argument('--ids', default=r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\g3_iq2xs\ids")
    ap.add_argument('--memory-limit-mib', default='80000')
    ap.add_argument('--gb-per-layer', type=float, default=GB_PER_LAYER,
                    help=f'GB of experts a layer saves against the base (default {GB_PER_LAYER} for the IQ1_M '
                         f'pack; the IQ2_XXS pack is 0.3775)')
    ap.add_argument('--timeout', type=float, default=2400.0, help='per arm, seconds')
    ap.add_argument('--skip-done', action='store_true', default=True,
                    help='an arm whose results.json already covers these prompts is not rerun')
    ap.add_argument('--redo', action='store_true', help='rerun every arm even when its results are complete')
    args = ap.parse_args(argv)

    a_ref, a_ids, a_mem = args.ref, args.ids, str(args.memory_limit_mib)
    base = Path(args.base)
    cold = Path(args.cold) if args.cold else None
    out_root = Path(args.out)
    arms = parse_arms(args.arms)
    names = [n.strip() for n in args.prompts.split(',') if n.strip()]
    wanted = set(names) if names else {'chinese_local_llm', 'code_lru', 'math_bakery',
                                       'multi_turn_threads', 'prose_moe', 'tool_weather'}
    prompts = ','.join(names)

    t0 = time.time()
    for a in arms:
        arm_dir = out_root / a[0]
        arm_dir.mkdir(parents=True, exist_ok=True)
        if not args.redo:
            d = scored(arm_dir, wanted)
            if d:
                o = d['overall']
                print(f'[{a[0]}] already scored: {100*o["top1_agreement"]:.2f}% / {o["kl64_mean"]:.4f} '
                      f'({o["positions"]} pos)', flush=True)
                continue
        cmd = arm_cmd(a, base, cold, arm_dir, prompts)
        print(f'[{a[0]}] {" ".join(cmd[1:])}', flush=True)
        left = args.timeout
        t = time.time()
        log = open(TMP / f'arm_{a[0]}.log', 'w', encoding='utf-8', errors='replace')
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               text=True, timeout=max(60.0, left))
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        finally:
            log.close()
        print(f'[{a[0]}] rc {rc} in {(time.time()-t)/60:.1f} min', flush=True)
        if rc != 0:
            tail = (TMP / f'arm_{a[0]}.log').read_text(encoding='utf-8', errors='replace').splitlines()[-6:]
            for line in tail:
                print('    ' + line, flush=True)
        # a completed arm's partial dumps stay; the next invocation resumes them
        if time.time() - t0 > args.timeout * len(arms):
            print('out of session budget; stopping here', flush=True)
            break

    print('\n' + summary(out_root, arms, wanted, args.gb_per_layer), flush=True)
    return 0


def layers_of(what: str) -> list[int]:
    """The layer range an arm overrides, from its `LO-HI` spec ([] for base/file arms)."""
    if not what or what.startswith('@'):
        return []
    lo, _, hi = what.partition('-')
    return list(range(int(lo), int(hi or lo) + 1))


def summary(out_root: Path, arms: list[tuple[str, str]], wanted: set[str],
            gb_per_layer: float = GB_PER_LAYER) -> str:
    rows = []
    for name, what in arms:
        d = scored(out_root / name, wanted)
        if not d:
            continue
        o = d['overall']
        nlayers = len(layers_of(what))
        rows.append({'arm': name, 'layers': layers_of(what), 'positions': o['positions'],
                     'top1': round(100.0 * o['top1_agreement'], 2),
                     'kl64': round(o['kl64_mean'], 4),
                     'gb_saved': round(nlayers * gb_per_layer, 2) if nlayers else 0.0})
    if not rows:
        return 'no arm produced a complete result'
    base_row = rows[0]
    lines = ['| arm | positions | top-1 | KL64 | dKL64 | top-1 delta | GB saved | dKL64/GB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        r['dkl64'] = round(r['kl64'] - base_row['kl64'], 4)
        r['top1_delta'] = round(r['top1'] - base_row['top1'], 2)
        per = (r['dkl64'] / r['gb_saved']) if r['gb_saved'] else None
        r['dkl64_per_gb'] = round(per, 4) if per is not None else None
        lines.append(f"| {r['arm']} | {r['positions']} | {r['top1']:.2f}% | {r['kl64']:.4f} | "
                     f"{r['dkl64']:+.4f} | {r['top1_delta']:+.2f} | {r['gb_saved']:.2f} | "
                     f"{'' if per is None else f'{per:+.4f}'} |")
    (out_root / 'summary.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    return '\n'.join(lines)


if __name__ == '__main__':
    sys.exit(main())
