"""Sensitivity sweep: how much G3 top-1 does one block of MoE layers cost when it goes from D3 to IQ1_M?

A1's per-layer mix is only worth building if the loss per gigabyte saved is small, so the question is *which* layers
carry the least quality: this swaps one block of consecutive MoE layers at a time to the cold donor, gates it with
the same all-CPU harness as the baseline, and prints loss-per-GB.  Blocks are named on the command line, so the same
driver serves a quick 3-block probe and the full 12-block table.

Every step goes through `pc_launch.py` (lock + RAM floor + watchdog + hard timeout), one block per PC turn, and each
candidate pack is deleted right after its gate - a 124 GB pack per block would otherwise fill the drive.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(os.environ.get('TEMP', r'C:\Users\admin\AppData\Local\Temp'))
LOCK = Path(r"E:\Jon's Projects\AI\Strata Fork Minimax\pc-engine.lock")


def lock_free() -> bool:
    try:
        return LOCK.read_bytes().strip() == b'free'
    except OSError:
        return False


def wait_free(deadline: float) -> bool:
    while time.time() < deadline:
        if lock_free():
            return True
        time.sleep(15)
    return False


def run(cmd: list[str], tag: str, est_gb: float, timeout: float, deadline: float) -> int:
    """Run `cmd` under pc_launch, waiting for a free lock first.  Returns the child's exit code (-1 on give-up)."""
    if not wait_free(deadline):
        print(f'{tag}: no free lock before the deadline', flush=True)
        return -1
    launch = [sys.executable, str(ROOT / 'tools' / 'minimax' / 'pc_launch.py'), '--owner', 'maya',
              '--est-ram-gb', str(est_gb), '--min-free-gb', '10', '--'] + cmd
    log = open(TMP / f'sweep_{tag}.log', 'w')
    p = subprocess.Popen(launch, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, text=True,
                         encoding='utf-8', errors='replace')
    try:
        p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        print(f'{tag}: TIMEOUT after {timeout:.0f}s', flush=True)
    log.close()
    print(f'{tag}: rc {p.returncode}', flush=True)
    return p.returncode


def gate(model: Path, out: Path, prompts: str | None, timeout: float, deadline: float) -> dict | None:
    cmd = [sys.executable, str(ROOT / 'tools' / 'minimax' / 'g3_sparsa.py'), '--model', str(model), '--out', str(out),
           '--tier', 'MIX', '--ref', r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\ref_out\logits",
           '--ids', r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\g3_iq2xs\ids",
           '--no-ppl', '--gpus', '0,1', '--vram-slots', '0', '--max-context', '8192', '--force',
           '--memory-limit-mib', '80000', '--layer-split', 'auto', '--cpu-threads', '12']
    if prompts:
        cmd += ['--prompts', prompts]
    if run(cmd, out.name, 30, timeout, deadline) != 0:
        return None
    f = out / 'results.json'
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d.get('overall')


def build(hot: Path, cold: Path, layers: list[int], out: Path, profile: str) -> bool:
    cmd = [sys.executable, str(ROOT / 'tools' / 'minimax' / 'mix_layers.py'), '--hot', str(hot), '--cold', str(cold),
           '--out', str(out), '--profile', profile, '--cold-layers', ','.join(str(x) for x in layers)]
    print('  ' + ' '.join(cmd[1:])[:200], flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(r.stdout[-800:], r.stderr[-800:], flush=True)
    return r.returncode == 0 and out.exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hot', required=True)
    ap.add_argument('--cold', required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--blocks', required=True, help='semicolon-separated blocks, e.g. "3-7;8-12;13-17"')
    ap.add_argument('--pack', default=r'D:\hf-sweep\cand.gguf')
    ap.add_argument('--out', default=str(TMP / 'sweep'))
    ap.add_argument('--prompts', default='chinese_local_llm,code_lru', help='the subset for the sweep')
    ap.add_argument('--full-prompts', default='',
                    help='also gate these blocks with the full set, to check the subset ranks the same way')
    ap.add_argument('--full-blocks', default='', help='which blocks get the full gate on top of the subset')
    ap.add_argument('--est-gb-per-layer', type=float, default=0.688,
                    help='GB saved per layer moved to the cold donor (D3 2.322 -> IQ1_M 1.680 GB a layer)')
    ap.add_argument('--minutes', type=float, default=120.0, help='give up waiting for the lock after this long')
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hot, cold, pack = Path(a.hot), Path(a.cold), Path(a.pack)
    pack.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.minutes * 60
    rows = []
    full_blocks = {b for b in a.full_blocks.split(';') if b}
    for spec in a.blocks.split(';'):
        if '-' in spec:
            lo, hi = (int(x) for x in spec.split('-'))
        else:
            lo = hi = int(spec)
        layers = list(range(lo, hi + 1))
        saved = len(layers) * a.est_gb_per_layer
        print(f'block {spec}: {len(layers)} layers, {saved:.2f} GB', flush=True)
        if pack.exists():
            pack.unlink()
        if not build(hot, cold, layers, pack, a.profile):
            print('  build failed', flush=True)
            continue
        row = {'block': spec, 'layers': layers, 'gb_saved': round(saved, 2)}
        o = gate(pack, out_dir / f'block_{spec}', a.prompts, 780, deadline)
        if o:
            row['subset_top1'] = round(100.0 * o['top1_agreement'], 2)
            row['subset_kl64'] = round(o['kl64_mean'], 3)
            row['subset_positions'] = o['positions']
        if spec in full_blocks and a.full_prompts:
            o2 = gate(pack, out_dir / f'block_{spec}_full', a.full_prompts, 1150, deadline)
            if o2:
                row['full_top1'] = round(100.0 * o2['top1_agreement'], 2)
                row['full_kl64'] = round(o2['kl64_mean'], 3)
        rows.append(row)
        (out_dir / 'sweep.json').write_text(json.dumps(rows, indent=1))
        # an aborted gate can still hold the file open; the next block's build overwrites it anyway
        for _ in range(30):
            try:
                if pack.exists():
                    pack.unlink()
                break
            except PermissionError:
                time.sleep(2)
        print('  ' + json.dumps(row), flush=True)
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
