"""P16: an `--expert-source` layer, forced onto the SSD tier, reads the donor's bytes.

The bug this test exists for: the SSD tier resolved its file by shard index against the **base** pack, so a layer
whose bytes live in an `--expert-source` pack was read from the wrong file and the run died with *"minimax SSD read:
short read"*.  Here a tiny `--memory-limit-mib` puts every expert on the SSD tier and `--minimax-verify-tensors`
asks the engine to report, for every tensor it resolved, the shard and **absolute** file offset it will read - then
`verify_pack.py` streams those ranges and SHA-256s them against the donors.

Two arms, both all-SSD, both without VRAM expert slots:

  * the base pack alone, and
  * the base pack with layers 3-7 taken from the IQ1_M donor.

The second must (a) name the donor's file for `blk.3..7` `_exps` tensors, (b) resolve every tensor to bytes that are
SHA-256-identical to one of the two donors, and (c) not fail the read path.  Skipped when the donor is absent.

    python tools/minimax/test_expert_source.py            (powreshell-free; runs the engine through pc_launch)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(os.environ.get('TEMP', r'C:\Users\admin\AppData\Local\Temp')) / 'p16'
D3 = Path(r'D:\hf-d3\IQ2_XS-D3\MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf')
IQ1 = Path(r'D:\hf-iq1m\MiniMax-M3.1-Flash-IQ1_M\MiniMax-M3.1-Flash-IQ1_M-00001-of-00013.gguf')
EXE = ROOT / 'build' / 'strata-minimax.exe'
LAUNCH = ROOT / 'tools' / 'minimax' / 'pc_launch.py'
BLOCK = (3, 7)


def run(tag: str, extra: list[str], timeout: int = 900) -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    ids = TMP / 'tiny.txt'
    ids.write_text('1 2 3 4 5 6 7 8', encoding='utf-8')
    cmd = [EXE, '--model', str(D3), '--minimax-eval', str(ids), '--minimax-dump-logits', str(TMP / f'{tag}.npy'),
           '--max-context', '64', '--gpus', '0,1', '--minimax-tiers', '--minimax-vram-slots-per-layer', '0',
           # tiny RAM tier: everything lands on the SSD tier, which is the path under test
           '--memory-limit-mib', '600', '--minimax-verify-tensors', '--minimax-verify-exit',
           '--cpu-threads', '8'] + extra
    launch = [sys.executable, str(LAUNCH), '--owner', 'maya', '--est-ram-gb', '25', '--min-free-gb', '10',
              '--max-minutes', '45', '--'] + cmd
    with open(TMP / f'{tag}.log', 'w') as log:
        p = subprocess.run(launch, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    print(f'{tag}: rc {p.returncode}')
    return p.returncode


def main() -> int:
    if not D3.exists():
        print('SKIP: the base pack is not on this machine')
        return 0
    if not IQ1.exists():
        print('SKIP: the IQ1_M donor is not on this machine')
        return 0
    if run('base', []) != 0:
        print('FAIL: the all-SSD base arm did not run')
        return 1
    alt = run('alt', ['--expert-source', f'{BLOCK[0]}-{BLOCK[1]}={IQ1}'])
    if alt != 0:
        print('FAIL: the all-SSD arm with --expert-source did not run (this is the short-read bug)')
        return 1
    # every tensor must hash to one of the two donors, and the block must name the donor's file
    verify = [sys.executable, str(ROOT / 'tools' / 'minimax' / 'verify_pack.py'),
              '--pack', str(D3), '--engine-log', str(TMP / 'base.log'), '--engine-log', str(TMP / 'alt.log'),
              '--label', 'base', '--label', 'alt', '--donor', str(D3), '--donor', str(IQ1)]
    r = subprocess.run(verify, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print('FAIL: verify_pack did not pass on both arms')
        return 1
    donor_leaf = IQ1.name
    named = 0
    for line in (TMP / 'alt.log').read_text(encoding='utf-8', errors='replace').splitlines():
        if 'INFO minimax-verify blk.' not in line or '_exps.weight' not in line:
            continue
        parts = line.split()
        name = parts[3]
        layer = int(name.split('.')[1])
        if BLOCK[0] <= layer <= BLOCK[1] and parts[-1] == donor_leaf:
            named += 1
    print(f'alt: {named} expert tensors of layers {BLOCK[0]}-{BLOCK[1]} resolved from {donor_leaf}')
    if named == 0:
        print('FAIL: no block tensor resolved to the donor file')
        return 1
    print('PASS: the SSD tier read the alternate pack for the block and every tensor matched a donor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
