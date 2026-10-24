"""Cut an STRP routing profile down to a layer subset, so a profile traced on the whole model can place the
static tiers of a slice (or of a model whose MoE-layer count differs).

The engine refuses a profile whose header does not match the model exactly (`n_moe x n_expert`), which is the
right check at run time; this tool produces the matching file for the `p5a-l0-14` style slices, which carry the
first 15 MoE layers of M3.1.  Per-layer rank order is preserved and packed back into one global hot-to-cold list.

    python tools/minimax/slice_profile.py --in profile.bin --layers 0-14 --out slice.bin
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

MAGIC = b"STRP"


def read(path: Path) -> tuple[int, int, int, list[tuple[int, int]], list[int]]:
    raw = path.read_bytes()
    if raw[:4] != MAGIC:
        raise SystemExit(f'{path}: not an STRP profile')
    ver, n_moe, n_expert, slots, n_ranked = struct.unpack_from('<5I', raw, 4)
    ranked = [struct.unpack_from('<HH', raw, 24 + 4 * i) for i in range(n_ranked)]
    table = list(struct.unpack_from(f'<{n_moe * n_expert}i', raw, 24 + 4 * n_ranked))
    return ver, n_moe, n_expert, ranked, table


def write(path: Path, n_moe: int, n_expert: int, ranked: list[tuple[int, int]]) -> None:
    table = [-1] * (n_moe * n_expert)
    for r, (l, e) in enumerate(ranked):
        table[l * n_expert + e] = r
    out = bytearray(MAGIC)
    out += struct.pack('<5I', 1, n_moe, n_expert, len(ranked), len(ranked))
    for l, e in ranked:
        out += struct.pack('<HH', l, e)
    out += struct.pack(f'<{len(table)}i', *table)
    path.write_bytes(bytes(out))


def parse_layers(spec: str, n_moe: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [l for l in out if not (0 <= l < n_moe)]
    if bad:
        raise SystemExit(f'layers out of range for a {n_moe}-layer profile: {bad}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', required=True, help='"0-14" or "3,7,11" (MoE-layer indices)')
    a = ap.parse_args()
    ver, n_moe, n_expert, ranked, _ = read(Path(a.src))
    if ver != 1:
        raise SystemExit(f'{a.src}: version {ver}')
    keep = parse_layers(a.layers, n_moe)
    remap = {l: i for i, l in enumerate(keep)}
    # the file's ranked list is already global hot-to-cold; filtering it keeps that order, which is what the
    # engine's greedy placement walks (a layer with more route mass appears earlier and gets more slots)
    out = [(remap[l], e) for l, e in ranked if l in remap]
    write(Path(a.out), len(keep), n_expert, out)
    print(f'{a.src}: {n_moe}x{n_expert} -> {len(keep)}x{n_expert}, {len(out)} ranked pairs -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
