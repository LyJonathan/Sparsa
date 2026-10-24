"""Build a MiniMax STRP expert-usage profile from routing counts.

The engine writes the same STRP file directly with `--minimax-profile-save`, and a text table with
`--minimax-routing-counts`.  This tool merges several of those text tables (or hand-written counts) into
one profile, so a trace from one workload can be scored on another.

Text format: one `layer expert count` triple per line.  `layer` is the MoE-layer index (0 is the first
layer after the dense prefix, so layer 3 of M3.1 is 0 here), `expert` is 0..n_expert-1, and `count` is an
integer or float.  `#` starts a comment.

STRP is the format in include/strata/core/expert_cache.hpp and tools/make_profile.py:
`STRP`, then `version, n_layers, n_expert, slots, n_ranked` (uint32 little-endian), the ranked
`(layer, expert)` pairs (uint16 each, hottest first), then an `n_layers x n_expert` int32 table of each
pair's rank (-1 when unranked).  MiniMax uses one row per MoE layer (57 x 128 for M3.1), not per model
layer: the three leading dense layers carry no routed experts.

    python tools/minimax/make_profile.py --n-moe-layers 1 --n-expert 8 --out p.bin counts.txt
    python tools/minimax/make_profile.py --n-moe-layers 57 --n-expert 128 --out p.bin a.txt b.txt
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

MAGIC = b"STRP"
VERSION = 1


def read_counts(path, n_moe, n_expert):
    freq = {}
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 3:
            raise SystemExit(f"{path}:{lineno}: expected 'layer expert count'")
        layer, expert, count = int(parts[0]), int(parts[1]), float(parts[2])
        if not (0 <= layer < n_moe and 0 <= expert < n_expert):
            raise SystemExit(f"{path}:{lineno}: ({layer}, {expert}) is outside {n_moe}x{n_expert}")
        freq[(layer, expert)] = freq.get((layer, expert), 0.0) + count
    return freq


def rank_profile(freq, n_moe, n_expert):
    pairs = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked = [p for p, c in pairs if c > 0]
    seen = set(ranked)
    for e in range(n_expert):
        for layer in range(n_moe):
            if (layer, e) not in seen:
                ranked.append((layer, e))
    return ranked


def write_profile(path, ranked, n_moe, n_expert):
    table = [[-1] * n_expert for _ in range(n_moe)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_moe, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_moe):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def read_profile(path):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not an STRP profile")
    version, n_moe, n_expert, slots, n_ranked = struct.unpack_from("<5I", blob, 4)
    if version != VERSION:
        raise SystemExit(f"{path}: STRP version {version}, this tool writes {VERSION}")
    ranked = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n_ranked)]
    return n_moe, n_expert, ranked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("counts", nargs="*", type=Path, help="text routing-count tables")
    ap.add_argument("--n-moe-layers", type=int, required=True, help="routed-expert layers (57 for M3.1)")
    ap.add_argument("--n-expert", type=int, required=True, help="experts per layer (128 for M3.1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="read the profile back and compare its ranking")
    a = ap.parse_args(argv)
    freq = {}
    for p in a.counts:
        for k, v in read_counts(p, a.n_moe_layers, a.n_expert).items():
            freq[k] = freq.get(k, 0.0) + v
    ranked = rank_profile(freq, a.n_moe_layers, a.n_expert)
    write_profile(a.out, ranked, a.n_moe_layers, a.n_expert)
    if a.check:
        n_moe, n_expert, got = read_profile(a.out)
        assert (n_moe, n_expert) == (a.n_moe_layers, a.n_expert)
        assert got == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs over {a.n_moe_layers}x{a.n_expert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
