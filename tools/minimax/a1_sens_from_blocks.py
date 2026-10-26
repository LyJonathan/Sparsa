"""Turn measured G3 block deltas into a per-tensor sensitivity table for `a1_dp.py`.

Why not use the per-tensor error directly: `pack_sensitivity.py` ranks *bytes*, and on this model its
relative-Frobenius column does **not** rank blocks the way G3 does - over the three blocks measured with all six
prompts the predicted spread is 5% (sum drelF 2.552 / 2.423 / 2.440) while the measured KL64 deltas span -0.094 to
+0.121 (README, "The proxy does not rank blocks").  So the block *level* has to come from G3, and the per-tensor
error is used only for what it does describe: how much of a block's cost belongs to gate, up and down (down's
relF rises 1.82x against gate/up's 1.52x, consistently on every layer).

The mapping, for a group g in block B with option o:

    value[g, o] = delta_B * relF[g, o] / mean_over_B(relF[., o])

so a block entirely at option o costs exactly its measured delta_B, and a mixed block is a linear interpolation
(the standard RCO approximation, and the reason the candidate is measured whole).  Blocks that were not measured
are interpolated linearly in depth between their measured neighbours and held constant outside the range; that is
stated as an assumption, not a measurement.  A block that measured *better* than D3 is clamped to 0 for the
objective (a gain is not a licence to spend it twice), which the README records.

    python tools/minimax/a1_sens_from_blocks.py --sens d3=sens_d3_iq1m.json --sens iq1m=... --sens iq2xxs=... \
        --blocks blocks.json --out dp_input_kl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROLES = ('gate', 'up', 'down')


def load_sens(specs: list[str]) -> tuple[dict, list[str], dict]:
    """{(layer, role): {label: {'relf','bytes','type'}}}, labels, and {(layer): bytes per label}"""
    table: dict[tuple[int, str], dict] = {}
    labels: list[str] = []
    for spec in specs:
        label, _, path = spec.partition('=')
        labels.append(label)
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        for r in d['rows']:
            p = r['packs'].get(label)
            if not p or p.get('relf') is None:
                continue
            table.setdefault((int(r['layer']), r['role']), {})[label] = {
                'relf': float(p['relf']), 'bytes': int(p['bytes']), 'type': str(p['type'])}
    return table, labels, {}


def block_deltas(blocks: list[dict], layers: list[int]) -> dict[int, float]:
    """Layer -> delta, by linear interpolation in depth between measured blocks."""
    pts = sorted((b['layers'][0], b['layers'][1], b['delta']) for b in blocks)
    out = {}
    for L in layers:
        if L < pts[0][0]:
            out[L] = pts[0][2]
            continue
        if L > pts[-1][1]:
            out[L] = pts[-1][2]
            continue
        for lo, hi, dv in pts:
            if lo <= L <= hi:
                out[L] = dv
                break
        else:
            # between two measured blocks: interpolate on the layer index
            for (lo1, hi1, d1), (lo2, hi2, d2) in zip(pts, pts[1:]):
                if hi1 < L < lo2:
                    t = (L - hi1) / (lo2 - hi1)
                    out[L] = d1 + t * (d2 - d1)
                    break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', action='append', required=True, help='label=json from pack_sensitivity.py')
    ap.add_argument('--blocks', required=True, help='JSON: [{"block": "3-7", "delta": -0.0939}, ...] '
                                                    '(delta = the metric named by --metric)')
    ap.add_argument('--metric', default='kl64', help='only a label; --blocks carries one delta per block')
    ap.add_argument('--clamp', type=float, default=0.0, help='floor for a block delta (a gain is not spendable)')
    ap.add_argument('--eff', action='append', default=[],
                    help='label=KL-per-GB measured for that type (e.g. iq2xxs=0.0244); with --ref-label this '
                         'prices a byte saved by one type differently from a byte saved by another')
    ap.add_argument('--ref-label', default=None,
                    help='the pack the block deltas were measured with; an all-ref block costs exactly delta_B')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    table, labels, _ = load_sens(args.sens)
    eff = {}
    for spec in args.eff:
        k, _, v = spec.partition('=')
        eff[k] = float(v)
    ref_label = args.ref_label
    layers = sorted({L for L, _ in table})
    blocks = []
    for b in json.loads(Path(args.blocks).read_text(encoding='utf-8')):
        lo, hi = (int(x) for x in b['block'].split('-'))
        blocks.append({'layers': [lo, hi], 'delta': max(args.clamp, float(b['delta']))})
    per_layer = block_deltas(blocks, layers)
    print('layer deltas:', {L: round(per_layer[L], 4) for L in layers if L % 8 == 3})

    rows = []
    for L in layers:
        for role in ROLES:
            g = (L, role)
            opts = table[g]
            base = opts['d3']['relf']
            vals = {}
            for lab in labels:
                if lab == 'd3':
                    vals[lab] = 0.0
                    continue
                # the block's measured cost, split by each option's *incremental* error over D3 in that block,
                # normalised so that a block entirely at this option sums to exactly delta_B.  (Normalising by the
                # option's own relF instead would make every option look alike and leave the DP choosing on bytes.)
                blk_lo, blk_hi = next((b['layers'] for b in blocks if b['layers'][0] <= L <= b['layers'][1]),
                                      (L, L))
                peers = [table[(L2, r2)][lab]['relf'] - table[(L2, r2)]['d3']['relf']
                         for L2 in range(blk_lo, blk_hi + 1) for r2 in ROLES
                         if (L2, r2) in table and lab in table[(L2, r2)]]
                mean = sum(peers) / len(peers) if peers else 1.0
                vals[lab] = round(per_layer[L] * (opts[lab]['relf'] - base) / mean, 6) if mean else 0.0
            if eff and ref_label:
                # price a saved byte by the type's measured KL cost per GB, so the DP is not free to prefer the
                # type that saves the most bytes at equal modelled error (that is what the relF split does, and it
                # is how an earlier run of this DP came out 146/171 at IQ1_M)
                blk_lo, blk_hi = next((b['layers'] for b in blocks if b['layers'][0] <= L <= b['layers'][1]),
                                      (L, L))
                for lab in labels:
                    if lab == 'd3':
                        vals[lab] = 0.0
                        continue
                    save = opts['d3']['bytes'] - opts[lab]['bytes']
                    ref_saves = [table[(L2, r2)]['d3']['bytes'] - table[(L2, r2)][ref_label]['bytes']
                                 for L2 in range(blk_lo, blk_hi + 1) for r2 in ROLES
                                 if (L2, r2) in table and ref_label in table[(L2, r2)]]
                    ref_mean = (sum(ref_saves) / len(ref_saves)) if ref_saves else 1
                    vals[lab] = round(per_layer[L] * (eff[lab] * save) / (eff[ref_label] * ref_mean), 6) if ref_mean else 0.0
            rows.append({'layer': L, 'role': role,
                         'packs': {lab: {'relf': vals[lab], 'bytes': opts[lab]['bytes'],
                                         'type': opts[lab]['type']} for lab in labels}})
    out = {'source': 'a1_sens_from_blocks', 'metric': args.metric, 'blocks': blocks,
           'labels': labels, 'rows': rows}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'wrote {args.out}: {len(rows)} groups')
    return 0


if __name__ == '__main__':
    sys.exit(main())
