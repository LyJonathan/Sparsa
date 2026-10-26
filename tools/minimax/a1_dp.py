"""Exact-budget allocation of expert types over the 171 expert tensors.

RCO's assignment stage, minus the gradient search (research/gsq-rco, sections 3 and
5): a multiple-choice knapsack where each *group* is one expert tensor
(`blk.L.ffn_{gate,up,down}_exps` - 57 layers x 3 projections, the finest granularity
a GGUF can express, since one tensor carries one ggml type) and each *option* is a
published pack's type for it.  The values are the measured per-tensor errors from
`pack_sensitivity.py` (relative Frobenius error of the pack's stored bytes against
the source checkpoint); the costs are the packs' own `n_bytes`.

    minimise   sum_g  value[g, option]           (objective: sum | sumsq)
    subject to sum_g  bytes[g, option] <= budget

solved exactly by the classic DP over the budget in 1 MiB units - exact for the
quantised budget (worst-case slack 171 MiB = 0.17% of 100 GB), not greedy.

The objective is non-decomposable in reality, so `pack_sensitivity.py`'s ranking is
calibrated against measured G3 block deltas before its output is trusted, and the
top candidates are measured whole (README, "A1: the sensitivity table").

    python tools/minimax/a1_dp.py --sens d3=sens_d3_iq1m.json --sens iq1m=... \
        --budget-gb 100 --objective sum --out dp_c1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UNIT = 1 << 20  # 1 MiB


def load(specs: list[str]) -> dict[str, dict[tuple[int, str], dict]]:
    """label -> {(layer, role): {'relf': float, 'bytes': int}}"""
    out = {}
    for spec in specs:
        label, _, path = spec.partition('=')
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        rows = {}
        for r in d['rows']:
            p = r['packs'].get(label)
            if not p or p.get('relf') is None:
                continue
            rows[(int(r['layer']), r['role'])] = {'relf': float(p['relf']), 'bytes': int(p['bytes']),
                                                  'type': str(p['type'])}
        out[label] = rows
        print(f'{label}: {len(rows)} groups from {path}', flush=True)
    return out


def solve(groups, options, values, costs, budget_units, objective):
    """groups: list of keys; options: list of labels; exact DP over budget units."""
    import numpy as np
    INF = np.float32(1e30)
    n = budget_units + 1
    dp = np.full(n, INF, dtype=np.float64)
    dp[0] = 0.0
    choice = np.zeros((len(groups), n), dtype=np.int8)
    for gi, g in enumerate(groups):
        new = np.full(n, INF, dtype=np.float64)
        pick = np.zeros(n, dtype=np.int8)
        for oi, lab in enumerate(options):
            c = costs[g][lab]
            v = values[g][lab]
            if objective == 'sumsq':
                v = v * v
            if c > budget_units:
                continue
            cand = np.full(n, INF, dtype=np.float64)
            cand[c:] = dp[:n - c] + v
            better = cand < new
            new[better] = cand[better]
            pick[better] = oi
        dp, choice[gi] = new, pick
        if gi % 20 == 0:
            print(f'  dp {gi}/{len(groups)}', flush=True)
    best = int(np.argmin(dp))
    if not np.isfinite(dp[best]) or dp[best] >= 1e29:
        # every option for some group costs more than the whole budget
        return None, 0, float('inf')
    # walk the choices back to an allocation
    alloc = {}
    c = best
    for gi in range(len(groups) - 1, -1, -1):
        oi = int(choice[gi][c])
        lab = options[oi]
        alloc[groups[gi]] = lab
        c -= costs[groups[gi]][lab]
    return alloc, best, float(dp[best])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', action='append', required=True, help='label=json from pack_sensitivity.py')
    ap.add_argument('--budget-gb', type=float, required=True)
    ap.add_argument('--objective', choices=('sum', 'sumsq'), default='sum')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    packs = load(args.sens)
    labels = list(packs)
    groups = sorted(set.intersection(*(set(v) for v in packs.values())))
    if not groups:
        raise SystemExit('the packs share no groups')
    for lab, rows in packs.items():
        missing = len(rows) - len(groups)
        if missing:
            print(f'{lab}: {missing} groups excluded (not in every pack)', flush=True)
    values = {g: {lab: packs[lab][g]['relf'] for lab in labels} for g in groups}
    costs = {g: {lab: max(1, packs[lab][g]['bytes'] // UNIT) for lab in labels} for g in groups}
    budget_units = int(args.budget_gb * 1e9 // UNIT)
    print(f'{len(groups)} groups, {len(labels)} options, budget {args.budget_gb} GB '
          f'= {budget_units} units of 1 MiB', flush=True)

    alloc, used, obj = solve(groups, labels, values, costs, budget_units, args.objective)
    if alloc is None:
        raise SystemExit(f'no allocation fits {args.budget_gb} GB: the cheapest option for some group '
                         f'exceeds the budget')
    rows = []
    for g in groups:
        lab = alloc[g]
        rows.append({'layer': g[0], 'role': g[1], 'pack': lab,
                     'type': packs[lab][g]['type'], 'bytes': packs[lab][g]['bytes'],
                     'relf': packs[lab][g]['relf'],
                     'relf_d3': packs['d3'][g]['relf'] if 'd3' in packs else None})
    total_bytes = sum(r['bytes'] for r in rows)
    base_bytes = sum(packs['d3'][g]['bytes'] for g in groups) if 'd3' in packs else None
    pred = sum(r['relf'] for r in rows)
    pred_d3 = sum(r['relf_d3'] for r in rows) if 'd3' in packs else None
    per_pack = {}
    for r in rows:
        per_pack[r['pack']] = per_pack.get(r['pack'], 0) + 1
    out = {'budget_gb': args.budget_gb, 'objective': args.objective, 'labels': labels,
           'groups': len(rows), 'total_bytes': total_bytes, 'total_gb': round(total_bytes / 1e9, 2),
           'base_gb': round(base_bytes / 1e9, 2) if base_bytes else None,
           'saved_gb': round((base_bytes - total_bytes) / 1e9, 2) if base_bytes else None,
           'pred_sum_relf': round(pred, 4),
           'pred_sum_relf_d3': round(pred_d3, 4) if pred_d3 else None,
           'per_pack': per_pack, 'rows': rows}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'{per_pack}  total {out["total_gb"]} GB (D3 {out["base_gb"]} GB, saved {out["saved_gb"]} GB)')
    print(f'predicted sum relF {out["pred_sum_relf"]} vs D3 {out["pred_sum_relf_d3"]}')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
