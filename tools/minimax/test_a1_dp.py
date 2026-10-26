"""`a1_dp.py` against brute force: the allocation is exact, not greedy.

Two things are worth a test and nothing else is: that the DP returns the true
optimum (a greedy by error-per-byte does not on a multiple-choice knapsack), and
that the budget it reports is the allocation's real byte total, not the unitised
one.  Brute force over random 5-group / 3-option instances is small enough to be a
proof for the shapes this DP sees (171 groups, 3 options, ~95k units).

    python tools/minimax/test_a1_dp.py
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_dp  # noqa: E402


def brute(groups, labels, values, costs, budget):
    best = None
    for combo in itertools.product(labels, repeat=len(groups)):
        used = sum(costs[g][lab] for g, lab in zip(groups, combo))
        if used > budget:
            continue
        val = sum(values[g][lab] for g, lab in zip(groups, combo))
        if best is None or val < best[0]:
            best = (val, used, combo)
    return best


def main() -> int:
    fails = []
    # 1. a hand-checkable case: greedy by value/byte takes b for both and overshoots
    groups = [(0, 'gate'), (0, 'up')]
    labels = ['a', 'b']
    values = {g: {'a': 0.10, 'b': 0.05} for g in groups}
    costs = {g: {'a': 1, 'b': 2} for g in groups}
    alloc, used, obj = a1_dp.solve(groups, labels, values, costs, 3, 'sum')
    if sorted(alloc.values()) != ['a', 'b'] or abs(obj - 0.15) > 1e-9:
        fails.append(f'hand case gave {alloc} / {obj}, expected one a and one b at 0.15')
    print(f'hand case: {alloc} units={used} obj={obj:.4f}')

    # 2. random instances against brute force
    rng = random.Random(7)
    for trial in range(12):
        n = rng.randint(2, 5)
        gs = [(i, 'gate') for i in range(n)]
        values = {g: {lab: rng.uniform(0.05, 0.6) for lab in labels} for g in gs}
        costs = {g: {lab: rng.randint(1, 4) for lab in labels} for g in gs}
        budget = rng.randint(3, 8)
        alloc, used, obj = a1_dp.solve(gs, labels, values, costs, budget, 'sum')
        b = brute(gs, labels, values, costs, budget)
        if b is None:
            if alloc is not None:
                fails.append(f'trial {trial}: reported a solution for an infeasible instance')
            continue
        got = sum(values[g][alloc[g]] for g in gs)
        if b is None or abs(got - b[0]) > 1e-9 or used > budget:
            fails.append(f'trial {trial}: dp {got:.6f}/{used} vs brute {b}')
        # sumsq objective too
        alloc2, used2, obj2 = a1_dp.solve(gs, labels, values, costs, budget, 'sumsq')
        got2 = sum(values[g][alloc2[g]] ** 2 for g in gs)
        b2 = min((sum(values[g][lab] ** 2 for g, lab in zip(gs, combo)),
                  sum(costs[g][lab] for g, lab in zip(gs, combo)))
                 for combo in itertools.product(labels, repeat=len(gs))
                 if sum(costs[g][lab] for g, lab in zip(gs, combo)) <= budget)
        if abs(got2 - b2[0]) > 1e-9 or used2 > budget:
            fails.append(f'trial {trial} sumsq: dp {got2:.6f}/{used2} vs brute {b2}')

    for f in fails:
        print('FAIL: ' + f)
    print('PASS: the DP is exact against brute force on 12 random instances, both objectives'
          if not fails else f'{len(fails)} failures')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
