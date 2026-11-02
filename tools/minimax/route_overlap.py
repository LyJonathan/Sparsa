#!/usr/bin/env python3
"""How much expert reads a verify window shares, from a STRATA_MM_ROUTE_TRACE file.

The trace has one line per decoded token and MoE layer: "pos layer e0 e1 ...".  A verify step over W consecutive
tokens reads the union of their routed experts once per layer, where W single-token decodes read each token's set
separately.  R(W) is the mean over all windows of (sum over layers of |union of the W sets|) divided by the mean
single-token sum (sum over layers of |set|).  R(1) is 1.  R(W) is the factor by which the expert-read work of a
W-token verify grows over one decode token, when every routed expert is read on the same lane.
"""
import argparse
import collections
import sys


def load(path):
    sets = collections.defaultdict(dict)   # pos -> layer -> frozenset(experts)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            pos, layer = int(parts[0]), int(parts[1])
            sets[pos][layer] = frozenset(int(x) for x in parts[2:])
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--w", type=int, nargs="+", default=[2, 3, 4, 5])
    args = ap.parse_args()
    sets = load(args.trace)
    pos_all = sorted(sets)
    if len(pos_all) < 2:
        print("trace has fewer than two tokens", file=sys.stderr)
        return 1
    # Only positions that form a consecutive run with every layer present.
    runs = []
    cur = [pos_all[0]]
    for p in pos_all[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    print("tokens %d, consecutive runs %d (longest %d)" % (len(pos_all), len(runs), max(len(r) for r in runs)))
    single_total = 0
    single_n = 0
    for run in runs:
        for p in run:
            single_total += sum(len(s) for s in sets[p].values())
            single_n += 1
    single_mean = single_total / max(1, single_n)
    print("experts per token (all MoE layers): %.1f" % single_mean)
    for W in args.w:
        union_total = 0
        windows = 0
        for run in runs:
            for k in range(len(run) - W + 1):
                win = run[k:k + W]
                layers = set()
                for p in win:
                    layers.update(sets[p].keys())
                u = 0
                for layer in layers:
                    members = set()
                    for p in win:
                        members.update(sets[p].get(layer, ()))
                    u += len(members)
                union_total += u
                windows += 1
        if windows == 0:
            continue
        mean_union = union_total / windows
        print("W=%d  mean experts read per verify step %.1f  R(W)=%.3f  (over %d windows)"
              % (W, mean_union, mean_union / single_mean, windows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
