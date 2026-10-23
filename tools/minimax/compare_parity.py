#!/usr/bin/env python3
"""Compare the engine's per-layer probes with the FP32 reference dumps (P4 / gate G1).

`tools/minimax/engine` writes its probes with `--minimax-dump-dir DIR` (optionally restricted with
`--minimax-dump-layers 0,3,40`).  The reference (`tools/minimax/ref`, see `_local/ref_out/dumps/`) writes the
same tensors, one directory per layer.  This script lines them up and prints max-abs, relative error and
cosine per stage, the router top-k id agreement, and the per-expert output error.

    python tools/minimax/compare_parity.py --ref _local/ref_out/dumps/short --ours _probe/short --layers 0,3,40
"""

import argparse
import os
import numpy as np

# Reference stores the per-head tensors as [1, heads, T, dim]; the engine stores [T, heads*dim].
HEAD_MAJOR = {"q_after_rope", "k_after_rope", "index_q_after_rope", "index_k_after_rope",
              "index_block_scores", "index_block_topk_scores", "index_block_indices"}
TOKEN_MAJOR = {"attn_core_out"}


def load(path):
    a = np.load(path)
    return a


def norm_ref(name, a):
    if name in HEAD_MAJOR:
        a = np.squeeze(a, axis=0)          # [H, T, D]
        a = np.transpose(a, (1, 0, 2))     # [T, H, D]
    elif name in TOKEN_MAJOR:
        a = np.squeeze(a, axis=0)          # [T, H, D]
    else:
        a = np.squeeze(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1).astype(np.float64)


def stats(ref, ours):
    ref = np.asarray(ref, dtype=np.float64).reshape(ref.shape[0], -1)
    ours = np.asarray(ours, dtype=np.float64).reshape(ours.shape[0], -1)
    fin = np.isfinite(ref) & np.isfinite(ours)
    both_inf = (~np.isfinite(ref)) & (~np.isfinite(ours)) & (np.sign(ref) == np.sign(ours))
    d = np.where(fin, np.abs(ref - ours), 0.0)
    scale = np.abs(ref[fin]).max() if fin.any() else 0.0
    rf = np.where(fin, ref, 0.0)
    of = np.where(fin, ours, 0.0)
    denom = np.linalg.norm(rf) * np.linalg.norm(of)
    cos = float((rf * of).sum() / denom) if denom > 0 else 0.0
    inf_mismatch = int((~fin & ~both_inf).sum())
    return dict(rows=ref.shape[0], cols=ref.shape[1], max_abs=float(d.max()),
                rel=float(d.max() / scale) if scale > 0 else 0.0, cos=cos,
                inf_mismatch=inf_mismatch)


def dump_dirs(root):
    """{source layer number: path} for `dump-L03` style reference dirs."""
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("dump-L"):
            out[int(name[len("dump-L"):])] = p
    return out


def find_ours(root, layer):
    for name in (f"dump-L{layer}", f"dump-L{layer:02d}", f"dump-L{layer:03d}"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference dumps root, e.g. _local/ref_out/dumps/short")
    ap.add_argument("--ours", required=True, help="engine --minimax-dump-dir root")
    ap.add_argument("--layers", default="", help="comma list; default: every layer both sides have")
    ap.add_argument("--json", default="", help="also write the table as json here")
    args = ap.parse_args()

    refs = dump_dirs(args.ref)
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else sorted(refs)
    stages = ["attn_input", "q_after_rope", "k_after_rope", "index_q_after_rope", "index_k_after_rope",
              "attn_core_out", "attn_out", "router_logits", "router_weights", "shared_expert_out",
              "moe_out", "layer_out", "index_block_scores"]
    report = {}
    print("| layer | stage | rows | max abs | rel | cosine |")
    print("|---|---|---|---|---|---|")
    for L in layers:
        rp = refs.get(L)
        op = find_ours(args.ours, L)
        if not rp or not op:
            print(f"| {L} | (missing dumps) | | | | |")
            continue
        entry = {}
        for st in stages:
            rpath = os.path.join(rp, st + ".npy")
            opath = os.path.join(op, st + ".npy")
            if not os.path.exists(rpath) or not os.path.exists(opath):
                continue
            ref = norm_ref(st, load(rpath))
            ours = load(opath).astype(np.float64)
            if ours.shape != ref.shape and st in HEAD_MAJOR and ours.shape[0] == ref.shape[0]:
                # the block-score probes pad to the sequence's block count, and the two sides may pad to a
                # different maximum (the engine pads to the eval length, the reference to the prompt length)
                heads = 4 if ref.shape[1] % 4 == 0 else 1
                nb_ref = ref.shape[1] // heads
                ob = ours.reshape(ours.shape[0], -1, ours.shape[1] // heads)
                if ob.shape[1] == heads:
                    ours = ob[:, :, :nb_ref].reshape(ours.shape[0], -1)
            if ours.shape[0] != ref.shape[0]:
                print(f"| {L} | {st} | row mismatch ref {ref.shape[0]} ours {ours.shape[0]} | | | |")
                continue
            s = stats(ref, ours)
            entry[st] = s
            print(f"| {L} | {st} | {s['rows']} | {s['max_abs']:.3e} | {s['rel']:.3e} | {s['cos']:.6f} |")

        # MSA block selection: the selected block ids of every (token, indexer head), -1 padded on both sides.
        rp_bi = os.path.join(rp, "index_block_indices.npy")
        op_bi = os.path.join(op, "index_block_indices.npy")
        if os.path.exists(rp_bi) and os.path.exists(op_bi):
            a = np.squeeze(load(rp_bi), axis=0).astype(np.int64)      # [HI, T, k]
            b = load(op_bi).astype(np.int64)                          # [T, HI*k]
            k = a.shape[2]
            b = b.reshape(b.shape[0], -1, k)
            T = min(a.shape[1], b.shape[0])
            same = inter = union = 0
            for t in range(T):
                for h in range(a.shape[0]):
                    sa = set(int(x) for x in a[h, t] if x >= 0)
                    sb = set(int(x) for x in b[t, h] if x >= 0)
                    same += 1 if sa == sb else 0
                    inter += len(sa & sb)
                    union += len(sa | sb)
            n = T * a.shape[0]
            entry["blocks"] = dict(rows=T, sets_equal=same / n, jaccard=inter / union if union else 1.0)
            print(f"| {L} | index_block_indices | {T} | sets {same / n:.4f} | jaccard {inter / union if union else 1.0:.4f} | |")

        # the routed+shared MoE output: the reference has no single dump for it, but sum(expert_out)*scale + shared
        # is exactly what the layer adds. This is the aggregate "expert out" number.
        om = os.path.join(op, "moe_out.npy")
        rs = os.path.join(rp, "shared_expert_out.npy")
        if os.path.exists(om) and os.path.exists(rs):
            shared = np.asarray(load(rs), dtype=np.float64).reshape(-1, 6144)
            routed = np.zeros_like(shared)
            for name in sorted(os.listdir(rp)):
                if name.startswith("expert_") and name.endswith("_out.npy"):
                    e = name[len("expert_"):-len("_out.npy")]
                    out = np.asarray(load(os.path.join(rp, name)), dtype=np.float64)
                    tok = load(os.path.join(rp, f"expert_{e}_token_idx.npy")).reshape(-1).astype(np.int64)
                    routed[tok] += out
            ref_moe = 2.0 * routed + shared
            ours = load(om).astype(np.float64)
            s = stats(ref_moe, ours)
            entry["moe_out"] = s
            print(f"| {L} | moe_out (2*routed+shared) | {s['rows']} | {s['max_abs']:.3e} | {s['rel']:.3e} | "
                  f"{s['cos']:.6f} |")
            # routing differences dominate the aggregate, so report the same number on the rows where both sides
            # chose the same expert set.
            ri = os.path.join(rp, "router_topk_ids.npy")
            oi = os.path.join(op, "router_topk_ids.npy")
            if os.path.exists(ri) and os.path.exists(oi):
                a_ids = load(ri).reshape(load(ri).shape[0], -1).astype(np.int64)
                b_ids = load(oi).astype(np.int64)
                n = min(len(a_ids), len(b_ids))
                rows = [i for i in range(n) if set(a_ids[i].tolist()) == set(b_ids[i].tolist())]
                if rows:
                    s2 = stats(ref_moe[rows], ours[rows])
                    entry["moe_out_same_routing"] = s2
                    print(f"| {L} | moe_out, same routing ({len(rows)}/{n} rows) | {s2['rows']} | "
                          f"{s2['max_abs']:.3e} | {s2['rel']:.3e} | {s2['cos']:.6f} |")

        # router top-k ids: exact row-wise set agreement and per-slot agreement
        rp_ids = os.path.join(rp, "router_topk_ids.npy")
        op_ids = os.path.join(op, "router_topk_ids.npy")
        if os.path.exists(rp_ids) and os.path.exists(op_ids):
            a = load(rp_ids).reshape(load(rp_ids).shape[0], -1).astype(np.int64)
            b = load(op_ids).astype(np.int64)
            n = min(len(a), len(b))
            set_eq = sum(1 for i in range(n) if set(a[i].tolist()) == set(b[i].tolist()))
            entry["router_topk"] = dict(rows=n, set_agreement=set_eq / n)
            # torch.topk(sorted=False) returns an arbitrary order, so weights are compared per expert id.
            rp_w = os.path.join(rp, "router_weights.npy")
            op_w = os.path.join(op, "router_weights.npy")
            wmax, wsum = 0.0, 0.0
            if os.path.exists(rp_w) and os.path.exists(op_w):
                wa = load(rp_w).reshape(-1, a.shape[1]).astype(np.float64)
                wb = load(op_w).astype(np.float64)
                for i in range(n):
                    ma = {int(k): wa[i, j] for j, k in enumerate(a[i])}
                    mb = {int(k): wb[i, j] for j, k in enumerate(b[i])}
                    for k in set(ma) & set(mb):
                        wmax = max(wmax, abs(ma[k] - mb[k]))
                    wsum = max(wsum, abs(sum(mb.values()) - sum(ma.values())))
                entry["router_weights"] = dict(max_abs_by_id=wmax, max_abs_sum=wsum)
            print(f"| {L} | router_topk_ids | {n} | set {set_eq / n:.4f} | "
                  f"weights by id {wmax:.3e} | sum diff {wsum:.3e} |")

        # per-expert outputs: the reference stores [n_hits, n_embd] and token_idx [n_hits]
        exp_err = []
        for name in sorted(os.listdir(rp)):
            if not (name.startswith("expert_") and name.endswith("_out.npy")):
                continue
            e = name[len("expert_"):-len("_out.npy")]
            oo = os.path.join(op, f"expert_{e}_out.npy")
            ot = os.path.join(op, f"expert_{e}_token_idx.npy")
            rt = os.path.join(rp, f"expert_{e}_token_idx.npy")
            if not (os.path.exists(oo) and os.path.exists(ot) and os.path.exists(rt)):
                continue
            ref_out = np.asarray(load(os.path.join(rp, name)), dtype=np.float64)
            our_out = np.asarray(load(oo), dtype=np.float64)
            ref_t = load(rt).reshape(-1).astype(np.int64)
            our_t = load(ot).reshape(-1).astype(np.int64)
            same = ref_t.shape == our_t.shape and bool((ref_t == our_t).all())
            if not same:
                exp_err.append((e, None, None, False))
                continue
            s = stats(ref_out, our_out)
            exp_err.append((e, s["max_abs"], s["rel"], True))
        ok = [x for x in exp_err if x[3] and x[1] is not None]
        if exp_err:
            bad = [x[0] for x in exp_err if not x[3]]
            worst = max(ok, key=lambda x: x[1]) if ok else None
            entry["experts"] = dict(count=len(exp_err), token_idx_mismatch=bad,
                                    compared=len(ok),
                                    worst_expert=worst[0] if worst else None,
                                    worst_max_abs=worst[1] if worst else None,
                                    worst_rel=worst[2] if worst else None)
            print(f"| {L} | expert out | {len(ok)}/{len(exp_err)} token-exact"
                  f"{'; idx differs: ' + str(len(bad)) if bad else ''} | "
                  f"{worst[1] if worst else float('nan'):.3e} | {worst[2] if worst else float('nan'):.3e} | |")
        report[L] = entry

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
