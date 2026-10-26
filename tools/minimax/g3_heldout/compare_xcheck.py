"""Cross-check: the box's fresh fp32 reference vs the original G3 reference.

Same GPU type, same reference code path, same precision, so the two should agree
to within float non-determinism of the same kernels.  Reports, per prompt:

    max abs diff, mean abs diff, exact-match fraction
    KL64 mean   (refs top-64, both renormalised - the Strata yardstick)
    KLvocab mean (full 200,064-way KL)
    top-1 agreement

Usage:
    python compare_xcheck.py --old <name>=<npy> ... --new <out_dir>
"""

from __future__ import annotations

import argparse
import json

import numpy as np

TOPK = 64


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def metrics(ref, new):
    lp = log_softmax(ref)
    lq = log_softmax(new)
    p = np.exp(lp)
    kl_vocab = np.sum(p * (lp - lq), axis=-1)

    order = np.argsort(-ref, axis=-1)[:, :TOPK]
    lpr = log_softmax(np.take_along_axis(ref, order, 1))
    lpq = log_softmax(np.take_along_axis(new, order, 1))
    kl64 = np.sum(np.exp(lpr) * (lpr - lpq), axis=-1)

    d = np.abs(ref - new)
    return {
        "positions": int(ref.shape[0]),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "exact_match_frac": float((ref == new).mean()),
        "kl64_mean": float(kl64.mean()),
        "kl64_max": float(kl64.max()),
        "kl_vocab_mean": float(kl_vocab.mean()),
        "kl_vocab_max": float(kl_vocab.max()),
        "top1_agreement": float((ref.argmax(-1) == new.argmax(-1)).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", action="append", required=True, help="name=path to the original npy")
    ap.add_argument("--new", required=True, help="directory of the new run (<name>/logits_last128.npy)")
    ap.add_argument("--out", default=None, help="write the JSON here")
    args = ap.parse_args()

    rows = []
    for spec in args.old:
        name, path = spec.split("=", 1)
        ref = np.load(path).astype(np.float64)
        new = np.load(f"{args.new}/{name}/logits_last128.npy").astype(np.float64)
        if ref.shape != new.shape:
            raise SystemExit(f"{name}: shape {new.shape} != {ref.shape}")
        r = {"prompt": name, **metrics(ref, new)}
        rows.append(r)
        print(f"{name:16s} pos {r['positions']:4d}  maxabs {r['max_abs_diff']:.3e}  "
              f"meanabs {r['mean_abs_diff']:.3e}  exact {100 * r['exact_match_frac']:.1f}%  "
              f"KL64 {r['kl64_mean']:.3e}  KLvocab {r['kl_vocab_mean']:.3e}  "
              f"top1 {100 * r['top1_agreement']:.2f}%", flush=True)

    overall = {k: float(np.mean([r[k] for r in rows]))
               for k in ("max_abs_diff", "mean_abs_diff", "exact_match_frac", "kl64_mean",
                         "kl_vocab_mean", "top1_agreement")}
    overall["positions"] = int(sum(r["positions"] for r in rows))
    print(f"{'OVERALL':16s} pos {overall['positions']:4d}  maxabs {overall['max_abs_diff']:.3e}  "
          f"meanabs {overall['mean_abs_diff']:.3e}  exact {100 * overall['exact_match_frac']:.1f}%  "
          f"KL64 {overall['kl64_mean']:.3e}  KLvocab {overall['kl_vocab_mean']:.3e}  "
          f"top1 {100 * overall['top1_agreement']:.2f}%")
    out = {"per_prompt": rows, "overall": overall}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
