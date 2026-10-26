"""Alignment and scoring tests for g3_sparsa.py - no engine, no GPU.

The gate is only as good as its position mapping, so this pins it down three ways:

1. *Self-consistency*: feeding the reference logits back in as the "quant" logits must give
   KL64 0, full-vocab KL 0, top-1 agreement 100% and the reference's own NLL.  Any off-by-one,
   wrong slice or row-order bug shows up as a nonzero KL here.
2. *Shift scan*: scoring the reference against ids shifted by -2..+2 must be worst at every
   non-zero shift; the standard "row i predicts token i+1" convention has to win on every prompt.
3. *Known reference-only numbers*: the reference's own nll/top-1 on these prompts are fixed
   values (they are model behaviour on untemplated text, see results_diagnosis.md 3.7).  They are
   asserted per prompt, so a change in how a dump's last-K window is mapped to the prompt fails
   the test instead of quietly changing the gate.

    python tools/minimax/test_g3_sparsa.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g3_sparsa as G  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASE = os.path.join(os.path.dirname(HERE), "Strata-MiniMax", "_local")

# nll_true_ref / top1_acc_vs_true_ref of the fp32 reference on each prompt, as scored by this
# harness and independently by _local/g3_iq2xs/tools/compare.py (g3_results.json).
REF_NLL = {
    "chinese_local_llm": 6.441547,
    "code_lru": 11.569326,
    "long_code_stdlib": 8.707076,
    "long_prose_austen": 5.915379,
    "math_bakery": 3.518818,
    "multi_turn_threads": 6.370866,
    "prose_moe": 5.143090,
    "tool_weather": 5.835807,
}
REF_TOP1 = {
    "chinese_local_llm": 0.208333,
    "code_lru": 0.061947,
    "long_code_stdlib": 0.165354,
    "long_prose_austen": 0.181102,
    "math_bakery": 0.370079,
    "multi_turn_threads": 0.275591,
    "prose_moe": 0.240964,
    "tool_weather": 0.464567,
}


def dataset(base):
    ref = os.path.join(base, "ref_out", "logits")
    ids = os.path.join(base, "g3_iq2xs", "ids")
    out = {}
    for name in sorted(REF_NLL):
        x = np.load(os.path.join(ref, name, "logits_last128.npy"))
        i = np.fromfile(os.path.join(ids, name + ".bin"), dtype=np.int32)
        out[name] = (ref, x, i)
    return out


def test_self_consistent(ref_dir, name, logits, ids):
    r = G.score_prompt(name, ref_dir, logits, ids)
    assert abs(r["kl64_mean"]) < 1e-12, (name, r["kl64_mean"])
    assert abs(r["kl_vocab_mean"]) < 1e-12, (name, r["kl_vocab_mean"])
    assert r["top1_agreement"] == 1.0, (name, r["top1_agreement"])
    assert abs(r["nll_true_ref"] - r["nll_true_quant"]) < 1e-12, name
    assert abs(r["nll_true_ref"] - REF_NLL[name]) < 1e-5, (name, r["nll_true_ref"], REF_NLL[name])
    assert abs(r["top1_acc_vs_true_ref"] - REF_TOP1[name]) < 1e-5, (name, r["top1_acc_vs_true_ref"])
    # Feeding the reference back in must also agree with it on every conditioned slice, and be a perfect top-1
    # on the two on-path slices (there the reference's argmax is the true token).  conf99 may include confident
    # reference errors, so only its agreement is 1.0.  This pins the slice masks: a threshold applied to the
    # wrong array would not be 100% here.
    for key, s in r["conditioned"].items():
        if s["n"]:
            assert s["top1_agreement"] == 1.0, (name, key, s)
            if key != "conf99":
                assert s["top1"] == 1.0, (name, key, s)
    return r


def test_conditioned_slices():
    """Synthetic logits: the three slices count and score the rows they say they do."""
    vocab, n = 6, 6
    ids = np.array([100, 100, 1, 2, 3, 4, 5], dtype=np.int64)  # start=1; rows 0..4 predict ids[2..6]
    ref = np.full((n, vocab), -12.0, dtype=np.float64)
    q = np.full((n, vocab), -12.0, dtype=np.float64)

    ref[0, 1] = 12.0; q[0, 1] = 12.0            # on path, p ~ 1.0, quant hits
    ref[1, 2] = 3.2; ref[1, 0] = -1.0; q[1, 2] = 3.2        # on path, p ~ 0.985, quant hits
    ref[2, 3] = 1.001; ref[2, 0] = 1.0; q[2, 0] = 1.001     # on path, p ~ 0.50, quant misses
    ref[3, 0] = 12.0; q[3, 0] = 12.0            # OFF path (true is 4), p ~ 1.0, confident reference error
    ref[4, 0] = 1.001; ref[4, 2] = 1.0; q[4, 2] = 1.001     # off path, p ~ 0.50, quant misses
    q[5, 5] = 1.0                                # last row has no target and must not be counted

    c = G.conditioned_slices(ref, q, ids)
    assert c["on_path"]["n"] == 3, c
    assert abs(c["on_path"]["top1"] - 2 / 3) < 1e-12, c
    assert c["on_path_conf90"]["n"] == 2, c
    assert c["on_path_conf90"]["top1"] == 1.0, c
    assert c["conf99"]["n"] == 2, c
    assert c["conf99"]["top1"] == 0.5, c          # row 3 is a confident reference error, not a quant hit
    assert c["conf99"]["top1_agreement"] == 1.0, c

    # An empty slice is reported as n=0 with no rate, not as a 0% that would look like a quant failure.
    empty = G.conditioned_slices(np.full((2, vocab), -5.0), np.full((2, vocab), -5.0), np.array([1, 2], dtype=np.int64))
    assert all(s["n"] == 0 and np.isnan(s["top1"]) for s in empty.values()), empty
    return c


def test_shift_scan(ref_dir, name, logits, ids):
    """The reference must predict ids[start+i+1]; every other shift is worse."""
    lp = G.log_softmax(np.asarray(logits, dtype=np.float64))
    n, start = logits.shape[0], len(ids) - logits.shape[0]

    def nll(shift):
        idx = start + np.arange(n) + 1 + shift
        ok = (idx >= 0) & (idx < len(ids))
        return float(-lp[np.arange(n)[ok], np.asarray(ids, dtype=np.int64)[idx[ok]]].mean())

    at = {s: nll(s) for s in (-2, -1, 0, 1, 2)}
    assert at[0] == min(at.values()), (name, at)
    return at


def main():
    base = os.environ.get("G3_BASE", DEFAULT_BASE)
    ref = os.path.join(base, "ref_out", "logits")
    if not os.path.isdir(ref):
        raise SystemExit("reference not found: %s (set G3_BASE)" % ref)
    data = dataset(base)
    c = test_conditioned_slices()
    print("conditioned slices on synthetic logits: on-path n=%d top1 %.2f%%, on-path p>=.90 n=%d top1 %.2f%%, "
          "p>=.99 n=%d top1 %.2f%%" % (c["on_path"]["n"], 100 * c["on_path"]["top1"],
                                       c["on_path_conf90"]["n"], 100 * c["on_path_conf90"]["top1"],
                                       c["conf99"]["n"], 100 * c["conf99"]["top1"]))
    v = os.path.join(base, "ref_out", "logits", "chinese_local_llm", "logits_last128.npy")
    del v
    for name, (ref_dir, logits, ids) in data.items():
        r = test_self_consistent(ref_dir, name, logits, ids)
        at = test_shift_scan(ref_dir, name, logits, ids)
        print("%-20s T=%-4d rows=%-3d ref nll %.4f  top1 %.2f%%  shift-scan nll %s"
              % (name, len(ids), logits.shape[0], r["nll_true_ref"], 100 * r["top1_acc_vs_true_ref"],
                 " ".join("%+d:%.3f" % (s, at[s]) for s in sorted(at))))
    print("PASS: %d prompts, KL 0 and top-1 100%% on the reference itself, ids[i+1] is the best shift"
          % len(data))


if __name__ == "__main__":
    main()
