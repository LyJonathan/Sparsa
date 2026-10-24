"""Collect the per-tier G3 results into the shapes the report needs.

    python tools/minimax/g3_summary.py --root _local/g3_sparsa

Reads `<root>/<run>/results.json` (written by g3_sparsa.py), copies each to
`<root>/results_<TIER>.json` and prints one aggregate table plus the per-prompt KL64
and top-1 rows.  Numbers only: nothing here recomputes or rounds a metric that the
runner did not measure.
"""
import argparse
import json
import os
import shutil


def load_runs(root):
    runs = []
    for d in sorted(os.listdir(root)):
        j = os.path.join(root, d, "results.json")
        if not os.path.isfile(j):
            continue
        with open(j, encoding="utf-8") as f:
            r = json.load(f)
        if not r.get("overall"):
            continue
        r["_run"] = d
        runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("_local", "g3_sparsa"))
    args = ap.parse_args()
    root = args.root
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no results.json under {root}")

    for r in runs:
        tier = r["tier"].replace("/", "_")
        dst = os.path.join(root, "results_%s.json" % tier)
        shutil.copyfile(os.path.join(root, r["_run"], "results.json"), dst)
        print("wrote", dst)

    print()
    print("| tier | positions | KL64 | KL64 p99 | top-1 | ppl | ppl/39.69 | engine s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        o = r["overall"]
        p = r.get("ppl") or {}
        ppl = "%.2f" % p["ppl"] if p else "-"
        ratio = "%.2f" % p["ppl_ratio"] if p else "-"
        print("| %s | %d | %.4f | %.4f | %.2f%% | %s | %s | %.0f |" % (
            r["tier"], o["positions"], o["kl64_mean"], o["kl64_p99"],
            100 * o["top1_agreement"], ppl, ratio, r["engine_seconds"]))

    names = sorted({p["prompt"] for r in runs for p in r["per_prompt"]})
    print()
    print("| prompt | " + " | ".join(r["tier"] for r in runs) + " |")
    print("|---|" + "---:|" * len(runs))
    for n in names:
        cells = []
        for r in runs:
            row = next((p for p in r["per_prompt"] if p["prompt"] == n), None)
            cells.append("%.3f / %.0f%%" % (row["kl64_mean"], 100 * row["top1_agreement"]) if row else "-")
        print("| %s | %s |" % (n, " | ".join(cells)))
    print("\n(cell = KL64 mean / top-1 agreement)")

    for r in runs:
        v = r.get("validation")
        if v:
            print("\n## %s vs the llama.cpp harness" % r["tier"])
            print("| metric | llama.cpp | Sparsa | delta |")
            print("|---|---:|---:|---:|")
            for row in v["rows"]:
                print("| %s | %s | %s | %s |" % (row["metric"], row["llamacpp"], row["sparsa"], row["delta"]))


if __name__ == "__main__":
    main()
