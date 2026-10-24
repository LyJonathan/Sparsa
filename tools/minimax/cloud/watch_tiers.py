"""Measure each tier as its upload finishes, in order, and stop at the first failure.

    python3 tools/minimax/cloud/watch_tiers.py --tiers IQ2_XXS,IQ1_S,IQ2_KT --work /work

Polls the HF `SHA256SUMS` every --interval seconds; a tier is ready only when all 13 of its lines exist
(a partially uploaded tier has fewer). Tiers that already have `<work>/out/<TIER>/results.json` are
skipped, so restarting the watcher is safe. The first tier whose G3 or serve smoke fails ends the run:
the parts stay on disk and the log says which stage failed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = "LyJonathan/MiniMax-M3.1-Flash-GGUF"
HERE = os.path.dirname(os.path.abspath(__file__))


def ready_tiers():
    txt = urllib.request.urlopen("https://huggingface.co/%s/resolve/main/SHA256SUMS" % REPO, timeout=60).read().decode()
    counts = {}
    for line in txt.splitlines():
        for frag in line.split():
            if frag.startswith("MiniMax-M3.1-Flash-") and "/" in frag:
                tier = frag.split("MiniMax-M3.1-Flash-", 1)[1].split("/", 1)[0]
                counts[tier] = counts.get(tier, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--work", default="/work")
    ap.add_argument("--exe", default="/work/build/strata-minimax")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--max-wait-hours", type=float, default=8.0)
    args = ap.parse_args()

    todo = [t.strip() for t in args.tiers.split(",") if t.strip()]
    deadline = time.time() + args.max_wait_hours * 3600
    while todo and time.time() < deadline:
        try:
            counts = ready_tiers()
        except Exception as e:
            print("poll failed: %r" % e, flush=True)
            counts = {}
        print("%s ready: %s" % (time.strftime("%H:%M:%SZ", time.gmtime()),
                               ", ".join("%s=%d" % (t, counts.get(t, 0)) for t in todo)), flush=True)
        for tier in list(todo):
            if counts.get(tier, 0) < 13:
                continue
            res = os.path.join(args.work, "out", tier, "results.json")
            if os.path.exists(res):
                print("%s already measured (%s); skipping" % (tier, res), flush=True)
                todo.remove(tier)
                continue
            print("=== running %s" % tier, flush=True)
            p = subprocess.run([sys.executable, os.path.join(HERE, "run_tier_g3.py"),
                                "--tier", tier, "--work", args.work, "--exe", args.exe],
                               capture_output=True, text=True)
            print(p.stdout, flush=True)
            if p.stderr:
                print("STDERR: " + p.stderr[-4000:], flush=True)
            if p.returncode != 0:
                print("STOP: %s failed rc=%d - not continuing to the next tier" % (tier, p.returncode), flush=True)
                return 1
            todo.remove(tier)
        if todo:
            time.sleep(args.interval)
    print("WATCH_DONE remaining=%s" % ",".join(todo), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
