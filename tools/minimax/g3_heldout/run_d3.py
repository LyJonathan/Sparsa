"""Score a pack on the held-out set, on the shared PC, in resumable phases.

`g3_sparsa.py` already refuses to reuse a partial dump and happily reuses a
complete one, so the only thing a long gate needs on top of it is: take the PC
lock when it is free, give it back when the turn limit hits, and pick up where
it stopped next time.  That is all this does.

Three phases, cheapest first, so a usable aggregate exists early:

    1. the short prompts (everything but the long-context one)
    2. the long-context prompt on its own (it dominates the runtime)
    3. a final pass over all of them: every dump is complete by then, so this
       only recomputes the aggregate over the whole set together

Every phase goes through `pc_launch.py` (the only sanctioned way to start an
engine on this box).  A lock refusal (exit 2) or a RAM refusal (exit 3) is not a
failure: it is "not my turn yet", and it is retried every `--retry` seconds until
`--max-wait` runs out.  Running out of time is also not a failure - completed
dumps stay, and the next invocation continues.

Usage (from the repo root):

    python tools/minimax/g3_heldout/run_d3.py \
        --model 'D:\\hf-d3\\IQ2_XS-D3\\MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf' \
        --tier MM-IQ2_XS-D3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def local_root() -> Path:
    """The `_local` directory holding the reference artefacts.

    `_local/` is git-ignored, so it lives in whichever checkout actually ran the
    jobs - normally the main one.  A worktree gets no `_local` of its own, so fall
    back to the sibling main checkout (`../Strata-MiniMax`) when it is missing.
    """
    env = os.environ.get("STRATA_LOCAL")
    if env:
        return Path(env)
    here = REPO / "_local"
    if (here / "ref_out" / "heldout").is_dir():
        return here
    sibling = REPO.parent / "Strata-MiniMax" / "_local"
    if (sibling / "ref_out" / "heldout").is_dir():
        return sibling
    return here
LONG = "long_science_darwin"
# The sweep's flags (the A1-mix recipe is scored with exactly these).
SWEEP_FLAGS = ["--vram-slots", "0", "--layer-split", "auto", "--cpu-threads", "12"]
EXIT_LOCK_HELD, EXIT_NO_RAM = 2, 3


def heldout_names(ref: Path) -> list[str]:
    names = sorted(d for d in os.listdir(ref) if os.path.isdir(ref / d))
    if LONG not in names:
        raise SystemExit(f"{ref} has no {LONG}: is this the held-out reference dir?")
    return names


def phase_cmd(args, prompts: list[str], tag: str) -> list[str]:
    g3 = [sys.executable, str(REPO / "tools" / "minimax" / "g3_sparsa.py"),
          "--model", args.model, "--out", args.out, "--tier", args.tier,
          "--ref", args.ref, "--ids", args.ids, "--no-ppl",
          "--memory-limit-mib", str(args.memory_limit_mib),
          "--prompts", ",".join(prompts), *SWEEP_FLAGS]
    if args.rescore:
        g3.append("--rescore")
    # deliberately no --allow-partial: that flag means "score only prompts that
    # already have a complete dump", which would skip everything on a first run.
    # Without it each phase runs whatever is missing and reuses what is there.
    return [sys.executable, str(REPO / "tools" / "minimax" / "pc_launch.py"),
            "--owner", args.owner, "--max-minutes", str(args.max_minutes),
            "--", *g3]


def run_phase(args, prompts: list[str], tag: str) -> str:
    """Returns 'ok', 'timeout' (no turn inside --max-wait) or 'failed'."""
    deadline = time.time() + args.max_wait * 60
    cmd = phase_cmd(args, prompts, tag)
    while True:
        print(f"\n=== phase {tag}: {len(prompts)} prompts ===", flush=True)
        rc = subprocess.call(cmd)
        if rc == 0:
            return "ok"
        if rc == EXIT_LOCK_HELD or rc == EXIT_NO_RAM:
            if time.time() + args.retry > deadline:
                print(f"phase {tag}: no PC turn within {args.max_wait:.0f} min; stopping here", flush=True)
                return "timeout"
            print(f"phase {tag}: rc={rc} (not my turn); retrying in {args.retry}s", flush=True)
            time.sleep(args.retry)
            continue
        # anything else (a real engine failure, or the turn limit stopping the
        # child) leaves reusable dumps behind: a later run continues from them.
        if time.time() + args.retry > deadline:
            print(f"phase {tag}: rc={rc}, out of wait budget", flush=True)
            return "failed"
        print(f"phase {tag}: rc={rc} (interrupted or failed); retrying in {args.retry}s", flush=True)
        time.sleep(args.retry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", default=None)
    ap.add_argument("--ref", default=str(local_root() / "ref_out" / "heldout"))
    ap.add_argument("--ids", default=str(local_root() / "g3_heldout" / "ids"))
    ap.add_argument("--out", default=str(local_root() / "g3_heldout" / "iq2xs-d3"))
    ap.add_argument("--memory-limit-mib", type=int, default=80000,
                    help="pinned expert budget; 80000 is the sweep setting. Lower it only when the shared PC "
                         "has less free RAM than the launcher's 10 GB floor allows - it changes speed, "
                         "not the logits")
    ap.add_argument("--owner", default="heldout")
    ap.add_argument("--max-minutes", type=float, default=45.0, help="pc_launch turn limit per attempt")
    ap.add_argument("--max-wait", type=float, default=60.0, help="minutes to wait for each phase's turn")
    ap.add_argument("--retry", type=float, default=600.0, help="seconds between lock retries")
    ap.add_argument("--rescore", action="store_true", help="phase 3 only, never start the engine")
    ap.add_argument("--only-phase", choices=["short", "long", "aggregate"], default=None)
    args = ap.parse_args()
    args.tier = args.tier or os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    os.makedirs(args.out, exist_ok=True)

    ref = Path(args.ref)
    names = heldout_names(ref)
    short = [n for n in names if n != LONG]
    phases = [("short", short), ("long", [LONG]), ("aggregate", names)]
    if args.only_phase:
        phases = [p for p in phases if p[0] == args.only_phase]

    status = {}
    for tag, prompts in phases:
        status[tag] = run_phase(args, prompts, tag)
        if status[tag] != "ok" and tag != "aggregate":
            print(f"stopping after phase {tag}: {status[tag]}", flush=True)
            break
    print("\nphases:", status)
    print(f"results: {os.path.join(args.out, 'results.json')} / results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
