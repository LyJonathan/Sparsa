#!/usr/bin/env python3
"""Capture layer-40 routed-expert inputs for documents that are NOT the calibration document.

The GPTQ Hessian for the layer-40 pilots was built from `long_prose_austen` (4093 tokens of
Pride and Prejudice, `_local/ref_out/dumps/long/dump-L40`). Scoring a quantizer on the same
document's next-nearest tokens measures interpolation, not generalisation: the first pilot's
-71%/-75% on gate/up came from exactly that mistake.

This script produces the fix. It streams layers 0..40 of the fp32 reference for one or more
documents from `_local/ref_out/tools/g3_heldout/prompts.jsonl` (the 14 held-out prompts, which the
imatrix calibration corpus `calibration.txt` is documented as disjoint from), dumps layer 40's
per-expert tensors, and rebuilds the layer input with `exp_input.py`, exactly as the calibration
dump was built. The result goes to `strata-mm-quant --exp-eval-x/--exp-eval-dumps`: those rows are
only ever used for `output_rel` and never enter a Hessian, so `--exp-holdout 0` on the fit side is
correct.

Run it inside a pc_launch turn (it uses tens of GB of RAM and about half an hour):

    python tools/minimax/quant/heldout_capture.py --doc long_science_darwin \\
        --model-dir "E:/Jon's Projects/AI/Strata Fork Minimax/models/MiniMax-M3.1-Flash" \\
        --ref "E:/Jon's Projects/AI/Strata Fork Minimax/Strata-MiniMax/_local/ref_out/tools/ref" \\
        --out "E:/Jon's Projects/AI/Strata Fork Minimax/Strata-MiniMax/_local/gptq/heldout"

Cost, measured and estimated: 41 layers x ~12-20 s per document at bf16 on the 5900X, so two
documents is about 30 minutes. `--release-shards` keeps the process near one layer's footprint
instead of letting the safetensors mmaps accumulate (the 60-layer fp32 pass peaked at 135 GB).

The script never takes the lock itself; wrap it:

    python tools/minimax/pc_launch.py --owner quant --est-ram-gb 60 --max-minutes 45 -- \\
        <venv-ref python> tools/minimax/quant/heldout_capture.py ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str], log_path: Path | None = None) -> int:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if log_path is None:
        return subprocess.call([str(c) for c in cmd])
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def expert_row_counts(dump: Path, n_expert: int = 128) -> list[tuple[int, int]]:
    import numpy as np

    out = []
    for e in range(n_expert):
        p = dump / f"expert_{e:03d}_token_idx.npy"
        if p.is_file():
            out.append((e, int(np.load(p).size)))
    out.sort(key=lambda kv: -kv[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", action="append", required=True,
                    help="prompt id from g3_heldout/prompts.jsonl (repeat for several, one pass)")
    ap.add_argument("--model-dir", required=True, help="MiniMax-M3.1-Flash fp32/bfloat16 checkpoint")
    ap.add_argument("--ref", required=True, help="directory holding stream_run.py, exp_input.py's model_ref.py")
    ap.add_argument("--prompts", default=None,
                    help="prompts.jsonl (default: the g3_heldout set next to the ref tools)")
    ap.add_argument("--out", required=True, help="output root for dumps and exp-x files")
    ap.add_argument("--python", default=sys.executable, help="the venv-ref interpreter")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="truncate each document (default: all of it)")
    ap.add_argument("--sleep-layers", action="store_true",
                    help="pass --release-shards to stream_run.py (default: on)")
    ap.add_argument("--no-release-shards", action="store_true")
    args = ap.parse_args()

    ref = Path(args.ref)
    prompts = Path(args.prompts) if args.prompts else ref.parent / "g3_heldout" / "prompts.jsonl"
    if not prompts.is_file():
        raise SystemExit(f"missing prompt set {prompts}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = [d.strip() for d in args.doc if d.strip()]

    # ---- 1. one streaming pass, layer 40 dumped for every document ----------------------------
    cmd = [
        args.python, str(ref / "stream_run.py"),
        "--model-dir", args.model_dir,
        "--prompts", str(prompts),
        "--only", ",".join(docs),
        "--layers", f"0-{args.layer}",
        "--dump-layers", str(args.layer),
        "--dump-prompts", ",".join(docs),
        "--dtype", args.dtype,
        "--out", str(out / "dumps"),
    ]
    if not args.no_release_shards:
        cmd.append("--release-shards")
    if args.max_tokens:
        cmd += ["--max-tokens", str(args.max_tokens)]
    rc = run(cmd, out / "stream.log")
    if rc != 0:
        raise SystemExit(f"stream_run.py exited {rc}; see {out / 'stream.log'}")

    # ---- 2. rebuild the layer input of each document (exp_input.py) ---------------------------
    summary = {"docs": [], "layer": args.layer, "dtype": args.dtype, "prompts": str(prompts)}
    dumps = out / "dumps"
    for doc in docs:
        named = dumps / f"dump-L{args.layer:02d}-{doc}"
        plain = dumps / f"dump-L{args.layer:02d}"
        dump = named if named.is_dir() else plain
        if not dump.is_dir():
            raise SystemExit(f"no dump for {doc} under {dumps}")
        x = out / f"exp-x-L{args.layer:02d}-{doc}.npy"
        rc = run([args.python, str(HERE / "exp_input.py"),
                  "--model", args.model_dir, "--dump", str(dump),
                  "--layer", str(args.layer), "--out", str(x)], out / f"exp_input-{doc}.log")
        if rc != 0:
            raise SystemExit(f"exp_input.py exited {rc} for {doc}; see {out / f'exp_input-{doc}.log'}")
        counts = expert_row_counts(dump)
        summary["docs"].append({
            "id": doc,
            "dump": str(dump),
            "exp_x": str(x),
            "experts_with_rows": len(counts),
            "rows_total": sum(c for _, c in counts),
            "top_experts": counts[:8],
        })
        print(f"{doc}: {sum(c for _, c in counts)} routed rows over {len(counts)} experts; "
              f"top {counts[:5]}")

    (out / "capture.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out / 'capture.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
