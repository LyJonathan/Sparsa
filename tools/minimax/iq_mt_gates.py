"""tools/minimax/iq_mt_gates.py - the multi-token IQ2_XS row's gates on real weights.

The unit test (`build/iq_mt_test.exe`) proves a column of `mt::dot_rows` equals ggml's one-pair `vec_dot` bit for bit
on random data.  This script proves the engine's own prefill path is that same arithmetic: for every prompt length
below, the batched prompt (`--minimax-prefill-batch N`, CPU routed experts) must produce the SAME logits as the
one-token path (`--minimax-prefill-batch 0`), max abs 0.0, and the same with the multi-token row switched off
(`STRATA_IQ_MT=0`) - so the row itself is the only thing that changed.

The lengths straddle the kernel's column groups (kMaxCols = 8): 1, 2, 7, 8, 9, 15, 16, 17.  Plus a whole chat turn
rendered with the checkpoint's own template (198 tokens on the MiniMax card's system prompt: 24 whole 8-column
groups and a partial one) and the same turn cut mid-way, so the last group is partial.

    python tools/minimax/iq_mt_gates.py --model ../models/quant-d3/p5a-l0-14-iq2xs.gguf \
        --slice-args "--gpus 0 --minimax-dense-layers-gpu 15 --minimax-tiers --minimax-vram-slots-per-layer 0 \
        --minimax-ram-experts-per-layer 128 --minimax-kv vram --cpu-threads 12 --memory-limit-mib 40000 --allow"

Everything goes through tools/minimax/pc_launch.py (the PC lock and the RAM floor).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LENGTHS = [1, 2, 7, 8, 9, 15, 16, 17]

TEXT = ("The history of the printing press begins in the fifteenth century, when movable type made it possible to "
        "reproduce a page of text quickly and cheaply. Within fifty years, presses were operating in more than two "
        "hundred towns across Europe, and the number of books in circulation had grown from a few thousand to "
        "millions. Printers experimented with typefaces, page layouts and paper sizes, and each experiment changed "
        "how readers encountered a text.")

# The MiniMax card's system prompt, rendered the way the checkpoint's template does it (the same text the FP32
# reference run used, Strata-MiniMax/_local/ref_out/dspark_taps/meta.json).
CHAT = ("]~!b[]~b]system\nYour model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. "
        "Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers "
        "of AI towards AGI.\n\n<thinking_instructions>\nYou have a thinking capability that allows you to reason "
        "step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags "
        "before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. "
        "When thinking is adaptive, decide on your own whether to think for the current turn.\nCurrent thinking mode: "
        "adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing "
        "function/tool results.\n</thinking_instructions>\n\n<model_id>8bb8149fe</model_id>[e~[\n]~b]developer\nYou are "
        "a helpful assistant.[e~[\n]~b]user\nExplain why the sky is blue.[e~[\n]~b]ai\n")


def tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def run(exe: str, model: str, ids, out: pathlib.Path, args: str, ctx: int, mt: int, launcher: str) -> np.ndarray:
    txt = out.with_suffix(".ids")
    txt.write_text(" ".join(str(i) for i in ids), encoding="utf-8")
    cmd = [sys.executable, launcher, "--owner", "dspark", "--max-minutes", "60", "--",
           exe, "--model", model, "--minimax-eval", str(txt), "--minimax-dump-logits", str(out),
           "--max-context", str(ctx), "--minimax-allow-nonstandard"] + args.split()
    env = dict(os.environ, STRATA_IQ_MT=str(mt))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=7200)
    if p.returncode != 0 or not out.exists():
        raise SystemExit(f"engine failed rc={p.returncode} (mt={mt}):\n" + p.stderr[-2500:])
    a = np.load(out)
    if a.shape[0] != len(ids):
        raise SystemExit(f"logits rows {a.shape[0]} != {len(ids)} ids (mt={mt}): the run did not finish")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slice-args", default="")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--exe", default=str(ROOT / "build" / "strata-minimax.exe"))
    ap.add_argument("--launcher", default=str(HERE / "pc_launch.py"))
    ap.add_argument("--tokenizer", default="D:/hf-d3/IQ2_XS-D3/tokenizer/tokenizer.json")
    ap.add_argument("--label", default="slice")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tok = tokenizer(a.tokenizer)
    base = tok.encode(TEXT).ids * 3
    chat = tok.encode(CHAT).ids
    cases = [(f"text {n}", base[:n]) for n in LENGTHS]
    cases.append(("chat turn (whole)", chat))
    cases.append(("chat turn (cut, partial chunk)", chat[:len(chat) - 8]))
    print(f"{a.label}: {len(chat)}-token chat turn, {len(chat) % 8} tokens in its last 8-column group", flush=True)

    rows = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="iqmt-"))
    for name, ids in cases:
        one = run(a.exe, a.model, ids, tmp / "one.npy", a.slice_args + " --minimax-prefill-batch 0", a.ctx, 1,
                  a.launcher)
        batch = run(a.exe, a.model, ids, tmp / "batch.npy",
                    a.slice_args + " --minimax-prefill-batch 512 --minimax-prefill-experts cpu", a.ctx, 1,
                    a.launcher)
        off = run(a.exe, a.model, ids, tmp / "off.npy",
                  a.slice_args + " --minimax-prefill-batch 512 --minimax-prefill-experts cpu", a.ctx, 0,
                  a.launcher)
        d_on = float(np.abs(batch - one).max())
        d_off = float(np.abs(off - one).max())
        ok = d_on == 0.0 and d_off == 0.0
        rows.append({"model": a.label, "case": name, "tokens": len(ids), "mt_on_maxabs": d_on,
                     "mt_off_maxabs": d_off, "pass": ok})
        print(f"  {name:32s} {len(ids):4d} tokens  MT on vs one-token {d_on:.1f}  MT off vs one-token {d_off:.1f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    npass = sum(1 for r in rows if r["pass"])
    print(f"{npass}/{len(rows)} cases pass (max abs 0.0 both ways)", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    sys.exit(0 if npass == len(rows) else 1)


if __name__ == "__main__":
    main()
