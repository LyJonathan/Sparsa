#!/usr/bin/env python3
"""Greedy continuations from one serve session, as the input of spec_accept.py.

Two prompts, both built from token ids (a repetitive one and an open-ended one):

  repetitive   a block of `--block` ids repeated three times (prompt lookup's easy case: the continuation copies)
  open         a run of ids from a different part of the text (free-running continuation)

Each request is `GEN <max_new> temperature=0 <ids>` on the serve pipe (greedy, one request after another); the
`T <id>` lines are the output.  The trace goes to --out as JSON.

  python3 tools/minimax/greedy_trace.py --exe build/strata-minimax --model MODEL.gguf --ids ids.txt --out t.json
"""
import argparse
import json
import os
import subprocess
import sys


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="whitespace-separated token ids of real text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--block", type=int, default=200)
    ap.add_argument("--open-len", type=int, default=500)
    ap.add_argument("--max-context", default="32768")
    ap.add_argument("--extra", default="", help="extra engine flags, split on spaces")
    args = ap.parse_args()

    ids = read_ids(args.ids)
    b = args.block
    block = ids[0:b]
    prompts = [
        ("repetitive", block * 3),
        ("open", ids[b * 3:b * 3 + args.open_len]),
    ]
    flags = ["--gpus", "0,1", "--vram-reserve-mib", "700", "--minimax-kv", "vram", "--minimax-tiers",
             "--minimax-prefill-batch", "4096"] + args.extra.split()
    cmd = [args.exe, "--model", args.model, "--serve", "--max-context", args.max_context] + flags
    req = "".join("GEN %d temperature=0 %s\n" % (args.max_new, " ".join(str(t) for t in p))
                  for _, p in prompts) + "STOP\n"
    p = subprocess.run(cmd, input=req, capture_output=True, text=True, timeout=3000)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        return p.returncode
    outs = []
    cur = []
    for line in p.stdout.splitlines():
        if line.startswith("T "):
            cur.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            outs.append(cur)
            cur = []
    if len(outs) != len(prompts):
        sys.stderr.write("expected %d DONE lines, got %d\n" % (len(prompts), len(outs)))
        sys.stderr.write(p.stdout[-2000:])
        return 1
    traces = [{"name": name, "prompt": pr, "output": out} for (name, pr), out in zip(prompts, outs)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(traces, f)
    for t in traces:
        print("%s: prompt %d tokens, output %d tokens" % (t["name"], len(t["prompt"]), len(t["output"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
