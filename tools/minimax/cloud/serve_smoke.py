"""Does this tier actually load, prefill and decode? One short serve-mode GEN.

    python3 tools/minimax/cloud/serve_smoke.py --model /work/hf/.../...-00001-of-00013.gguf \
        --ids /work/g3ref/ids/ppl_text_hf.bin

G3 proves the numbers; this proves the tier runs. It drives the engine's `--serve` protocol with a short
real-text prompt (the first 40 ids of the reference plain-text ids) and 16 greedy tokens, then prints the
prompt and the continuation detokenized with the model's tokenizer.json (put it at /work/tok/tokenizer.json;
the G3 box does not need it, this script only uses it when present). `--minimax-stats` is on so the tier
counts print too.

Exit code is 0 only when the engine reached READY, prefilled, and emitted 16 tokens and a DONE line.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

CUDA_LIB = "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib"
TOKENIZER = "/work/tok/tokenizer.json"


def detok(ids, tokenizer=TOKENIZER):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizer).decode([int(i) for i in ids], skip_special_tokens=False)
    except Exception as e:
        return "<no tokenizer: %s>" % e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", default="/work/build/strata-minimax")
    ap.add_argument("--ids", default="/work/g3ref/ids/ppl_text_hf.bin")
    ap.add_argument("--prompt-tokens", type=int, default=40)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--cpu-threads", default="20")
    ap.add_argument("--memory-limit-mib", default="140000")
    ap.add_argument("--vram-slots", default="0")
    ap.add_argument("--max-context", default="8192")
    ap.add_argument("--temperature", default="0", help="0 is greedy, the reproducible choice")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help="tokenizer.json used only to print the prompt and continuation")
    ap.add_argument("--cuda-lib", default=CUDA_LIB,
                    help="extra LD_LIBRARY_PATH entry; skipped when the directory is absent")
    args = ap.parse_args()

    prompt_ids = [int(x) for x in np.fromfile(args.ids, dtype=np.int32)[:args.prompt_tokens]]
    cmd = [args.exe, "--model", args.model, "--serve",
           "--minimax-tiers", "--gpus", args.gpus, "--layer-split", "auto", "--minimax-kv", "vram",
           "--max-context", str(args.max_context), "--minimax-dense-layers-gpu", "-1",
           "--minimax-prefill-batch", "512", "--minimax-prefill-experts", "cpu",
           "--minimax-prefill-dense", "f32", "--minimax-vram-slots-per-layer", args.vram_slots,
           "--vram-reserve-mib", "700", "--memory-limit-mib", args.memory_limit_mib,
           "--cpu-threads", args.cpu_threads, "--minimax-stats"]
    print("CMD " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    if os.path.isdir(args.cuda_lib):
        env["LD_LIBRARY_PATH"] = args.cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    logpath = "/tmp/serve_smoke.log"
    log = open(logpath, "wb")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)
    q = queue.Queue()

    def reader():
        for raw in p.stdout:
            q.put(raw.decode("utf-8", "replace").rstrip())

    threading.Thread(target=reader, daemon=True).start()

    t0 = time.time()
    ready = False
    while time.time() - t0 < 1200:
        try:
            line = q.get(timeout=10)
        except queue.Empty:
            continue
        print("| " + line, flush=True)
        if line.startswith("READY"):
            ready = True
            break
        if line.startswith("ERR") or line.startswith("FATAL"):
            print("STAGE load FAILED", flush=True)
            p.kill()
            sys.exit(2)
    if not ready:
        print("STAGE load FAILED: no READY in 1200s", flush=True)
        p.kill()
        sys.exit(2)
    print("STAGE load OK (%.0fs)" % (time.time() - t0), flush=True)

    t1 = time.time()
    p.stdin.write(("GEN %d temperature=%s " % (args.gen, args.temperature)
                   + ",".join(str(i) for i in prompt_ids) + "\n").encode())
    p.stdin.flush()
    out_ids, done = [], None
    while time.time() - t1 < 2400:
        try:
            line = q.get(timeout=60)
        except queue.Empty:
            print("(waiting)", flush=True)
            continue
        if line.startswith("T "):
            out_ids.append(int(line.split()[1]))
            continue
        if line.startswith("PP"):
            print("| " + line + "  (prefill OK)", flush=True)
            continue
        if line.startswith("DONE"):
            done = line
            break
        if line.startswith("ERR") or line.startswith("FATAL"):
            print("STAGE decode FAILED: " + line, flush=True)
            p.kill()
            sys.exit(3)
    p.kill()
    log.close()

    if not done or len(out_ids) != args.gen:
        print("STAGE decode FAILED: %d/%d tokens, done=%r" % (len(out_ids), args.gen, done), flush=True)
        sys.exit(3)
    print("STAGE decode OK: %d tokens in %.0fs" % (len(out_ids), time.time() - t1), flush=True)

    txt = open(logpath, "rb").read().decode("utf-8", "replace")
    for line in [l for l in txt.splitlines() if "minimax-stats" in l][-2:]:
        print(line, flush=True)
    print("---- prompt ----", flush=True)
    print(detok(prompt_ids, args.tokenizer), flush=True)
    print("---- continuation (greedy, %d tokens) ----" % len(out_ids), flush=True)
    print(detok(out_ids, args.tokenizer), flush=True)
    print("---- ids ----", flush=True)
    print(" ".join(str(i) for i in out_ids), flush=True)
    f = done.split()
    if len(f) >= 6:
        print("prompt_tokens=%s prompt_ms=%s decode_ms=%s finish=%s"
              % (f[2], f[3], f[4], f[5]), flush=True)


if __name__ == "__main__":
    main()
