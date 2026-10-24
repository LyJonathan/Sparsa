"""One timed serve-mode request, for a quick speed number on a new box (about 7 minutes).

    python3 tools/minimax/cloud/speed_probe.py --model /work/hf/.../...-00001-of-00013.gguf \
        --ids /work/g3ref/ids/ppl_text_hf.bin --exe /work/build/strata-minimax \
        --tokens 200 --gen 64

Drives the engine's own `--serve` protocol instead of the Python frontend: one `GEN <n> <ids>` line on
stdin, `PP`/`T`/`DONE` on stdout. `DONE <generated> <prompt> <prompt_ms> <decode_ms> ...` carries the
timings, and `--minimax-stats` (stderr) carries the tier counts.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", default="/work/build/strata-minimax")
    ap.add_argument("--ids", default="/work/g3ref/ids/ppl_text_hf.bin",
                    help="an int32 token-id file; the first --tokens ids become the prompt")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--cpu-threads", default="20")
    ap.add_argument("--memory-limit-mib", default="140000")
    ap.add_argument("--vram-slots", default="0", help="0 disables the VRAM expert tier")
    ap.add_argument("--max-context", default="8192")
    args = ap.parse_args()

    ids = np.fromfile(args.ids, dtype=np.int32)[:args.tokens]
    cmd = [args.exe, "--model", args.model, "--serve",
           "--minimax-tiers", "--gpus", args.gpus, "--layer-split", "auto", "--minimax-kv", "vram",
           "--max-context", str(args.max_context), "--minimax-dense-layers-gpu", "-1",
           "--minimax-prefill-batch", "512", "--minimax-prefill-experts", "cpu",
           "--minimax-prefill-dense", "f32", "--minimax-vram-slots-per-layer", args.vram_slots,
           "--vram-reserve-mib", "700", "--memory-limit-mib", args.memory_limit_mib,
           "--cpu-threads", args.cpu_threads, "--minimax-stats"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CUDA_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    print("CMD " + " ".join(cmd), flush=True)
    logpath = "/tmp/speed_probe.log"
    log = open(logpath, "wb")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)
    q = queue.Queue()

    def reader():
        for raw in p.stdout:
            q.put(raw.decode("utf-8", "replace").rstrip())

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            line = q.get(timeout=5)
        except queue.Empty:
            continue
        print("| " + line, flush=True)
        if line.startswith("READY"):
            break
    else:
        p.kill()
        raise SystemExit("engine never printed READY")

    print("load %.0fs; GEN %d with %d prompt ids" % (time.time() - t0, args.gen, len(ids)), flush=True)
    t1 = time.time()
    p.stdin.write(("GEN %d " % args.gen + ",".join(str(int(i)) for i in ids) + "\n").encode())
    p.stdin.flush()
    done = None
    while time.time() - t1 < 3600:
        try:
            line = q.get(timeout=60)
        except queue.Empty:
            print("(waiting)", flush=True)
            continue
        if line.startswith("T "):
            continue
        print("| " + line, flush=True)
        if line.startswith("DONE"):
            done = line
            break
    p.kill()
    log.close()

    txt = open(logpath, "rb").read().decode("utf-8", "replace")
    for line in [l for l in txt.splitlines() if "minimax-stats" in l][-3:]:
        print(line, flush=True)
    if not done:
        raise SystemExit("no DONE line")
    f = done.split()
    generated, prompt, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
    print("generated=%d prompt_tokens=%d prompt_ms=%.0f decode_ms=%.0f finish=%s"
          % (generated, prompt, pms, dms, f[5]), flush=True)
    print("prefill tok/s = %.2f" % (prompt / pms * 1000.0), flush=True)
    print("decode  tok/s = %.2f" % (generated / dms * 1000.0), flush=True)


if __name__ == "__main__":
    main()
