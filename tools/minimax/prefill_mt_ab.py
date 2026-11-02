"""tools/minimax/prefill_mt_ab.py - the multi-token IQ2_XS row, A/B on the prompt path.

Two questions, both answered by the same script because they are the same measurement with a different prompt:

  (b) the DEFAULT prefill path (`--minimax-prefill-batch 4096`, no `--minimax-prefill-experts`), 8K-token document,
      STRATA_IQ_MT=1 against =0, 3 interleaved reps.  If the default keeps the routed experts on the GPU, the CPU
      row is not on the path at all and the two arms must be identical within noise - the script says which.
  (c) the CPU-expert prefill arm (`--minimax-prefill-experts cpu`), ~2,400-token prompt, 3 interleaved reps.

Every run is its own `pc_launch.py` invocation (so the lock is taken and returned between reps and the other task
can slot in), and the drive is probed before each rep because a cold SSD rep is not the same measurement.

    python tools/minimax/prefill_mt_ab.py --doc README.md --mode default --reps 3

Only the engine's own counters are printed: prompt tokens, prompt ms, slots, cpu_ram, ssd MB.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def drive_gbs(path: str, gb: float = 2.0) -> float:
    size = 1 << 30
    n = int(gb * size)
    t0 = time.time()
    with open(path, "rb", buffering=0) as f:
        f.seek(os.path.getsize(path) // 3)
        read = 0
        while read < n:
            b = f.read(1 << 20)
            if not b:
                break
            read += len(b)
    dt = time.time() - t0
    return read / size / dt if dt > 0 else 0.0


def field(text: str, key: str, cast=float):
    for part in text.split():
        if part.startswith(key + "="):
            try:
                return cast(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=str(ROOT / "README.md"), help="a text file whose tokens make the prompt")
    ap.add_argument("--tokens", type=int, default=0, help="prompt length; 0 = 8192 for 'default', 2400 for 'cpu'")
    ap.add_argument("--mode", choices=["default", "cpu"], default="default")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default="D:/hf-d3/IQ2_XS-D3/MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf")
    ap.add_argument("--exe", default=str(ROOT / "build" / "strata-minimax.exe"))
    ap.add_argument("--launcher", default=str(HERE / "pc_launch.py"))
    ap.add_argument("--tokenizer", default="D:/hf-d3/IQ2_XS-D3/tokenizer/tokenizer.json")
    ap.add_argument("--memory-limit-mib", type=int, default=54000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    n_tok = a.tokens or (8192 if a.mode == "default" else 2400)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    ids = tok.encode(pathlib.Path(a.doc).read_text(encoding="utf-8", errors="replace")).ids
    while len(ids) < n_tok:
        ids = ids + ids
    ids = ids[:n_tok]
    text = f"GEN 1 temperature=0 seed=1 {' '.join(str(i) for i in ids)}\nSTOP\n"

    base = ["--serve", "--max-context", "16384", "--gpus", "0,1", "--layer-split", "auto",
            "--minimax-profile", "D:/hf-d3/profiles/mm3-d3.bin", "--cpu-threads", "12",
            "--memory-limit-mib", str(a.memory_limit_mib), "--minimax-prefill-batch", "4096",
            "--minimax-profile-time", "--minimax-stats"]
    if a.mode == "cpu":
        base += ["--minimax-prefill-experts", "cpu"]

    rows = []
    for rep in range(a.reps):
        for mt in (1, 0):
            g = drive_gbs(a.model)
            tries = 0
            while g < 1.0 and tries < 5:
                print(f"   drive {g:.2f} GB/s < 1: waiting 60 s", flush=True)
                time.sleep(60)
                g = drive_gbs(a.model)
                tries += 1
            cmd = [sys.executable, a.launcher, "--owner", "dspark", "--max-minutes", "20",
                   "--bench", "--record-peak", "--",
                   a.exe, "--model", a.model] + base
            env = dict(os.environ, STRATA_IQ_MT=str(mt))
            t0 = time.time()
            p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env=env, timeout=3600)
            wall = time.time() - t0
            done = [l for l in p.stdout.splitlines() if l.startswith("DONE ")]
            err = p.stderr.splitlines()
            row = {"rep": rep, "mode": a.mode, "mt": mt, "tokens": n_tok, "drive_gbs": round(g, 2),
                   "wall_s": round(wall, 1), "rc": p.returncode}
            if not done:
                row["status"] = "failed"
                row["tail"] = err[-3:] + ["stdout: " + l for l in p.stdout.splitlines()[-4:]]
            else:
                f = done[0].split()
                pms = float(f[3])
                row.update({"status": "ok", "prompt_tokens": int(f[2]), "prompt_s": round(pms / 1000.0, 1),
                            "pp_tok_s": round(int(f[2]) / pms * 1000.0, 2)})
                prof = [l for l in err if l.startswith("INFO minimax-profile token=")]
                if prof:
                    row["cpu_ram_ms"] = field(prof[-1], "cpu_ram")
                    row["ssd_wait_ms"] = field(prof[-1], "ssd_wait")
                tot = [l for l in err if l.startswith("INFO minimax-stats total")]
                if tot:
                    row["ssd_mb_per_token"] = round(field(tot[0], "ssd_bytes", int) / 1048576.0 / int(f[2]), 1)
                    row["ram_experts_per_layer"] = field(tot[0], "ram_experts_per_layer", int)
                slots = [l for l in err if "minimax-vram-slots dev=" in l]
                row["slots"] = {str(field(l, "dev", int)): field(l, "slots_here", int) for l in slots}
            rows.append(row)
            print(json.dumps(row), flush=True)
    good = [r for r in rows if r.get("status") == "ok"]
    for mt in (1, 0):
        v = [r["pp_tok_s"] for r in good if r["mt"] == mt]
        if v:
            print(f"mode={a.mode} MT={'on ' if mt else 'off'} {n_tok} tokens: "
                  f"pp {min(v):.1f}-{max(v):.1f} tok/s ({len(v)} reps)", flush=True)
    if len(good) == 2 * a.reps:
        on = [r["pp_tok_s"] for r in good if r["mt"] == 1]
        off = [r["pp_tok_s"] for r in good if r["mt"] == 0]
        d = (max(on) - max(off)) / max(off) * 100.0
        print(f"mode={a.mode}: on-vs-off best {100.0 + d:+.1f} %"
              f"  ({'the CPU rows are on this path' if abs(d) > 3 else 'the CPU rows are NOT on this path'})",
              flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
