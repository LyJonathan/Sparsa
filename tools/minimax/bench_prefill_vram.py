"""P12 follow-up: is `--minimax-prefill-vram swap` the better default on this box?

Three arms, interleaved, one fresh process per rep, every engine start through pc_launch.py:

  reserve  the current default: the ~1.5 GiB/device batched-prompt workspace is subtracted before the slots are sized
  swap     the workspace borrows the slots for the prompt and returns them, so decode keeps the decode-only count
  batch0   --minimax-prefill-batch 0: no batched workspace at all (the token path runs the prompt)

The arm's own flags are appended to one baseline config so every number is comparable, and the drive is probed
(>= 1 GB/s) before each rep because a cold SSD rep is not the same measurement.

    python tools/minimax/bench_prefill_vram.py --reps 3 --order reserve,swap,batch0 --exe <binary> --out b.jsonl

Nothing is printed from the engine that holds a token id; only the counters below are kept.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

MODEL = "D:/hf-d3/IQ2_XS-D3/MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf"
PROFILE = "D:/hf-d3/profiles/mm3-d3.bin"

# The baseline every speed claim on this box uses.  --memory-limit-mib is 45000, not the 86000 of the brief: this box
# has 73.8 GB of available RAM right now (the WSL VM alone holds 25.5 GB of the 128), the launcher's 10 GB floor
# refuses 86000 outright, and at 54000 the estimate is 64.8 GB and every rep was stopped (three by the watchdog at
# 10.0 GB effective, then the rest refused at 8.2 GB left).  45000 estimates ~55 GB, which leaves ~19 GB.  All three
# arms share this config; stated with every number below.
BASE = ["--serve", "--max-context", "16384", "--gpus", "0,1", "--layer-split", "auto",
        "--minimax-profile", PROFILE, "--memory-limit-mib", "45000", "--cpu-threads", "12",
        "--minimax-prefill-batch", "4096", "--minimax-prefill-vram", "swap",
        "--minimax-prefill-experts", "mmvq", "--minimax-profile-time", "--minimax-stats"]

# Each arm overrides only what it is: the baseline's own flags already carry swap/batch 4096, so an arm removes them
# first (the last flag wins in the engine, but the arm list is what the reader sees).
ARMS = {
    "reserve": [("--minimax-prefill-vram", "reserve")],
    "swap": [("--minimax-prefill-vram", "swap")],
    "batch0": [("--minimax-prefill-batch", "0")],
}


def arm_args(arm):
    over = dict(ARMS[arm])
    out = []
    i = 0
    while i < len(BASE):
        if BASE[i] in over:
            out += [BASE[i], over[BASE[i]]]
            i += 2
            continue
        out.append(BASE[i])
        i += 1
    return out


def drive_gbs(path, gb=2.0):
    """A 2 GB unbuffered sequential read: the drive's rate right now."""
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


def gb_s(M, min_gbs=1.0, tries=6):
    g = drive_gbs(M)
    for _ in range(tries):
        if g >= min_gbs:
            break
        print(f"   drive {g:.2f} GB/s < {min_gbs}: waiting 60 s and re-probing", flush=True)
        time.sleep(60)
        g = drive_gbs(M)
    return g


def field(text, key, cast=float):
    for part in text.split():
        if part.startswith(key + "="):
            try:
                return cast(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "strata-minimax.exe"))
    ap.add_argument("--ids", default=os.path.join(ROOT, "ids249.txt"))
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--order", default="reserve,swap,batch0")
    ap.add_argument("--out", default=os.path.join(ROOT, "prefill_vram.jsonl"))
    ap.add_argument("--owner", default="dspark")
    ap.add_argument("--max-minutes", type=int, default=20)
    ap.add_argument("--launcher", default=os.path.join(ROOT, "tools", "minimax", "pc_launch.py"))
    a = ap.parse_args()

    ids = " ".join(open(a.ids).read().split())
    text = f"GEN {a.tokens} temperature=0 seed=1 {ids}\nSTOP\n"
    order = [x for x in a.order.split(",") if x]
    done = 0
    for rep in range(a.reps):
        for arm in order:
            g = gb_s(MODEL)
            cmd = [sys.executable, a.launcher, "--owner", a.owner, "--max-minutes", str(a.max_minutes), "--record-peak",
                   "--", a.exe, "--model", MODEL] + arm_args(arm)
            # another task may hold the lock: wait for it rather than recording a failed rep
            for attempt in range(40):
                t0 = time.time()
                p = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=a.max_minutes * 60 + 120)
                if p.returncode != 2:
                    break
                print(f"   rep {rep} {arm}: the PC is held by another task (rc=2); waiting 120 s", flush=True)
                time.sleep(120)
            wall = time.time() - t0
            lines = [l for l in p.stdout.splitlines() if l.startswith("DONE ")]
            err = p.stderr.splitlines()
            row = {"rep": rep, "arm": arm, "drive_gbs": round(g, 2), "wall_s": round(wall, 1),
                   "rc": p.returncode, "ctx": 16384, "tier_mib": 45000}
            if not lines:
                row["status"] = "failed"
                row["tail"] = err[-3:]
                print(json.dumps(row), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if p.returncode in (3, 4):
                    print(f"the launcher would not let the run start (rc={p.returncode}: the RAM floor); "
                          f"lower --memory-limit-mib and start again", flush=True)
                    sys.exit(1)
                continue
            f = lines[0].split()
            gen, ptok, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
            row.update({
                "status": "ok", "gen": gen, "prompt_tok": ptok,
                "prompt_s": round(pms / 1000.0, 1), "decode_s": round(dms / 1000.0, 1),
                "tok_s": round(gen / dms * 1000.0, 2) if dms > 0 else None,
            })
            slots = [l for l in err if "minimax-vram-slots dev=" in l]
            row["slots"] = {}
            for l in slots:
                d = field(l, "dev", int)
                if d is not None:
                    row["slots"][str(d)] = field(l, "slots_here", int)
            if row["slots"]:
                row["slots_min"] = min(row["slots"].values())
                row["slots_pair"] = "/".join(str(row["slots"][k]) for k in sorted(row["slots"]))
            tot = [l for l in err if l.startswith("INFO minimax-stats total")]
            if tot:
                row["ram_experts_per_layer"] = field(tot[0], "ram_experts_per_layer", int)
                row["ssd_reads"] = field(tot[0], "ssd_reads", int)
                row["ssd_bytes"] = field(tot[0], "ssd_bytes", int)
                if row.get("ssd_bytes") and row.get("gen"):
                    row["ssd_mb_per_token"] = round(row["ssd_bytes"] / 1048576.0 / a.tokens, 1)
            prof = [l for l in err if l.startswith("INFO minimax-profile token=")]
            if prof:
                row["cpu_ram_ms"] = field(prof[-1], "cpu_ram")
                row["ssd_wait_ms"] = field(prof[-1], "ssd_wait")
                row["cpu_gbs"] = field(prof[-1], "cpu_gbs")
            ws = [l for l in err if "minimax-vram-slots the" in l or "minimax-vram-slots reserving" in l]
            if ws:
                row["workspace"] = ws[0].split("minimax-vram-slots", 1)[1].strip()[:120]
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            done += 1
    print(f"{done} reps recorded to {a.out}", flush=True)


if __name__ == "__main__":
    main()
