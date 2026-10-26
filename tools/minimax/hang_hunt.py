"""tools/minimax/hang_hunt.py - the batch-8 prefill stall, with evidence when it happens.

The prompt-length gate's window arm (`--minimax-prefill-batch 8`, CPU routed experts) stops without a result in about
2 of 40 runs, on both the multi-token and the untouched per-token path, and some runs merely take ~5x longer
(118 s against 23 s).  This runs that arm N times per MT setting and, when a run passes twice its own median, captures

  * `pc_dump.py stacks` - every thread's stack with module!function+offset (StackWalk64 + the module export tables),
  * `pc_dump.py dump`   - a minidump every 10 s afterwards, and
  * `pc_dump.py sample` - the cheap thread-state/wait-reason sample,

then kills the tree at the 120 s watchdog and prints where each thread was waiting.

    python tools/minimax/hang_hunt.py --reps 20            # both MT settings, interleaved
    python tools/minimax/hang_hunt.py --reps 20 --mt 1     # one arm only
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parents[1]
LAUNCH = HERE / "pc_launch.py"
DEFAULT_EXE = ROOT / "build" / "strata-minimax.exe"
DEFAULT_MODEL = ROOT / "models" / "quant-d3" / "p5a-l0-14-iq2xs.gguf"
DEFAULT_IDS = HERE / "p13_ids.txt"
SLICE_ARGS = ("--max-context 1024 --minimax-allow-nonstandard --gpus 0 --minimax-dense-layers-gpu 15 "
              "--minimax-tiers --minimax-vram-slots-per-layer 0 --minimax-ram-experts-per-layer 128 "
              "--minimax-kv vram --cpu-threads 12 --memory-limit-mib 40000 --minimax-prefill-experts cpu "
              "--minimax-prefill-batch 8").split()


def stack_evidence(pid: int, dump_dir: Path, tag: str) -> list[str]:
    """Every dbghelp call runs in its own process: StackWalk64 can fault on a JIT frame, and a fault there must not
    take the harness down with it."""
    tool = [sys.executable, str(HERE / "pc_dump.py")]
    out = []
    st = subprocess.run(tool + ["stacks", "--pid", str(pid)], capture_output=True, text=True, timeout=120)
    out.append("stacks" + ("" if st.returncode == 0 else f" (rc={st.returncode})") + ":\n" + (st.stdout or st.stderr)[-4000:])
    dmp = dump_dir / f"hung_{tag}.dmp"
    du = subprocess.run(tool + ["dump", "--pid", str(pid), "--out", str(dmp)], capture_output=True, text=True,
                        timeout=120)
    out.append("dump: " + (du.stdout.strip() or du.stderr.strip())[-400:])
    if dmp.exists():
        rd = subprocess.run(tool + ["read", str(dmp)], capture_output=True, text=True, timeout=120)
        out.append("dump read:\n" + (rd.stdout or rd.stderr)[-4000:])
    sa = subprocess.run(tool + ["sample", "--pid", str(pid)], capture_output=True, text=True, timeout=60)
    out.append("sample:\n" + (sa.stdout or sa.stderr)[-1500:])
    return out


def lock_available(launcher: Path) -> bool:
    """`free`, or `stale` - a killed launcher leaves a dead PID that the next one takes over."""
    out = subprocess.run([sys.executable, str(launcher), "--who"], capture_output=True, text=True).stdout
    return "owner=free" in out or "state=stale" in out


def engine_pid(marker: str) -> int | None:
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='strata-minimax.exe'\" | "
          f"Where-Object {{ $_.CommandLine -like '*{marker}*' }} | Select-Object -First 1 -ExpandProperty ProcessId)")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return None


def run_once(exe: Path, model: Path, ids: Path, mt: int, out: Path, launcher: Path, watchdog: float,
             stack_at: float, dump_dir: Path) -> dict:
    for _ in range(30):
        if lock_available(launcher):
            break
        print(f"  [{time.strftime('%H:%M:%S')}] waiting for the lock", flush=True)
        time.sleep(10)
    cmd = [sys.executable, str(launcher), "--owner", "dspark", "--max-minutes", "5", "--bench", "--",
           str(exe), "--model", str(model), "--minimax-eval", str(ids), "--minimax-dump-logits", str(out)] + SLICE_ARGS
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**__import__("os").environ, "STRATA_IQ_MT": str(mt)})
    marker = str(ids)                      # the engine's own command line carries the id file, so it names the child
    evidence = []
    stacks_taken = False
    while True:
        try:
            so, se = proc.communicate(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            pass
        elapsed = time.time() - t0
        pid = engine_pid(marker)
        if elapsed >= stack_at and not stacks_taken and pid:
            evidence.extend(stack_evidence(pid, dump_dir, f"mt{mt}_{int(elapsed)}s"))
            stacks_taken = True
        if elapsed >= watchdog:
            if pid:
                evidence.extend(stack_evidence(pid, dump_dir, f"mt{mt}_final"))
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            proc.communicate()
            return {"mt": mt, "hung": True, "s": round(time.time() - t0, 1), "evidence": evidence}
    return {"mt": mt, "finished": out.exists(), "rc": proc.returncode, "s": round(time.time() - t0, 1),
            "evidence": evidence}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--mt", default="both", choices=["both", "1", "0"])
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--ids", default=str(DEFAULT_IDS))
    ap.add_argument("--launcher", default=str(LAUNCH))
    ap.add_argument("--tokens", type=int, default=30)
    ap.add_argument("--watchdog", type=float, default=120.0)
    ap.add_argument("--out-dir", default=str(Path.home() / "AppData/Local/Temp/hang_hunt"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = Path(a.ids).read_text().split()[: a.tokens]
    ids_path = out_dir / "p.ids"
    ids_path.write_text(" ".join(ids))
    exe, model = Path(a.exe), Path(a.model)
    arms = [1, 0] if a.mt == "both" else [int(a.mt)]
    rows: list[dict] = []
    for rep in range(a.reps):
        for mt in arms:
            out = out_dir / f"lg_mt{mt}_{rep}.npy"
            if out.exists():
                out.unlink()
            # the logits file name carries the id set; the engine is found through it in engine_pid()
            rows.append({"rep": rep, **run_once(exe, model, ids_path, mt, out, Path(a.launcher), a.watchdog,
                                                stack_at=45.0, dump_dir=out_dir)})
            r = rows[-1]
            print(json.dumps({k: v for k, v in r.items() if k != "evidence"}), flush=True)
            if r.get("hung"):
                for e in r["evidence"]:
                    print("  " + e.replace("\n", "\n  "), flush=True)
    ok1 = [r for r in rows if r["mt"] == 1 and not r.get("hung") and r.get("finished")]
    ok0 = [r for r in rows if r["mt"] == 0 and not r.get("hung") and r.get("finished")]
    for tag, v in (("MT on", ok1), ("MT off", ok0)):
        if v:
            t = sorted(r["s"] for r in v)
            print(f"{tag}: {len(v)}/{a.reps} finished, median {statistics.median(t):.1f}s, "
                  f"range {t[0]:.1f}-{t[-1]:.1f}s")
    print(f"hangs: MT on {sum(1 for r in rows if r['mt'] == 1 and r.get('hung'))}, "
          f"MT off {sum(1 for r in rows if r['mt'] == 0 and r.get('hung'))}")
    (out_dir / "result.json").write_text(json.dumps(rows, indent=1))
    print(f"evidence in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
