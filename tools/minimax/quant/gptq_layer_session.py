"""One PC turn, several `--expert-source` arms: the GPTQ layer against its own baseline.

The comparison this exists for is *within* one session: a layer's KL64 delta is a statement about
that prompt set, measured beside the baseline in the same turn (same page cache, same tier
admission, same CPU contention).  So every arm here runs as one child of `pc_launch.py`:

    python tools/minimax/pc_launch.py --owner quant --est-ram-gb 86 --max-minutes 45 -- \\
        python tools/minimax/quant/gptq_layer_session.py --base <D3 first shard> \\
        --arm d3= --arm old-iq2=pack-old.gguf --arm gptq-iq2=pack-gptq.gguf --out _local/gptq/session-iq2

An arm is `name=path`: an empty path is the base pack with no override; a path is a layer pack
written by `strata-mm-quant --exp-pack` and is passed to the engine as
`--expert-source <layer>-<layer>=<path>`.

Per-arm output resumes: `g3_sparsa.py` keeps the prompts it finished, so a turn that hits the
launcher's time limit leaves them for the next invocation (never pass `--force`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
G3 = ROOT / "tools" / "minimax" / "g3_sparsa.py"


def parse_arm(text: str) -> tuple[str, str]:
    name, eq, path = text.partition("=")
    if not name:
        raise SystemExit(f"arm without a name: {text!r}")
    return name.strip(), path.strip() if eq else ""


def run_arm(name: str, pack: str, a, out_dir: Path, layer: int) -> int:
    arm_dir = out_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(G3),
        "--model", str(a.base),
        "--out", str(arm_dir),
        "--tier", name,
        "--exe", str(a.exe),
        "--no-ppl",
        "--cpu-threads", str(a.cpu_threads),
        "--memory-limit-mib", str(a.memory_limit_mib),
        "--max-context", "8192",
        "--gpus", "0,1",
        "--vram-slots", "0",
    ]
    if a.prompts:
        cmd += ["--prompts", a.prompts]
    if pack:
        cmd += ["--engine-arg=--expert-source", f"--engine-arg={layer}-{layer}={pack}"]
    log = arm_dir / "session.log"
    print(f"[session] {name}: {' '.join(cmd)}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout)
    print(f"[session] {name}: rc {rc} (log {log})", flush=True)
    return rc


def read_overall(arm_dir: Path):
    f = arm_dir / "results.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d.get("overall")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the base pack's first shard")
    ap.add_argument("--arm", action="append", required=True, help="name=path (path empty: the base)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=40, help="the layer the packs override")
    ap.add_argument("--exe", default=str(ROOT / "build" / "strata-minimax.exe"))
    ap.add_argument("--prompts", default="", help="comma-separated subset; empty: the six short prompts")
    ap.add_argument("--memory-limit-mib", default="86000")
    ap.add_argument("--cpu-threads", default="16")
    ap.add_argument("--timeout", type=float, default=2400.0, help="per arm, seconds")
    a = ap.parse_args()
    if not Path(a.base).exists():
        raise SystemExit(f"base pack not found: {a.base}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [parse_arm(t) for t in a.arm]
    worst = 0
    for name, pack in arms:
        if pack and not Path(pack).exists():
            raise SystemExit(f"arm {name}: pack not found: {pack}")
        worst |= run_arm(name, pack, a, out, a.layer)
    base = None
    rows = []
    for name, pack in arms:
        o = read_overall(out / name)
        if o is None:
            rows.append((name, None))
            continue
        if base is None:
            base = o
        rows.append((name, o))
    print()
    print(f"{'arm':<18}{'positions':>10}{'KL64':>9}{'d KL64':>9}{'top1':>8}{'d top1':>8}")
    for name, o in rows:
        if o is None:
            print(f"{name:<18}{'incomplete':>10}")
            continue
        d = (o["kl64_mean"] - base["kl64_mean"]) if base else 0.0
        t = (100.0 * (o["top1_agreement"] - base["top1_agreement"])) if base else 0.0
        print(f"{name:<18}{o['positions']:>10}{o['kl64_mean']:>9.4f}{d:>9.4f}"
              f"{100.0 * o['top1_agreement']:>8.2f}{t:>8.2f}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
