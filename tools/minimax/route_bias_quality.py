"""P7b/P10: the route nudge's full-model quality, measured instead of extrapolated.

Round 3 measured `--minimax-route-bias on` at 4.70 against 4.37 tok/s on the full model (+7.5%), but never measured
what it costs in logits there.  This script builds one teacher-forced id stream from the six short G3 prompts, runs
`gate_logits.py` with two fresh engine loads (off = reference, on = the arm), and applies the agreed decision rule:

    make it the default only if top-1 drops by <= 1 percentage point and KL64 rises by <= 0.02 against off.

KL64 is the KL over the top-64 reference probabilities, the same metric `g3_sparsa.py` reports.

    python tools/minimax/route_bias_quality.py --exe build/strata-minimax.exe \
        --model D:/hf-d3/IQ2_XS-D3/MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf \
        --ids-dir ../Strata-MiniMax/_local/g3_iq2xs/ids --positions 320 \
        -- --gpus 0,1 --layer-split auto --minimax-profile D:/hf-d3/profiles/mm3-d3.bin --memory-limit-mib 90000

The engine arguments after `--` are passed to both arms.  The JSON also records the pass/fail of the rule.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SHORT = [
    "chinese_local_llm",
    "code_lru",
    "math_bakery",
    "prose_moe",
    "multi_turn_threads",
    "tool_weather",
]


def read_ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".bin" and len(raw) % 4 == 0:
        return [int(x) for x in np.frombuffer(raw, dtype=np.int32)]
    return [int(x) for x in raw.decode("utf-8", "replace").replace(",", " ").split() if x.lstrip("-").isdigit()]


def build_stream(ids_dir: Path, positions: int, out: Path) -> list[int]:
    per = max(1, positions // len(SHORT))
    ids: list[int] = []
    for name in SHORT:
        p = ids_dir / f"{name}.bin"
        if not p.exists():
            raise SystemExit(f"missing prompt ids: {p}")
        ids.extend(read_ids(p)[:per])
    ids = ids[:positions]
    out.write_text(" ".join(str(i) for i in ids), encoding="utf-8")
    return ids


def kl64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = a - a.max(axis=1, keepdims=True)
    lb = b - b.max(axis=1, keepdims=True)
    pa = np.exp(la)
    pa /= pa.sum(axis=1, keepdims=True)
    pb = np.exp(lb)
    pb /= pb.sum(axis=1, keepdims=True)
    k = 64
    ia = np.argpartition(-pa, k, axis=1)[:, :k]
    ib = np.argpartition(-pb, k, axis=1)[:, :k]
    rows = np.arange(pa.shape[0])[:, None]
    pra = pa[rows, ia]
    prb = pb[rows, ib]
    lpa = np.log(np.maximum(pra, 1e-12))
    lpb = np.log(np.maximum(prb, 1e-12))
    return np.sum(pra * (lpa - lpb), axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids-dir", required=True)
    ap.add_argument("--positions", type=int, default=320)
    ap.add_argument("--out", default="route_bias_quality.json")
    ap.add_argument("engine_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.engine_args if x != "--"]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ids = tmp / "route_bias_ids.txt"
        stream = build_stream(Path(a.ids_dir), a.positions, ids)
        out_off, out_on = tmp / "off.npy", tmp / "on.npy"
        for flag, out in (("off", out_off), ("on", out_on)):
            cmd = [a.exe, "--model", a.model, "--minimax-eval", str(ids), "--minimax-dump-logits", str(out),
                   "--max-context", str(a.positions), "--minimax-allow-nonstandard",
                   "--minimax-route-bias", flag, *extra]
            print("arm", flag, flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2400)
            if p.returncode != 0:
                raise SystemExit(f"engine failed ({flag}):\n" + p.stderr[-3000:])
        base = np.load(out_off).astype(np.float64)
        got = np.load(out_on).astype(np.float64)
        if base.shape != got.shape:
            raise SystemExit(f"logit shape mismatch: {base.shape} vs {got.shape}")
        k = kl64(got, base)
        top1 = float((got.argmax(axis=1) == base.argmax(axis=1)).mean())
        res = {
            "positions": int(base.shape[0]),
            "prompt_ids": len(stream),
            "top1_agreement": top1,
            "top1_drop_points": 100.0 * (1.0 - top1),
            "kl64_mean": float(k.mean()),
            "kl64_p99": float(np.percentile(k, 99)),
            "max_abs": float(np.abs(got - base).max()),
            "rule": "top-1 drop <= 1.0 point and KL64 mean increase <= 0.02",
            "default_on": bool(100.0 * (1.0 - top1) <= 1.0 and float(k.mean()) <= 0.02),
        }
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
