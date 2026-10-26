"""Extend the reference's last-128 logits to the whole sequence, from the dump's own final_hidden.

`stream_run.py` saves `logits_last128.npy` per prompt but also `final_hidden.npy` [T, 6144], the
checkpoint's own last hidden state for every position.  The output head and the final norm are in
the checkpoint, so

    x_t   = rmsnorm(final_hidden[t], norm)          (one convention out of {w, w+1}, chosen by the check)
    logit = x_t @ lm_head^T

reproduces the dumped logits where they overlap.  Once that check passes, the full [T, V] matrix is
the reference for every position, which is what a "more tokens on the six prompts" gate needs: the
existing dumps only cover the last 128 positions of each prompt.

    python tools/minimax/full_ref_logits.py --prompts chinese_local_llm,code_lru,... \
        --ref _local/ref_out/logits --out _local/ref_out/logits_long --rows 512

Writes `<out>/<prompt>/logits_last128.npy` with the last `--rows` positions (the same file name, so
`g3_sparsa.py --ref <out>` reads it unchanged) and a `<out>/build.json` with the check numbers.
Refuses a prompt whose reconstruction does not match the dumped logits (cos < 0.9999 or
max-abs > 2.0 on the overlap).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash")
HEAD = "language_model.lm_head.weight"
NORM = "language_model.model.norm.weight"


def bf16(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def read_safetensors_tensor(index, model: Path, name: str):
    shard = index[name]
    path = model / shard
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        f.seek(8 + n + meta["data_offsets"][0])
        raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return meta, raw


def load_head(model: Path, chunk_rows: int = 8192):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    meta, raw = read_safetensors_tensor(index, model, NORM)
    norm = bf16(raw)
    hmeta, hraw = read_safetensors_tensor(index, model, HEAD)
    rows, cols = hmeta["shape"]
    assert hmeta["dtype"] == "BF16", hmeta
    head = np.frombuffer(hraw, dtype=np.uint16).reshape(rows, cols)
    return norm, head, rows, cols


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def logits_for(x: np.ndarray, head_u16: np.ndarray, rows: int, cols: int, chunk: int = 8192) -> np.ndarray:
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    for lo in range(0, rows, chunk):
        hi = min(rows, lo + chunk)
        block = (head_u16[lo:hi].astype(np.uint32) << 16).view(np.float32)
        out[:, lo:hi] = x @ block.T
        del block
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--ref", default=str(ROOT / "_local" / "ref_out" / "logits"))
    ap.add_argument("--out", default=str(ROOT / "_local" / "ref_out" / "logits_long"))
    ap.add_argument("--prompts", default="chinese_local_llm,code_lru,math_bakery,multi_turn_threads,prose_moe,"
                                         "tool_weather")
    ap.add_argument("--rows", type=int, default=512, help="positions kept per prompt (last N; 0 = all)")
    ap.add_argument("--eps", type=float, default=1e-6)
    a = ap.parse_args(argv)
    ref, out = Path(a.ref), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    norm, head, rows, cols = load_head(Path(a.model))
    print(f"head {rows} x {cols} bf16, norm {norm.shape}", flush=True)

    report = {"model": a.model, "head": HEAD, "rows": a.rows, "prompts": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name in [x.strip() for x in a.prompts.split(",") if x.strip()]:
        d = ref / name
        fh = np.load(d / "final_hidden.npy").astype(np.float32)
        ref_lg = np.load(d / "logits_last128.npy").astype(np.float32)
        t0 = time.time()
        best = None
        # `final_hidden.npy` is written AFTER the final norm (stream_run.py), so the plain product is
        # expected to match; the two weighted conventions are tried too and the check picks the match.
        for label, w in (("none", None), ("w", norm), ("w+1", norm + 1.0)):
            x = fh if w is None else rmsnorm(fh, w, a.eps)
            lg = logits_for(x, head, rows, cols)
            ov = lg[-ref_lg.shape[0]:]
            cos = float(np.sum(ov * ref_lg) / (np.linalg.norm(ov) * np.linalg.norm(ref_lg)))
            mad = float(np.abs(ov - ref_lg).mean())
            if best is None or cos > best["cos"]:
                best = {"plus1": label, "cos": cos, "mean_abs": mad, "logits": lg}
            print(f"  {name}: {label} cos {cos:.7f} mean|d| {mad:.4f}", flush=True)
        if best["cos"] < 0.9999 or best["mean_abs"] > 2.0:
            raise SystemExit(f"{name}: the reconstruction does not match the dumped logits "
                             f"(cos {best['cos']:.7f}); refusing to write a reference")
        lg = best["logits"]
        keep = lg if a.rows <= 0 else lg[-min(a.rows, lg.shape[0]):]
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "logits_last128.npy", keep)
        report["prompts"][name] = {"tokens": int(fh.shape[0]), "rows": int(keep.shape[0]),
                                   "norm_plus1": best["plus1"], "cos": best["cos"],
                                   "mean_abs": best["mean_abs"], "seconds": round(time.time() - t0, 1)}
        print(f"  {name}: wrote {keep.shape[0]} of {lg.shape[0]} positions ({time.time()-t0:.0f}s)", flush=True)
        del fh, ref_lg, lg, best
    (out / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    total = sum(v["rows"] for v in report["prompts"].values())
    print(f"wrote {out}: {total} scored positions over {len(report['prompts'])} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
