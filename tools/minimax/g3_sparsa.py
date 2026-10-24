"""G3 quality gate on the Strata engine: teacher-forced logits vs the fp32 reference.

Same 8 prompts, same reference positions, same metrics as the llama.cpp harness
(`Strata-MiniMax/_local/g3_iq2xs/tools/compare.py` + `run_g3_d3.py`), but the
forward pass is `strata-minimax --minimax-eval`. The scoring math below is a copy
of `compare.py` (KL64, full-vocab KL, top-1 agreement, NLL against the true next
token) so the numbers are directly comparable; the ppl pass mirrors
`run_g3_d3.py` + `score_d3.py` (ppl_text_hf.bin, the reference's exact ids).

Reference layout (produced on the A100, fp32 weights):

  <ref>/<prompt>/token_ids.npy        int32/int64, the prompt's token ids
  <ref>/<prompt>/logits_last128.npy   float32 [rows, vocab], the LAST `rows`
                                      positions of the prompt (rows == all
                                      positions for the short prompts)
  <ref>/ppl_a100_fp32.json            {"ppl": 39.69, "mean_nll": 3.681, ...}

The engine dumps one row of next-token logits per input position, so
`--minimax-last-k <rows>` reproduces exactly the positions the reference saved.
Row i's target is token_ids[start + i + 1]; the final row has no target and is
excluded from NLL (compare.py does the same).

Usage
-----
    python tools/minimax/g3_sparsa.py --model D:\\hf-d3\\IQ2_XS-D3\\...-00001-of-00013.gguf \
        --out _local/g3_sparsa/iq2xs-d3 --tier MM-IQ2_XS-D3 \
        --ref "E:/.../Strata-MiniMax/_local/ref_out/logits" \
        --ids "E:/.../Strata-MiniMax/_local/g3_iq2xs/ids" \
        --gpus 0,1 --layer-split auto --kv auto --prefill-batch 512 \
        --vram-slots 0 --memory-limit-mib 96000 --max-context 8192

Everything the engine prints is kept: `<out>/logs/<prompt>.log` (stderr+stdout)
and the flags of every run in `results.json`.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

TOPK = 64
VOCAB_EXPECTED = 200064
PPL_IDS = "ppl_text_hf.bin"
PPL_REF = 39.69292841776886  # _local/ref_out/ppl_a100_fp32.json


# --------------------------------------------------------------------------- scoring (copy of compare.py)
def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def score_prompt(name, ref_dir, quant_logits, ids):
    """compare.py's `score`, with the quant logits already loaded (fp32 npy)."""
    ref = np.load(os.path.join(ref_dir, name, "logits_last128.npy")).astype(np.float64)
    ids = np.asarray(ids).astype(np.int64)
    q = np.asarray(quant_logits, dtype=np.float64)
    if q.shape != ref.shape:
        raise SystemExit(f"{name}: quant logits shape {q.shape} != reference {ref.shape}")
    n = ref.shape[0]

    lp = log_softmax(ref)
    lq = log_softmax(q)
    p = np.exp(lp)
    kl_vocab = np.sum(p * (lp - lq), axis=-1)

    order = np.argsort(-ref, axis=-1)[:, :TOPK]
    r64 = np.take_along_axis(ref, order, 1)
    q64 = np.take_along_axis(q, order, 1)
    lpr = log_softmax(r64)
    lpq = log_softmax(q64)
    pr = np.exp(lpr)
    kl64 = np.sum(pr * (lpr - lpq), axis=-1)

    top1_ref = ref.argmax(-1)
    top1_q = q.argmax(-1)
    start = len(ids) - n
    tgt = np.array([ids[start + i + 1] if start + i + 1 < len(ids) else -1 for i in range(n)])
    valid = tgt >= 0
    ar = np.arange(n)
    nll_ref = -lp[ar, np.clip(tgt, 0, None)][valid]
    nll_q = -lq[ar, np.clip(tgt, 0, None)][valid]

    return {
        "prompt": name,
        "positions": int(n),
        "kl64_mean": float(kl64.mean()),
        "kl64_p50": float(np.percentile(kl64, 50)),
        "kl64_p99": float(np.percentile(kl64, 99)),
        "kl64_max": float(kl64.max()),
        "kl_vocab_mean": float(kl_vocab.mean()),
        "kl_vocab_p99": float(np.percentile(kl_vocab, 99)),
        "top1_agreement": float((top1_ref == top1_q).mean()),
        "top1_acc_vs_true_ref": float((top1_ref[valid] == tgt[valid]).mean()) if valid.any() else float("nan"),
        "top1_acc_vs_true_quant": float((top1_q[valid] == tgt[valid]).mean()) if valid.any() else float("nan"),
        "nll_true_ref": float(nll_ref.mean()) if len(nll_ref) else float("nan"),
        "nll_true_quant": float(nll_q.mean()) if len(nll_q) else float("nan"),
        "_kl64": kl64,
        "_klv": kl_vocab,
    }


def overall(rows):
    k64 = np.concatenate([r["_kl64"] for r in rows])
    kv = np.concatenate([r["_klv"] for r in rows])
    n = len(k64)
    return {
        "positions": int(n),
        "kl64_mean": float(k64.mean()),
        "kl64_p50": float(np.percentile(k64, 50)),
        "kl64_p99": float(np.percentile(k64, 99)),
        "kl64_max": float(k64.max()),
        "kl_vocab_mean": float(kv.mean()),
        "kl_vocab_p99": float(np.percentile(kv, 99)),
        # position-weighted, as compare.py (the overall is over all 935 positions)
        "top1_agreement": float(np.sum([r["top1_agreement"] * r["positions"] for r in rows]) / n),
        "top1_acc_vs_true_ref": float(np.sum([r["top1_acc_vs_true_ref"] * r["positions"] for r in rows]) / n),
        "top1_acc_vs_true_quant": float(np.sum([r["top1_acc_vs_true_quant"] * r["positions"] for r in rows]) / n),
        "nll_true_ref": float(np.sum([r["nll_true_ref"] * r["positions"] for r in rows]) / n),
        "nll_true_quant": float(np.sum([r["nll_true_quant"] * r["positions"] for r in rows]) / n),
    }


# --------------------------------------------------------------------------- engine runs
def read_ids_bin(path):
    return np.fromfile(path, dtype=np.int32)


def write_ids_text(path, ids):
    """The engine's --minimax-eval reads whitespace/comma separated ids (# comments)."""
    with open(path, "w", encoding="ascii") as f:
        for i in range(0, len(ids), 256):
            f.write(" ".join(str(int(v)) for v in ids[i:i + 256]))
            f.write("\n")


def dump_complete(path, rows, vocab=VOCAB_EXPECTED):
    """A dump is reusable only when its header says exactly [rows, vocab] and the file holds every byte; a killed
    run leaves either no file or a partial one."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        if len(head) < 12 or head[:6] != b"\x93NUMPY":
            return False
        major = head[6]
        hlen = int.from_bytes(head[8:10], "little") if major == 1 else int.from_bytes(head[8:12], "little")
        lg = np.load(path, mmap_mode="r")
        offset = 10 + hlen if major == 1 else 12 + hlen
        return tuple(lg.shape) == (rows, vocab) and os.path.getsize(path) == offset + rows * vocab * 4
    except Exception:
        return False


def run_engine_retry(args, logpath, retries=2):
    """A failed run writes no dump, so retrying is safe: one process's logits are never mixed with another's."""
    total = 0.0
    for attempt in range(retries + 1):
        path = logpath if attempt == 0 else logpath[:-4] + f".retry{attempt}.log"
        rc, dt, out = run_engine(args, path)
        total += dt
        if rc == 0:
            return total
        print(f"  attempt {attempt + 1}/{retries + 1} failed rc={rc}", flush=True)
    raise SystemExit(f"engine failed {retries + 1}x rc={rc}: {' '.join(args)} (log {logpath})")


def run_engine(args, logpath):
    """Run one engine invocation; returns (returncode, seconds, output). Never raises on a non-zero exit."""
    t = time.time()
    with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           errors="replace")
        lf.write(p.stdout)
    dt = time.time() - t
    tail = "\n".join(p.stdout.strip().splitlines()[-12:])
    print(tail, flush=True)
    print(f"  rc={p.returncode} wall={dt:.1f}s", flush=True)
    return p.returncode, dt, p.stdout


def ppl_from_logits(path, ids):
    """NLL of ids[1:] from the dump's rows 0..n-2, ppl = exp(mean NLL); chunked to bound RAM."""
    lg = np.load(path, mmap_mode="r")
    rows, vocab = lg.shape
    if rows < len(ids) - 1:
        raise SystemExit(f"ppl dump has {rows} rows, need {len(ids) - 1}")
    n = len(ids) - 1
    nll = np.empty(n, dtype=np.float64)
    top1 = np.empty(n, dtype=np.int64)
    step = 128
    for i in range(0, n, step):
        k = min(step, n - i)
        x = np.asarray(lg[i:i + k], dtype=np.float64)
        lp = log_softmax(x)
        tgt = np.asarray(ids[i + 1:i + 1 + k], dtype=np.int64)
        nll[i:i + k] = -lp[np.arange(k), tgt]
        top1[i:i + k] = x.argmax(-1)
    return {
        "tokens": int(len(ids)),
        "scored_positions": int(n),
        "ppl": float(np.exp(nll.mean())),
        "mean_nll": float(nll.mean()),
        "top1_acc_vs_true": float((top1 == np.asarray(ids[1:], dtype=np.int64)).mean()),
        "ppl_ref_fp32": PPL_REF,
        "ppl_ratio": float(np.exp(nll.mean()) / PPL_REF),
        "_nll": nll,
    }


# --------------------------------------------------------------------------- report
def write_md(path, res):
    o = res["overall"]
    p = res["ppl"]
    v = res.get("validation")
    L = []
    L.append(f"# G3 on the Strata engine - {res['tier']}\n")
    L.append(f"model: `{res['model']}`  \n")
    L.append(f"engine: `{res['engine']}`  \n")
    L.append(f"config: `{res['engine_flags']}`\n")
    L.append("## Overall (935 scored positions, 8 reference prompts)\n")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| KL64 mean | {o['kl64_mean']:.4f} |")
    L.append(f"| KL64 p50 | {o['kl64_p50']:.4f} |")
    L.append(f"| KL64 p99 (all positions) | {o['kl64_p99']:.4f} |")
    L.append(f"| KL64 max | {o['kl64_max']:.4f} |")
    L.append(f"| full-vocab KL mean | {o['kl_vocab_mean']:.4f} |")
    L.append(f"| top-1 agreement | {100*o['top1_agreement']:.2f}% |")
    L.append(f"| top-1 vs true next (ref) | {100*o['top1_acc_vs_true_ref']:.2f}% |")
    L.append(f"| top-1 vs true next (quant) | {100*o['top1_acc_vs_true_quant']:.2f}% |")
    L.append(f"| mean NLL vs true next (ref) | {o['nll_true_ref']:.4f} |")
    L.append(f"| mean NLL vs true next (quant) | {o['nll_true_quant']:.4f} |")
    L.append("")
    L.append("## Per prompt\n")
    L.append("| prompt | pos | KL64 | p99 | max | KLvocab | top1 | NLL quant | s |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in res["per_prompt"]:
        L.append(f"| {r['prompt']} | {r['positions']} | {r['kl64_mean']:.4f} | {r['kl64_p99']:.4f} | "
                 f"{r['kl64_max']:.4f} | {r['kl_vocab_mean']:.4f} | {100*r['top1_agreement']:.1f}% | "
                 f"{r['nll_true_quant']:.4f} | {r['seconds']:.1f} |")
    L.append("")
    L.append("## Plain-text perplexity (2,048 reference ids, 2,047 scored positions)\n")
    if p:
        L.append("| build | mean NLL | ppl | ppl / fp32 (39.69) | top-1 |")
        L.append("|---|---:|---:|---:|---:|")
        L.append(f"| **{res['tier']}** | **{p['mean_nll']:.3f}** | **{p['ppl']:.2f}** | **{p['ppl_ratio']:.2f}** | "
                 f"{100*p['top1_acc_vs_true']:.1f}% |")
    else:
        L.append("_ppl pass skipped (--no-ppl)_")
    L.append("")
    if v:
        L.append("## Validation against the llama.cpp harness (same model, same positions)\n")
        L.append("| metric | llama.cpp | Sparsa | delta |")
        L.append("|---|---:|---:|---:|")
        for row in v["rows"]:
            L.append(f"| {row['metric']} | {row['llamacpp']} | {row['sparsa']} | {row['delta']} |")
        L.append("")
        if v.get("per_prompt"):
            L.append("| prompt | KL64 llama.cpp | KL64 Sparsa | delta | top1 llama.cpp | top1 Sparsa |")
            L.append("|---|---:|---:|---:|---:|---:|")
            for row in v["per_prompt"]:
                L.append(f"| {row['prompt']} | {row['kl64_llamacpp']:.4f} | {row['kl64_sparsa']:.4f} | "
                         f"{row['delta']:+.4f} | {100*row['top1_llamacpp']:.1f}% | "
                         f"{100*row['top1_sparsa']:.1f}% |")
            L.append("")
    L.append(f"engine seconds: {res['engine_seconds']:.1f}s total "
             f"({res['engine_seconds']/60:.1f} min); scoring seconds: {res['score_seconds']:.1f}s\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="first shard of the split GGUF set")
    ap.add_argument("--out", required=True, help="output directory (results.json, results.md, dumps/)")
    ap.add_argument("--tier", default=None, help="tier label for the tables")
    ap.add_argument("--exe", default=None, help="strata-minimax.exe (default: <repo>/build/strata-minimax.exe)")
    ap.add_argument("--ref", default=None, help="reference logits dir (default <repo>/_local/ref_out/logits)")
    ap.add_argument("--ids", default=None, help="reference ids dir (default <repo>/_local/g3_iq2xs/ids)")
    ap.add_argument("--prompts", default=None, help="comma-separated subset of prompts (smoke runs)")
    ap.add_argument("--no-ppl", action="store_true", help="skip the 2,048-token ppl pass")
    ap.add_argument("--only-ppl", action="store_true", help="skip the 8 prompt passes")
    ap.add_argument("--rescore", action="store_true",
                    help="only rescore existing dumps (never start the engine)")
    ap.add_argument("--force", action="store_true",
                    help="rerun a prompt even when a complete dump from an earlier run exists")
    ap.add_argument("--keep-dumps", action="store_true", help="keep the logits npy files (2 GB for ppl)")
    ap.add_argument("--retries", type=int, default=2,
                    help="retry an engine run this many times on a non-zero exit (the batched CPU prefill "
                         "crashes intermittently with many threads; a retry never mixes two processes' logits)")
    # engine configuration, passed through verbatim
    ap.add_argument("--validate-with", default=None,
                    help="llama.cpp harness g3_results.json to diff the overall metrics against")
    ap.add_argument("--validate-ppl", default=None,
                    help="llama.cpp harness ppl_results.json for the ppl row")
    ap.add_argument("--engine-arg", action="append", default=[],
                    help="extra raw engine flag, repeatable (e.g. --engine-arg=--minimax-stats)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--layer-split", default="auto")
    ap.add_argument("--kv", default="auto")
    ap.add_argument("--prefill-batch", default="512")
    ap.add_argument("--prefill-experts", default="cpu")
    ap.add_argument("--prefill-dense", default="f32")
    ap.add_argument("--vram-slots", default="0")
    ap.add_argument("--ram-experts", default=None)
    ap.add_argument("--memory-limit-mib", default=None)
    ap.add_argument("--vram-reserve-mib", default="700")
    ap.add_argument("--max-context", default="8192")
    ap.add_argument("--dense-layers-gpu", default="-1")
    ap.add_argument("--cpu-threads", default="24")
    ap.add_argument("--tiers", default="on", choices=["on", "off"])
    ap.add_argument("--no-adaptive", action="store_true")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exe = args.exe or os.path.join(repo, "build", "strata-minimax.exe")
    ref = args.ref or os.path.join(repo, "_local", "ref_out", "logits")
    ids_dir = args.ids or os.path.join(repo, "_local", "g3_iq2xs", "ids")
    tier = args.tier or os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    out = os.path.abspath(args.out)
    dumps = os.path.join(out, "dumps")
    logs = os.path.join(out, "logs")
    for d in (out, dumps, logs):
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(exe):
        raise SystemExit(f"engine not found: {exe} (build it with tools/minimax/build_windows.ps1)")

    flags = ["--model", os.path.abspath(args.model)]
    if args.tiers == "on":
        flags.append("--minimax-tiers")
    flags += ["--gpus", args.gpus, "--layer-split", args.layer_split, "--minimax-kv", args.kv,
              "--max-context", str(args.max_context),
              "--minimax-dense-layers-gpu", str(args.dense_layers_gpu),
              "--minimax-prefill-batch", str(args.prefill_batch),
              "--minimax-prefill-experts", args.prefill_experts,
              "--minimax-prefill-dense", args.prefill_dense,
              "--minimax-vram-slots-per-layer", str(args.vram_slots),
              "--vram-reserve-mib", str(args.vram_reserve_mib),
              "--cpu-threads", str(args.cpu_threads)]
    if args.ram_experts is not None:
        flags += ["--minimax-ram-experts-per-layer", str(args.ram_experts)]
    if args.memory_limit_mib is not None:
        flags += ["--memory-limit-mib", str(args.memory_limit_mib)]
    if args.no_adaptive:
        flags.append("--no-minimax-adaptive")
    flags += args.engine_arg

    names = sorted(d for d in os.listdir(ref) if os.path.isdir(os.path.join(ref, d)))
    if args.prompts:
        want = {s.strip() for s in args.prompts.split(",") if s.strip()}
        names = [n for n in names if n in want]
        missing = want - set(names)
        if missing:
            raise SystemExit(f"unknown prompts: {sorted(missing)}")

    res = {"tier": tier, "model": os.path.abspath(args.model), "engine": exe,
           "engine_flags": " ".join(flags), "prompts": names,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "per_prompt": [], "ppl": None}

    # ---- the 8 teacher-forced passes
    score_s = 0.0
    engine_s = 0.0
    if not args.only_ppl:
        for name in names:
            ids = read_ids_bin(os.path.join(ids_dir, name + ".bin"))
            ids_txt = os.path.join(dumps, name + ".ids.txt")
            write_ids_text(ids_txt, ids)
            npy = os.path.join(dumps, name + ".npy")
            rows = int(np.load(os.path.join(ref, name, "logits_last128.npy"), mmap_mode="r").shape[0])
            dt = 0.0
            reusable = dump_complete(npy, rows)
            if args.rescore and not reusable:
                raise SystemExit(f"--rescore: no complete dump for {name} ({npy})")
            if args.force or not reusable:
                run = [exe, *flags, "--minimax-eval", ids_txt, "--minimax-dump-logits", npy,
                       "--minimax-last-k", str(rows)]
                print(f"RUN {name}: {len(ids)} tokens, last {rows} dumped", flush=True)
                if os.path.exists(npy):
                    os.remove(npy)  # never score a dump from an earlier run
                dt = run_engine_retry(run, os.path.join(logs, name + ".log"), args.retries)
                engine_s += dt
                if not dump_complete(npy, rows):
                    raise SystemExit(f"engine exited 0 but {npy} is not a complete [{rows}, "
                                     f"{VOCAB_EXPECTED}] dump")
            else:
                print(f"REUSE {name}: complete dump already at {npy}", flush=True)
            q = np.load(npy, mmap_mode="r")[:rows]
            t = time.time()
            r = score_prompt(name, ref, q, ids)
            score_s += time.time() - t
            r["seconds"] = dt
            res["per_prompt"].append(r)
            print(f"{name:20s} T={r['positions']:4d} KL64 {r['kl64_mean']:.4f} p99 {r['kl64_p99']:.4f} "
                  f"top1 {100*r['top1_agreement']:.1f}%  (KLvocab {r['kl_vocab_mean']:.4f})", flush=True)

    if res["per_prompt"]:
        res["overall"] = overall(res["per_prompt"])
        for r in res["per_prompt"]:
            r.pop("_kl64", None)
            r.pop("_klv", None)

    # ---- the ppl pass (reference's exact 2,048 ids)
    if not args.no_ppl:
        pids = read_ids_bin(os.path.join(ids_dir, PPL_IDS))
        ids_txt = os.path.join(dumps, "ppl.ids.txt")
        write_ids_text(ids_txt, pids)
        npy = os.path.join(dumps, "ppl.npy")
        reusable = dump_complete(npy, len(pids))
        if args.rescore and not reusable:
            raise SystemExit(f"--rescore: no complete ppl dump ({npy})")
        if args.force or not reusable:
            run = [exe, *flags, "--minimax-eval", ids_txt, "--minimax-dump-logits", npy,
                   "--minimax-last-k", str(len(pids))]
            print(f"RUN ppl: {len(pids)} tokens, all dumped", flush=True)
            if os.path.exists(npy):
                os.remove(npy)
            engine_s += run_engine_retry(run, os.path.join(logs, "ppl.log"), args.retries)
            if not dump_complete(npy, len(pids)):
                raise SystemExit(f"engine exited 0 but {npy} is not a complete ppl dump")
        else:
            print("REUSE ppl: complete dump already present", flush=True)
        t = time.time()
        res["ppl"] = ppl_from_logits(npy, pids)
        score_s += time.time() - t
        p = res["ppl"]
        print(f"PPL {p['ppl']:.2f} (ref {PPL_REF:.2f}, x{p['ppl_ratio']:.2f}) mean NLL {p['mean_nll']:.4f} "
              f"top1 {100*p['top1_acc_vs_true']:.1f}%", flush=True)
        if not args.keep_dumps:
            os.remove(npy)
            res["ppl"].pop("_nll", None)

    res["engine_seconds"] = engine_s
    res["score_seconds"] = score_s

    # ---- optional table against the llama.cpp harness on the same model/positions
    if args.validate_with and res.get("overall"):
        ref_res = json.load(open(args.validate_with, encoding="utf-8"))
        ro, so = ref_res["overall"], res["overall"]

        def row(metric, a, b, fmt="{:.4f}"):
            d = b - a
            return {"metric": metric, "llamacpp": fmt.format(a), "sparsa": fmt.format(b),
                    "delta": ("+" if d >= 0 else "") + fmt.format(d)}

        rows = [
            row("KL64 mean (lower better)", ro["kl64_mean"], so["kl64_mean"]),
            row("KL64 p99, all positions", ro["kl64_p99"], so["kl64_p99"]),
            row("KL64 max", ro["kl64_max"], so["kl64_max"]),
            row("full-vocab KL mean", ro["kl_vocab_mean"], so["kl_vocab_mean"]),
            row("top-1 agreement %", 100 * ro["top1_agreement"], 100 * so["top1_agreement"], "{:.2f}"),
            row("mean NLL vs true next", ro["nll_true_quant"], so["nll_true_quant"]),
        ]
        if args.validate_ppl and res.get("ppl"):
            rp = json.load(open(args.validate_ppl, encoding="utf-8"))
            rows.append(row("ppl (2,048 tokens)", rp["ppl_quant"], res["ppl"]["ppl"], "{:.2f}"))
            rows.append(row("ppl mean NLL", rp["mean_nll_quant"], res["ppl"]["mean_nll"]))
        # per-prompt KL64, matched by name
        by_name = {r["prompt"]: r for r in ref_res.get("per_prompt", [])}
        per = []
        for r in res["per_prompt"]:
            o = by_name.get(r["prompt"])
            if o:
                per.append({"prompt": r["prompt"], "kl64_llamacpp": o["kl64_mean"], "kl64_sparsa": r["kl64_mean"],
                            "delta": r["kl64_mean"] - o["kl64_mean"],
                            "top1_llamacpp": o["top1_agreement"], "top1_sparsa": r["top1_agreement"]})
        res["validation"] = {"llamacpp_results": args.validate_with, "rows": rows, "per_prompt": per}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    write_md(os.path.join(out, "results.md"), res)
    print("wrote", os.path.join(out, "results.json"), os.path.join(out, "results.md"))


if __name__ == "__main__":
    main()
