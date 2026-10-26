"""G3 quality gate on the Strata engine: teacher-forced logits vs the fp32 reference.

Same prompts, same reference positions, same metrics as the llama.cpp harness
(`Strata-MiniMax/_local/g3_iq2xs/tools/compare.py` + `run_g3_d3.py`), but the
forward pass is `strata-minimax --minimax-eval`. The scoring math below is a copy
of `compare.py` (KL64, full-vocab KL, top-1 agreement, NLL against the true next
token) so the numbers are directly comparable; the ppl pass mirrors
`run_g3_d3.py` + `score_d3.py` (ppl_text_hf.bin, the reference's exact ids).

The gate stays teacher-forced top-1 and KL64.  Three extra slices are reported
next to it for comparison with a generation harness: top-1 on the reference's
own greedy path (`argmax(ref) == true next`), on path and ref p(argmax) >= 0.90,
and at ref p(argmax) >= 0.99.  They are extra report lines; nothing above is
recomputed from them.

Prompt set: by default the **6 short prompts** - chinese_local_llm, code_lru,
math_bakery, multi_turn_threads, prose_moe, tool_weather - are scored. The two
long ones (long_code_stdlib at 5,536 tokens and long_prose_austen at 4,093) are
opt-in with `--long-prompts`, because a teacher-forced pass over them costs about
an hour each on the host-KV path: `forward_batch` calls `attend_host` once per
token per layer, which is O(T^2) and single-threaded. `--prompts a,b` overrides
the set entirely (and warns when it names a long prompt).

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
# Scored by default. The two long prompts are opt-in (`--long-prompts`): a teacher-forced pass
# over them is O(T^2) on the host-attention path and costs about an hour each.
LONG_PROMPTS = ("long_code_stdlib", "long_prose_austen")


# --------------------------------------------------------------------------- scoring (copy of compare.py)
def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def conditioned_slices(ref, quant, ids):
    """The extra slices that make this benchmark comparable to a generation harness.

    The gate is teacher-forced top-1 on every prompt position, where most positions are open-ended and the
    reference itself picks the true next token only a quarter of the time.  A generation harness instead
    scores tokens the reference itself produced, which are on its own greedy path and mostly confident.
    These three slices approximate that, and are reported next to the gate (never instead of it):

      on_path          argmax(ref) == the true next token ("on the reference's greedy path")
      on_path_conf90   on_path and ref p(argmax) >= 0.90
      conf99           ref p(argmax) >= 0.99 (also includes a confident wrong reference, hence both top-1
                       against the true token and agreement with the reference are returned)

    `ids` are the prompt's token ids; row i predicts ids[len(ids) - n + i + 1], as `score_prompt`.
    """
    ref = np.asarray(ref, dtype=np.float64)
    q = np.asarray(quant, dtype=np.float64)
    ids = np.asarray(ids).astype(np.int64)
    n = ref.shape[0]
    lp = log_softmax(ref)
    p = np.exp(lp)
    top1_ref = ref.argmax(-1)
    top1_q = q.argmax(-1)
    start = len(ids) - n
    tgt = np.array([ids[start + i + 1] if start + i + 1 < len(ids) else -1 for i in range(n)])
    valid = tgt >= 0
    ref_p = p[np.arange(n), np.clip(top1_ref, 0, None)]
    agree = top1_ref == top1_q
    on_path = valid & (top1_ref == tgt)

    def sliced(mask):
        count = int(mask.sum())
        if count == 0:
            return {"n": 0, "top1": float("nan"), "top1_agreement": float("nan")}
        return {"n": count,
                "top1": float((top1_q[mask] == tgt[mask]).mean()),
                "top1_agreement": float(agree[mask].mean())}

    return {
        "on_path": sliced(on_path),
        "on_path_conf90": sliced(on_path & (ref_p >= 0.90)),
        "conf99": sliced(valid & (ref_p >= 0.99)),
    }


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
        "conditioned": conditioned_slices(ref, q, ids),
        "_kl64": kl64,
        "_klv": kl_vocab,
    }


def overall(rows):
    k64 = np.concatenate([r["_kl64"] for r in rows])
    kv = np.concatenate([r["_klv"] for r in rows])
    n = len(k64)
    conditioned = {}
    for key in ("on_path", "on_path_conf90", "conf99"):
        count = sum(r["conditioned"][key]["n"] for r in rows)
        if count == 0:
            conditioned[key] = {"n": 0, "top1": float("nan"), "top1_agreement": float("nan")}
            continue
        # weighted by the slice's own n, not by the prompt's positions: a prompt with no position in a slice
        # must not pull that slice's mean toward its own rate.
        conditioned[key] = {
            "n": int(count),
            "top1": float(sum(r["conditioned"][key]["top1"] * r["conditioned"][key]["n"]
                              for r in rows if r["conditioned"][key]["n"]) / count),
            "top1_agreement": float(sum(r["conditioned"][key]["top1_agreement"] * r["conditioned"][key]["n"]
                                        for r in rows if r["conditioned"][key]["n"]) / count),
        }
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
        "conditioned": conditioned,
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
    if not res.get("overall"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# G3 on the Strata engine - {res['tier']}\n\n"
                    f"model: `{res['model']}`\n\nno prompt dumps were scored"
                    + (f"; skipped: {', '.join(res['skipped'])}" if res.get("skipped") else "") + "\n")
        return
    o = res["overall"]
    p = res["ppl"]
    v = res.get("validation")
    L = []
    L.append(f"# G3 on the Strata engine - {res['tier']}\n")
    L.append(f"model: `{res['model']}`  \n")
    L.append(f"engine: `{res['engine']}`  \n")
    L.append(f"config: `{res['engine_flags']}`\n")
    if res.get("prompt_set") == "short":
        L.append("prompt set: the 6 short prompts (long_code_stdlib and long_prose_austen are opt-in with "
                 "`--long-prompts`; each costs about an hour on the host-KV path)\n")
    if res.get("skipped"):
        L.append(f"**skipped (no dump): {', '.join(res['skipped'])}** - the aggregate below covers "
                 f"the {len(res['per_prompt'])} prompts that were scored.\n")
    L.append(f"## Overall ({o['positions']} scored positions, {len(res['per_prompt'])} reference prompts)\n")
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
    c = o.get("conditioned")
    if c:
        L.append("### Conditioned top-1 (extra; the gate above stays the gate)\n")
        L.append("The prompt set is mostly open-ended, so the reference itself picks the true next token on only "
                 "about a quarter of the positions. A generation harness instead scores tokens the reference "
                 "produced. These slices approximate that; `top1` is against the true next token, "
                 "`top1 = ref` is agreement with the reference's argmax (identical on the two on-path rows).\n")
        L.append("| slice | n | top-1 vs true next | top-1 = ref |")
        L.append("|---|---:|---:|---:|")
        for key, label in (("on_path", "on the reference's greedy path (ref argmax == true next)"),
                           ("on_path_conf90", "on path and ref p(argmax) >= 0.90"),
                           ("conf99", "ref p(argmax) >= 0.99 (includes confident reference errors)")):
            s = c[key]
            if s["n"]:
                L.append(f"| {label} | {s['n']} | {100*s['top1']:.2f}% | {100*s['top1_agreement']:.2f}% |")
            else:
                L.append(f"| {label} | 0 | - | - |")
        L.append("")
    L.append("## Per prompt\n")
    L.append("| prompt | pos | KL64 | p99 | max | KLvocab | top1 | on-path top1 (n) | p>=.99 top1 (n) | NLL quant | s |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in res["per_prompt"]:
        c = r.get("conditioned")
        onpath = f"{100*c['on_path']['top1']:.1f}% ({c['on_path']['n']})" if c and c["on_path"]["n"] else "-"
        conf = f"{100*c['conf99']['top1']:.1f}% ({c['conf99']['n']})" if c and c["conf99"]["n"] else "-"
        L.append(f"| {r['prompt']} | {r['positions']} | {r['kl64_mean']:.4f} | {r['kl64_p99']:.4f} | "
                 f"{r['kl64_max']:.4f} | {r['kl_vocab_mean']:.4f} | {100*r['top1_agreement']:.1f}% | "
                 f"{onpath} | {conf} | {r['nll_true_quant']:.4f} | {r['seconds']:.1f} |")
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
        if v.get("matched_note"):
            L.append(f"_{v['matched_note']}; the KL64 p99/max and full-vocab rows are the llama.cpp "
                     f"file's all-prompt values and are not comparable when prompts are skipped._\n")
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
    ap.add_argument("--prompts", default=None,
                    help="comma-separated subset of prompts (default: the 6 short prompts; --long-prompts adds "
                         "long_code_stdlib and long_prose_austen)")
    ap.add_argument("--long-prompts", action="store_true",
                    help="also score long_code_stdlib (5,536 tok) and long_prose_austen (4,093 tok). Each "
                         "costs about an hour of engine time on the host-KV path (O(T^2) host attention), "
                         "so they are opt-in")
    ap.add_argument("--no-ppl", action="store_true", help="skip the 2,048-token ppl pass")
    ap.add_argument("--only-ppl", action="store_true", help="skip the 8 prompt passes")
    ap.add_argument("--rescore", action="store_true",
                    help="only rescore existing dumps (never start the engine)")
    ap.add_argument("--dumps-dir", default=None,
                    help="score the dumps in this directory instead of <out>/dumps; with --rescore this "
                         "re-scores an earlier run in place, with no engine and no copy")
    ap.add_argument("--allow-partial", action="store_true",
                    help="skip a prompt that has no complete dump instead of running the engine for it; "
                         "the results carry a `skipped` list. For the long prompts whose host-attention "
                         "prefill is O(T^2) and would take an hour each")
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
    dumps = os.path.abspath(args.dumps_dir) if args.dumps_dir else os.path.join(out, "dumps")
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
    prompt_set = "short"
    if args.prompts:
        want = {s.strip() for s in args.prompts.split(",") if s.strip()}
        names = [n for n in names if n in want]
        missing = want - set(names)
        if missing:
            raise SystemExit(f"unknown prompts: {sorted(missing)}")
        prompt_set = "custom"
    elif not args.long_prompts:
        names = [n for n in names if n not in LONG_PROMPTS]
    else:
        prompt_set = "all"
    if args.prompts and (set(names) & set(LONG_PROMPTS)):
        print("note: --prompts names a long prompt; each of those costs about an hour on the host-KV path",
              flush=True)

    res = {"tier": tier, "model": os.path.abspath(args.model), "engine": exe,
           "engine_flags": " ".join(flags), "prompts": names, "prompt_set": prompt_set,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "per_prompt": [], "ppl": None,
           "skipped": []}
    if args.allow_partial:
        have = [n for n in names
                if dump_complete(os.path.join(dumps, n + ".npy"),
                                 int(np.load(os.path.join(ref, n, "logits_last128.npy"), mmap_mode="r").shape[0]))]
        res["skipped"] = [n for n in names if n not in have]
        if res["skipped"]:
            print("SKIP (no complete dump): " + ", ".join(res["skipped"]), flush=True)
        names = have

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
                  f"top1 {100*r['top1_agreement']:.1f}%  (KLvocab {r['kl_vocab_mean']:.4f}, "
                  f"on-path {100*r['conditioned']['on_path']['top1']:.1f}% n={r['conditioned']['on_path']['n']}, "
                  f"p>=.99 {100*r['conditioned']['conf99']['top1']:.1f}% n={r['conditioned']['conf99']['n']})",
                  flush=True)

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

        rows = []
        for metric, key, fmt in (("KL64 mean (lower better)", "kl64_mean", "{:.4f}"),
                                 ("KL64 p99, all positions", "kl64_p99", "{:.4f}"),
                                 ("KL64 max", "kl64_max", "{:.4f}"),
                                 ("full-vocab KL mean", "kl_vocab_mean", "{:.4f}"),
                                 ("top-1 agreement %", "top1_agreement", "{:.2f}")):
            if key not in ro or key not in so:
                continue
            a, b = ro[key], so[key]
            if key == "top1_agreement":
                a, b = 100 * a, 100 * b
            rows.append(row(metric, a, b, fmt))
        for key in ("nll_true_ref", "nll_true_quant"):
            if key in ro and key in so:
                rows.append(row("mean NLL " + key.replace("nll_true_", "vs true ("), ro[key], so[key]))
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
        # The reference file's `overall` block covers every prompt it scored (935 positions).  When this run
        # covers a subset (skipped long prompts), an aggregate delta against that block would compare different
        # position sets, so recompute the reference aggregate over the matched prompts only.
        if per and len(per) < len(ref_res.get("per_prompt", [])):
            n = sum(r["positions"] for r in res["per_prompt"])
            ro = dict(ro)
            ro["kl64_mean"] = sum(o["kl64_mean"] * r["positions"]
                                  for o, r in zip([by_name[r["prompt"]] for r in res["per_prompt"]],
                                                  res["per_prompt"])) / n
            ro["top1_agreement"] = sum(o["top1_agreement"] * r["positions"]
                                       for o, r in zip([by_name[r["prompt"]] for r in res["per_prompt"]],
                                                       res["per_prompt"])) / n
            for row_ in rows:
                if row_["metric"].startswith("KL64 mean"):
                    row_["llamacpp"] = "{:.4f}".format(ro["kl64_mean"])
                    d = so["kl64_mean"] - ro["kl64_mean"]
                    row_["delta"] = ("+" if d >= 0 else "") + "{:.4f}".format(d)
                if row_["metric"].startswith("top-1 agreement"):
                    row_["llamacpp"] = "{:.2f}".format(100 * ro["top1_agreement"])
                    d = 100 * (so["top1_agreement"] - ro["top1_agreement"])
                    row_["delta"] = ("+" if d >= 0 else "") + "{:.2f}".format(d)
        res["validation"] = {"llamacpp_results": args.validate_with, "rows": rows, "per_prompt": per}
        if per and len(per) < len(ref_res.get("per_prompt", [])):
            res["validation"]["matched_prompts"] = len(per)
            res["validation"]["matched_note"] = (
                f"llama.cpp aggregate recomputed over the same {len(per)} prompts "
                f"({len(per)} of {len(ref_res.get('per_prompt', []))})")
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    write_md(os.path.join(out, "results.md"), res)
    print("wrote", os.path.join(out, "results.json"), os.path.join(out, "results.md"))


if __name__ == "__main__":
    main()
