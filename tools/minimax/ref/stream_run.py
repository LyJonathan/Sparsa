"""Layer-streaming runner for MiniMax-M3.1-Flash (text tower).

For each layer L = 0..59:
  * build the transformers decoder layer for L and load its dense weights
    (MXFP8 attention/indexer/dense-MLP/shared-expert dequantized to fp32 or bf16),
  * run every prompt in the set through it,
  * free the layer (routed experts are dequantized one at a time inside the
    layer and never persist).
Then the final Gemma RMSNorm and the untied lm_head are applied and the
requested outputs are written.

Outputs (per prompt, under ``--out``):
    <id>/token_ids.npy        int32 [T]
    <id>/top1.npy             int32 [T]        greedy token id at every position
    <id>/logits_last128.npy   f32   [min(128,T), 200064]
    <id>/final_hidden.npy     f32   [T, 6144]  after the final norm
    timings.json, meta.json

Modes:
    (default) run prompts.jsonl
    --ppl FILE            perplexity of a plain text file (teacher forced)
    --greedy "PROMPT"     greedy continuation, resident-model decode
    --dump-layer L        per-op activations for layer L (see README)

Large outputs belong in ``Strata-MiniMax/_local/ref_out`` (git-ignored).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chat import apply_chat_template
from model_ref import (
    DEFAULT_CONFIG,
    DEFAULT_MODEL_DIR,
    DumpContext,
    ReferenceModel,
    RSSMonitor,
    StreamingExperts,
)

DEFAULT_OUT = r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\ref_out"


def log(msg):
    print(msg, flush=True)


def load_prompts(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_layers(spec, n):
    if spec is None:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def encode_prompt(tok, item):
    text = item.get("rendered_text")
    if text is None:
        text = item.get("text", "")
    return tok.encode(text, add_special_tokens=False).ids


# ---------------------------------------------------------------------------
# dump helpers


def save_dump(dump_dir: Path, data: dict, meta: dict):
    dump_dir.mkdir(parents=True, exist_ok=True)
    index = {"tensors": {}, "meta": meta}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu()
            if arr.dtype == torch.bfloat16:
                arr = arr.to(torch.float32)
            np.save(dump_dir / f"{key}.npy", arr.numpy())
            index["tensors"][key] = {
                "file": f"{key}.npy",
                "dtype": str(np.dtype(arr.numpy().dtype)),
                "shape": list(arr.shape),
            }
        elif key.endswith("token_idx") or key.endswith("indices"):
            arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            np.save(dump_dir / f"{key}.npy", arr)
            index["tensors"][key] = {"file": f"{key}.npy", "dtype": str(arr.dtype), "shape": list(arr.shape)}
    (dump_dir / "index.json").write_text(json.dumps(index, indent=1))
    log(f"  dump: {len(index['tensors'])} tensors -> {dump_dir}")


# ---------------------------------------------------------------------------
# main streaming pass


def resolve_tokenizer(args) -> str:
    p = Path(args.tokenizer) if args.tokenizer else None
    if p is not None and p.exists():
        return str(p)
    local = Path(args.model_dir) / "tokenizer.json"
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        f"tokenizer.json not found (--tokenizer {args.tokenizer}, model dir {local})"
    )


def run_prompts(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = ReferenceModel(args.model_dir, dtype, args.config)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(resolve_tokenizer(args))
    items = load_prompts(args.prompts)[: args.limit]
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        by_id = {i["id"]: i for i in items}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit(f"--only ids not found: {missing}")
        items = [by_id[w] for w in wanted]
    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    if args.max_layers:
        layers = layers[: args.max_layers]

    encoded = []
    hidden = []
    position_ids = []
    for item in items:
        ids = encode_prompt(tok, item)
        t = torch.tensor([ids], dtype=torch.long)
        encoded.append(ids)
        position_ids.append(torch.arange(len(ids), dtype=torch.long).unsqueeze(0))
        hidden.append(model.embed_ids(t))
        log(f"prompt {item['id']}: {len(ids)} tokens")
        # free the embedding table; reloaded only if a later prompt needs it
    if not args.keep_embed:
        model._embed = None
        gc.collect()

    timings = {"layers": [], "total_start": time.time()}
    rss = RSSMonitor().start()
    dump_prompt_idx = None
    if args.dump_layer is not None:
        ids = [i["id"] for i in items]
        if args.dump_prompt not in ids:
            raise SystemExit(f"--dump-prompt must be one of {ids}")
        dump_prompt_idx = ids.index(args.dump_prompt)

    for pos, L in enumerate(layers):
        t0 = time.time()
        layer = model.make_layer(L)
        t_load = time.time() - t0
        t1 = time.time()
        dumps = None
        for i in range(len(items)):
            dump = None
            if dump_prompt_idx is not None and i == dump_prompt_idx and L == args.dump_layer:
                dump = DumpContext()
            hidden[i] = model.run_layer(layer, hidden[i], position_ids[i], dump)
            if dump is not None:
                dump.put("layer_out", hidden[i])
                dumps = dump.data
        t_run = time.time() - t1
        if dumps is not None:
            save_dump(
                outdir / f"dump-L{L:02d}",
                dumps,
                {"layer": L, "prompt": items[dump_prompt_idx]["id"], "dtype": args.dtype},
            )
        timings["layers"].append({"layer": L, "load_s": t_load, "run_s": t_run})
        log(f"layer {L:2d}: load {t_load:6.2f}s run {t_run:7.2f}s")
        del layer
        gc.collect()

    timings["stream_total_s"] = time.time() - timings["total_start"]
    log(f"layer streaming: {timings['stream_total_s']:.1f}s")

    results = {}
    if layers and layers[-1] == model.config.num_hidden_layers - 1:
        t0 = time.time()
        for item, ids, h in zip(items, encoded, hidden):
            d = outdir / item["id"]
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "token_ids.npy", np.asarray(ids, dtype=np.int32))
            fh = model.final_hidden(h)[0]
            np.save(d / "final_hidden.npy", fh.to(torch.float32).numpy())
            n_last = min(128, fh.shape[0])
            last_logits = model.logits(fh[-n_last:], chunk=args.logit_chunk)
            np.save(d / "logits_last128.npy", last_logits.numpy())
            # top-1 over all positions without materialising [T, vocab]
            top1 = []
            for i in range(0, fh.shape[0], args.logit_chunk):
                lg = model.logits(fh[i : i + args.logit_chunk], chunk=args.logit_chunk)
                top1.append(lg.argmax(-1))
            np.save(d / "top1.npy", torch.cat(top1).to(torch.int32).numpy())
            results[item["id"]] = {
                "n_tokens": len(ids),
                "top1_last": int(torch.cat(top1)[-1]),
            }
            log(f"  finalised {item['id']}")
        timings["head_s"] = time.time() - t0

    timings["total_s"] = time.time() - timings["total_start"]
    meta = {
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "layers": layers,
        "prompts": [i["id"] for i in items],
        "runtime_s": timings["total_s"],
        "peak_rss_gb": rss.stop(),
    }
    try:
        import psutil

        meta["rss_gb_at_end"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        pass
    (outdir / "timings.json").write_text(json.dumps(timings, indent=1))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"total {timings['total_s']:.1f}s -> {outdir}")
    return timings


# ---------------------------------------------------------------------------
# perplexity


def run_ppl(args):
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = ReferenceModel(args.model_dir, dtype, args.config)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(resolve_tokenizer(args))
    text = Path(args.ppl).read_text(encoding="utf-8")
    ids = tok.encode(text, add_special_tokens=False).ids[: args.max_tokens]
    log(f"ppl text: {len(ids)} tokens")
    t = torch.tensor([ids], dtype=torch.long)
    hidden = model.embed_ids(t)
    pos = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)
    t0 = time.time()
    rss = RSSMonitor().start()
    for L in range(model.config.num_hidden_layers):
        layer = model.make_layer(L)
        hidden = model.run_layer(layer, hidden, pos)
        del layer
        gc.collect()
        if (L + 1) % 5 == 0 or L == 0:
            log(f"layer {L:2d}  elapsed {time.time()-t0:7.1f}s")
    fh = model.final_hidden(hidden)[0]
    nll = torch.zeros(fh.shape[0] - 1, dtype=torch.float64)
    for i in range(0, fh.shape[0] - 1, args.logit_chunk):
        rows = fh[i : min(i + args.logit_chunk, fh.shape[0] - 1)]
        lg = model.logits(rows, chunk=args.logit_chunk).to(torch.float64)
        tgt = torch.tensor(ids[i + 1 : i + 1 + rows.shape[0]], dtype=torch.long)
        nll[i : i + rows.shape[0]] = torch.nn.functional.cross_entropy(lg, tgt, reduction="none")
    ppl = float(torch.exp(nll.mean()))
    out = {
        "tokens": len(ids),
        "ppl": ppl,
        "mean_nll": float(nll.mean()),
        "runtime_s": time.time() - t0,
        "dtype": args.dtype,
        "file": args.ppl,
        "peak_rss_gb": rss.stop(),
    }
    log(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "ppl.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# greedy continuation (resident model + KV cache)


def _lru_provider(provider, capacity):
    from collections import OrderedDict

    cache = OrderedDict()

    def get(expert):
        if expert in cache:
            cache.move_to_end(expert)
            return cache[expert]
        value = provider(expert)
        cache[expert] = value
        if len(cache) > capacity:
            cache.popitem(last=False)
        return value

    return get


def run_greedy(args):
    """Greedy decode with all layers resident.

    Decoding one token at a time with layer streaming would re-dequantize every
    attention weight per token, so this mode keeps the 60 decoder layers in RAM
    (bf16 by default) and only streams routed experts, with a bounded LRU
    expert cache per layer.  Memory: ~21 GB (bf16 dense) + cache entries
    (``--expert-cache`` x 113 MB per layer, default 4 -> ~27 GB).
    """
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = ReferenceModel(args.model_dir, dtype, args.config)
    layers = []
    for L in range(model.config.num_hidden_layers):
        layer = model.make_layer(L)
        if L >= 3 and args.expert_cache > 0:
            layer.mlp.experts.provider = _lru_provider(layer.mlp.experts.provider, args.expert_cache)
        layers.append(layer)
        if (L + 1) % 10 == 0:
            log(f"resident layer {L} loaded")

    text = apply_chat_template([{"role": "user", "content": args.greedy}])
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(resolve_tokenizer(args))
    ids = tok.encode(text, add_special_tokens=False).ids
    log(f"greedy prompt: {len(ids)} tokens")

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config)
    generated = []
    t_start = time.time()
    rss = RSSMonitor().start()
    with torch.no_grad():
        for step in range(args.max_new):
            if step == 0:
                inp = torch.tensor([ids], dtype=torch.long)
                pos = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)
            else:
                inp = torch.tensor([[generated[-1]]], dtype=torch.long)
                pos = torch.tensor([[len(ids) + step - 1]], dtype=torch.long)
            t0 = time.time()
            h = model.embed_ids(inp)
            pos_emb = model.rotary(h, position_ids=pos)
            for L, layer in enumerate(layers):
                h = layer(
                    h,
                    attention_mask=None,
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=pos_emb,
                )
            fh = model.final_hidden(h)
            lg = model.logits(fh[0, -1:], chunk=1)
            nxt = int(lg.argmax(-1)[0])
            generated.append(nxt)
            log(f"step {step:2d} {time.time()-t0:6.1f}s -> {nxt} {tok.decode([nxt])!r}")
            if nxt in (200020,):
                break
    out_text = tok.decode(generated)
    res = {
        "prompt": args.greedy,
        "prompt_tokens": len(ids),
        "generated_ids": generated,
        "generated_text": out_text,
        "runtime_s": time.time() - t_start,
        "peak_rss_gb": rss.stop(),
    }
    log(json.dumps(res, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "greedy.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--config", default=os.environ.get("MINIMAX_CONFIG", ""))
    ap.add_argument(
        "--tokenizer",
        default=os.environ.get("MINIMAX_TOKENIZER", ""),
    )
    ap.add_argument("--prompts", default=str(Path(__file__).with_name("prompts.jsonl")))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--layers", default=None)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated prompt ids to run")
    ap.add_argument("--logit-chunk", type=int, default=64)
    ap.add_argument("--dump-layer", type=int, default=None)
    ap.add_argument("--dump-prompt", default="code_lru")
    ap.add_argument("--keep-embed", action="store_true")
    ap.add_argument("--ppl", default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--greedy", default=None)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--expert-cache", type=int, default=4, help="LRU experts per layer in greedy mode; 0 disables")
    return ap


def main():
    args = build_parser().parse_args()
    torch.set_num_threads(int(os.environ.get("MINIMAX_THREADS", os.cpu_count() or 1)))
    if args.greedy is not None:
        run_greedy(args)
    elif args.ppl is not None:
        run_ppl(args)
    else:
        run_prompts(args)


if __name__ == "__main__":
    main()
