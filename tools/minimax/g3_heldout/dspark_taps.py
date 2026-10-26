"""DSpark tap dump: residual stream entering/leaving chosen layers, for one short
chat-templated prompt, at fp32 with the same reference math as the G3 logits.

Why: DSpark's drafter gets 0% acceptance inside Sparsa, which smells like a wrong
tap choice or a wrong position convention rather than a bad drafter.  This dump
gives an independent fingerprint of the main model's residual stream at five
depths, so the engine's own tap can be diffed against it directly.

Outputs, under ``--out``:
    token_ids.npy               int32   [T]
    layer{L:02d}_enter.npy      float32 [T, 6144]   input to layer L's first norm
    layer{L:02d}_leave.npy      float32 [T, 6144]   after layer L's FFN residual add
    final_hidden.npy            float32 [T, 6144]   after the final Gemma norm
    logits_last.npy             float32 [1, 200064] next-token logits at the last position
    top1_all.npy                int32   [T]         argmax at every position (sanity)
    meta.json

``enter[L]`` is exactly what the layer receives and ``leave[L]`` is exactly what
the next layer receives, so ``enter[L+1]`` must equal ``leave[L]`` bit for bit;
the script asserts that, which also proves the taps are not off by one.

Usage:
    python dspark_taps.py --prompt "Explain why the sky is blue." \
        --layers 3,17,31,45,59 --out <dir>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from chat import apply_chat_template
from model_ref import DEFAULT_CONFIG, DEFAULT_MODEL_DIR, ReferenceModel, RSSMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Explain why the sky is blue.")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--config", default=os.environ.get("MINIMAX_CONFIG", ""))
    ap.add_argument("--tokenizer", default=os.environ.get("MINIMAX_TOKENIZER", ""))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--layers", default="3,17,31,45,59")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logit-chunk", type=int, default=64)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("MINIMAX_THREADS", os.cpu_count() or 1)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taps = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    text = apply_chat_template([{"role": "user", "content": args.prompt}])
    tok_path = Path(args.tokenizer) if args.tokenizer else Path(args.model_dir) / "tokenizer.json"
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"prompt tokens: {len(ids)}  (rendered {len(text)} chars)", flush=True)

    model = ReferenceModel(args.model_dir, dtype, args.config, device=args.device)
    t = torch.tensor([ids], dtype=torch.long)
    pos = torch.arange(len(ids), dtype=torch.long, device=model.device).unsqueeze(0)
    hidden = model.embed_ids(t)
    np.save(out / "token_ids.npy", np.asarray(ids, dtype=np.int32))

    rss = RSSMonitor().start()
    timings = {}
    prev_out = None
    stats = {}
    t0 = time.time()
    for L in range(model.config.num_hidden_layers):
        layer = model.make_layer(L)
        tl = time.time()
        # what the layer receives is exactly what the previous layer returned: the
        # tap is the residual stream, not something recomputed for the dump.
        assert prev_out is None or prev_out is hidden, f"tap bookkeeping broke at layer {L}"
        entering = hidden
        if L in taps:
            np.save(out / f"layer{L:02d}_enter.npy", entering[0].to(torch.float32).cpu().numpy())
        hidden = model.run_layer(layer, hidden, pos)
        prev_out = hidden
        if L in taps:
            np.save(out / f"layer{L:02d}_leave.npy", hidden[0].to(torch.float32).cpu().numpy())
            e = entering[0].to(torch.float32)
            l = hidden[0].to(torch.float32)
            stats[f"layer_{L}"] = {
                "enter_mean": float(e.mean()), "enter_std": float(e.std()),
                "leave_mean": float(l.mean()), "leave_std": float(l.std()),
                "leave_vs_enter_max_abs": float((l - e).abs().max()),
                "leave_vs_enter_cos": float(torch.nn.functional.cosine_similarity(
                    e.flatten(), l.flatten(), dim=0)),
            }
            print(f"  tapped layer {L:2d} ({time.time() - tl:6.2f}s) "
                  f"enter_std {stats[f'layer_{L}']['enter_std']:.4f} "
                  f"leave_std {stats[f'layer_{L}']['leave_std']:.4f} "
                  f"cos {stats[f'layer_{L}']['leave_vs_enter_cos']:.4f}", flush=True)
        del layer
        gc.collect()
        timings[f"layer_{L}"] = time.time() - tl
    print(f"layers: {time.time() - t0:.1f}s", flush=True)

    fh = model.final_hidden(hidden)[0]
    np.save(out / "final_hidden.npy", fh.to(torch.float32).cpu().numpy())
    last = model.logits(fh[-1:], chunk=1)
    np.save(out / "logits_last.npy", last.cpu().numpy())
    top1 = []
    for i in range(0, fh.shape[0], args.logit_chunk):
        lg = model.logits(fh[i : i + args.logit_chunk], chunk=args.logit_chunk)
        top1.append(lg.argmax(-1))
    top1 = torch.cat(top1).to(torch.int32).cpu().numpy()
    np.save(out / "top1_all.npy", top1)

    meta = {
        "prompt": args.prompt,
        "rendered_text": text,
        "n_tokens": len(ids),
        "dtype": args.dtype,
        "model_dir": args.model_dir,
        "tap_layers": taps,
        "tap_kind": "enter = input to the layer's first norm; leave = after the FFN residual add",
        "shapes": {
            "layerXX_enter/leave": [len(ids), int(model.config.hidden_size)],
            "logits_last": [1, int(model.config.vocab_size)],
        },
        "last_position": len(ids) - 1,
        "layer_stats": stats,
        "last_top1": int(top1[-1]),
        "last_top1_text": tok.decode([int(top1[-1])]),
        "peak_rss_gb": rss.stop(),
        "runtime_s": time.time() - t0,
        "torch": torch.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_tokens", "last_top1", "last_top1_text",
                                           "peak_rss_gb", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main()
