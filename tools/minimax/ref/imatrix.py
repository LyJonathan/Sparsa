"""Calibration / imatrix collection for MiniMax-M3.1-Flash.

Accumulates, per quantizable linear, ``sum(x**2)`` over input columns and the
row count, exactly as llama.cpp's ``imatrix`` tool does, and writes a GGUF
imatrix in the format llama.cpp's ``quantize`` expects (spec section F):

    imatrix.datasets      ARRAY[STRING]
    imatrix.chunk_count   UINT32
    imatrix.chunk_size    UINT32
    imatrix.stats_schema  ARRAY[STRING]
    <tensor>.in_sum2      F32 [ne0, nmat]   (numpy (nmat, ne0))
    <tensor>.counts       F32 [1, nmat]     (numpy (nmat, 1))

``<tensor>`` is the GGUF model tensor name (``blk.i.attn_q.weight``,
``blk.i.ffn_gate_exps.weight``, ...).  Routed-expert tensors are stacked
expert-major, the same order as the converter's ``torch.stack(datas, dim=0)``.

The pass is layer-outer / chunk-inner: each layer's 128 routed experts are
dequantized once into RAM (~29 GB fp32) and reused for every calibration
chunk, so the 250 GB checkpoint is dequantized only once per full pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from model_ref import (
    DEFAULT_CONFIG,
    DEFAULT_MODEL_DIR,
    ReferenceModel,
    RSSMonitor,
    make_experts_resident,
)

DEFAULT_OUT = r"E:\Jon's Projects\AI\Strata Fork Minimax\Strata-MiniMax\_local\ref_out\imatrix-minimax-m3.1.gguf"

STATS_SCHEMA = [
    "sum_sq",
    "mean",
    "elements",
    "std_deviation",
    "skewness",
    "kurtosis",
    "gain",
    "h_norm",
    "l2_dist",
    "cossim",
    "pearson",
    "covariance",
]


class IMatrix:
    def __init__(self):
        self.entries: dict[str, dict] = {}

    def _entry(self, name: str, ne0: int, nmat: int):
        e = self.entries.get(name)
        if e is None:
            e = {
                "sum2": np.zeros((nmat, ne0), dtype=np.float32),
                "count": np.zeros((nmat,), dtype=np.int64),
                "ne0": ne0,
                "nmat": nmat,
            }
            self.entries[name] = e
        elif e["ne0"] != ne0 or e["nmat"] != nmat:
            raise ValueError(f"{name}: inconsistent shape {ne0}x{nmat} vs {e['ne0']}x{e['nmat']}")
        return e

    def add(self, name: str, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        e = self._entry(name, x.shape[-1], 1)
        x2 = x.to(torch.float32).pow(2).sum(0).numpy()
        e["sum2"][0] += x2
        e["count"][0] += x.shape[0]

    def add_expert(self, name: str, expert: int, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        e = self._entry(name, x.shape[-1], 128)
        x2 = x.to(torch.float32).pow(2).sum(0).numpy()
        e["sum2"][expert] += x2
        e["count"][expert] += x.shape[0]

    # -- GGUF ---------------------------------------------------------------

    def write(self, path: str, datasets, chunk_count, chunk_size):
        import gguf

        w = gguf.GGUFWriter(str(path), arch="minimax-m3")
        w.add_type("imatrix")
        w.add_array("imatrix.datasets", list(datasets))
        w.add_uint32("imatrix.chunk_count", int(chunk_count))
        w.add_uint32("imatrix.chunk_size", int(chunk_size))
        w.add_array("imatrix.stats_schema", STATS_SCHEMA)
        for name in sorted(self.entries):
            e = self.entries[name]
            w.add_tensor(name + ".in_sum2", e["sum2"])
            w.add_tensor(name + ".counts", e["count"].astype(np.float32).reshape(-1, 1))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return len(self.entries)


# ---------------------------------------------------------------------------
# hook wiring


def wire_layer(layer, L, im: IMatrix):
    """Attach pre-hooks to every quantized linear of one decoder layer."""
    hooks = []
    names = {
        "self_attn.q_proj": f"blk.{L}.attn_q.weight",
        "self_attn.k_proj": f"blk.{L}.attn_k.weight",
        "self_attn.v_proj": f"blk.{L}.attn_v.weight",
        "self_attn.o_proj": f"blk.{L}.attn_output.weight",
        "self_attn.indexer.q_proj": f"blk.{L}.indexer.q_proj.weight",
        "self_attn.indexer.k_proj": f"blk.{L}.indexer.k_proj.weight",
    }

    def mk(name):
        def hook(module, args):
            if args:
                im.add(name, args[0])

        return hook

    for attr, gname in names.items():
        mod = layer
        for part in attr.split("."):
            mod = getattr(mod, part)
        hooks.append(mod.register_forward_pre_hook(mk(gname)))

    def mk_two(name_a, name_b):
        def hook(module, args):
            if args:
                im.add(name_a, args[0])
                im.add(name_b, args[0])
            return None

        return hook

    if L < 3:
        dense = layer.mlp
        hooks.append(
            dense.gate_up_proj.register_forward_pre_hook(
                mk_two(f"blk.{L}.ffn_gate.weight", f"blk.{L}.ffn_up.weight")
            )
        )
        hooks.append(dense.down_proj.register_forward_pre_hook(mk(f"blk.{L}.ffn_down.weight")))
    else:
        shared = layer.mlp.shared_experts
        hooks.append(
            shared.gate_up_proj.register_forward_pre_hook(
                mk_two(f"blk.{L}.ffn_gate_shexp.weight", f"blk.{L}.ffn_up_shexp.weight")
            )
        )
        hooks.append(shared.down_proj.register_forward_pre_hook(mk(f"blk.{L}.ffn_down_shexp.weight")))
        experts = layer.mlp.experts
        experts.imatrix = im
        experts.imatrix_names = (
            f"blk.{L}.ffn_gate_exps.weight",
            f"blk.{L}.ffn_up_exps.weight",
            f"blk.{L}.ffn_down_exps.weight",
        )
    return hooks


def run(args):
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    torch.set_num_threads(args.threads)
    model = ReferenceModel(args.model_dir, dtype, args.config)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    text = Path(args.calibration).read_text(encoding="utf-8")
    ids_all = tok.encode(text, add_special_tokens=False).ids
    if args.max_tokens:
        ids_all = ids_all[: args.max_tokens]
    chunk = args.chunk_size
    chunks = [ids_all[i : i + chunk] for i in range(0, len(ids_all), chunk)]
    chunks = [c for c in chunks if len(c) >= 32]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    n_tokens = sum(len(c) for c in chunks)
    print(f"calibration: {n_tokens} tokens in {len(chunks)} chunks of <= {chunk}", flush=True)

    hidden = []
    for c in chunks:
        hidden.append(model.embed_ids(torch.tensor([c], dtype=torch.long)))
    positions = [torch.arange(len(c), dtype=torch.long).unsqueeze(0) for c in chunks]
    model._embed = None
    gc.collect()

    im = IMatrix()
    layers = list(range(args.layer_start, args.layer_end))
    timings = []
    t_all = time.time()
    rss = RSSMonitor().start()
    for L in layers:
        t0 = time.time()
        layer = model.make_layer(L)
        t_load = time.time() - t0
        if L >= 3 and not args.no_resident_experts:
            make_experts_resident(layer, L, model.store, dtype)
        hooks = wire_layer(layer, L, im)
        t1 = time.time()
        for i in range(len(chunks)):
            hidden[i] = model.run_layer(layer, hidden[i], positions[i])
        t_run = time.time() - t1
        for h in hooks:
            h.remove()
        del layer
        gc.collect()
        dt = time.time() - t0
        timings.append({"layer": L, "load_s": round(t_load, 2), "run_s": round(t_run, 2), "total_s": round(dt, 2)})
        print(
            f"layer {L:2d}: load {t_load:6.1f}s run {t_run:7.1f}s "
            f"({t_run / max(1, n_tokens) * 1000:6.2f} ms/token) elapsed {time.time()-t_all:7.0f}s",
            flush=True,
        )

    total = time.time() - t_all
    measured_layers = len(layers)
    per_layer = total / max(1, measured_layers)
    est_full = per_layer * 60
    print(
        f"\nmeasured {measured_layers} layers in {total:.1f}s -> {per_layer:.1f}s/layer; "
        f"projected full 60-layer pass: {est_full/3600:.2f} h for {n_tokens} tokens",
        flush=True,
    )

    result = {
        "calibration": str(args.calibration),
        "tokens": n_tokens,
        "chunks": len(chunks),
        "chunk_size": chunk,
        "dtype": args.dtype,
        "layers_measured": layers,
        "timings": timings,
        "measured_total_s": round(total, 1),
        "per_layer_s": round(per_layer, 1),
        "projected_full_pass_h": round(est_full / 3600, 2),
        "entries": len(im.entries),
        "peak_rss_gb": rss.stop(),
    }
    if args.estimate_only:
        print("estimate-only: no imatrix written")
    else:
        path = args.out
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        n = im.write(path, [Path(args.calibration).name], len(chunks), chunk)
        result["imatrix"] = str(path)
        result["entries_written"] = n
        print(f"wrote {path} ({n} entries)")
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tokenizer", default=r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\tokenizer.json")
    ap.add_argument("--calibration", default=str(Path(__file__).with_name("calibration.txt")))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=60)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--no-resident-experts", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
