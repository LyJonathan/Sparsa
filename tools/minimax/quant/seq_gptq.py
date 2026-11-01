"""Sequential GPTQ for MiniMax-M3.1's routed experts: each layer is calibrated on the already-quantized layers before it.

Per layer, on one GPU, over the chat-format calibration set (prep_calib_mm.py, 384 x 2048 tokens):
  1. the layer runs on the hidden states the quantized model so far produced (the PyTorch reference's layer: MSA,
     indexer, shared expert), with the MoE replaced by a capture of its normed input;
  2. routing is recomputed from that input; per expert, gate/up are GPTQ-rounded against sum(w^2 x x^T) plus a prior of
     the layer-wide Hessian, then down is rounded with the imatrix of its input through the *rounded* gate/up;
  3. the next layer's input is built from the rounded experts, so later layers absorb earlier rounding error.
The expert bytes are written per layer in GGUF order (expert-major, row-major superblocks) with the tensor types of a
template GGUF, so splice_experts.py can drop them into a copy of that GGUF.

GPTQ core (gptq(), the per-expert Hessian with a layer-wide prior, gate/up only, down by imatrix rounding) is ported from
Project Maya's tools/maya_quant/gptq.py (MIT, Copyright (c) 2026 Peasant Smith and the Project Maya contributors;
see THIRD_PARTY_NOTICES.md). New here: MiniMax's layers through our reference, the quantized-prefix sequential pass, and
the GGUF-order byte output.

    python seq_gptq.py --model E:/mm-src --calib D:/mm-ref/calib/out/calib.npy --template <uimx part 1>
        --out D:/mm-ref/v2 [--layers 3-59] [--n-seq 384] [--dev cuda:0]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
# On another machine: STRATA_REF_DIR (model_ref.py), STRATA_GGUF_PY (gguf-py) and STRATA_GGML_BASE (libggml-base.so).
sys.path.insert(0, os.environ.get("STRATA_REF_DIR", str(ROOT / "_local" / "ref_out" / "tools" / "ref")))
sys.path.insert(0, os.environ.get("STRATA_GGUF_PY", str(ROOT / "_baseline" / "llama.cpp" / "gguf-py")))
import gguf  # noqa: E402
import model_ref  # noqa: E402

BLOCK = 256
GGML_DLL = pathlib.Path(os.environ.get("STRATA_GGML_BASE",
                                       str(ROOT / "_baseline" / "llama.cpp" / "build" / "bin" / "ggml-base.dll")))


class _Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


class Codec:
    """ggml's quantizer and dequantizer for one type (ggml-base through ctypes; the calls release the GIL)."""

    def __init__(self, lib, t: int):
        self.lib, self.t = lib, t
        lib.ggml_quantize_init(t)
        tr = lib.ggml_get_type_traits(t).contents
        self.to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tr.to_float)
        self.bpb = int(lib.ggml_row_size(t, BLOCK))

    def quantize(self, src: np.ndarray, imat: np.ndarray) -> np.ndarray:
        """src float32 [rows, 256*k] -> uint8 [rows, k*bpb]"""
        rows, n = src.shape
        dst = np.empty((rows, n // BLOCK * self.bpb), dtype=np.uint8)
        self.lib.ggml_quantize_chunk(self.t, src.ctypes.data, dst.ctypes.data, 0, rows, n, imat.ctypes.data)
        return dst

    def dequantize(self, q: np.ndarray, n: int) -> np.ndarray:
        rows = q.shape[0]
        y = np.empty((rows, n), dtype=np.float32)
        self.to_float(q.ctypes.data, y.ctypes.data, rows * n)
        return y


def load_ggml():
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(GGML_DLL.parent))
    lib = ctypes.CDLL(str(GGML_DLL))
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    return lib


def gptq(W: torch.Tensor, H: torch.Tensor, codec: Codec, pool, damp: float):
    """W [R, n] fp32 (modified), H [n, n] fp32 (consumed) -> (uint8 [R, n/256*bpb], dequantized [R, n]). Maya's gptq()
    for one matrix: 256-column blocks, block importance diag(inv(Hinv_SS)), error pushed through chol(Hinv, upper)."""
    R, n = W.shape
    dev = W.device
    dg = H.diagonal()
    m = dg.mean()
    m = m if m > 0 else torch.ones_like(m)
    dead = dg <= 0
    dg += torch.where(dead, m, torch.zeros_like(dg)) + damp * m
    Lc = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Lc)
    del Lc, H
    U = torch.linalg.cholesky(Hinv, upper=True)
    del Hinv
    out = np.empty((R, n // BLOCK * codec.bpb), dtype=np.uint8)
    deq = torch.empty_like(W)
    Ib = torch.eye(BLOCK, device=dev)
    step = max(64, -(-R // (4 * pool._max_workers)))
    for b0 in range(0, n, BLOCK):
        b1 = b0 + BLOCK
        Uss = U[b0:b1, b0:b1]
        Uinv = torch.linalg.solve_triangular(Uss, Ib, upper=True)
        imp = Uinv.square().sum(-1).clamp(min=1e-20)
        imp = (imp / imp.mean()).cpu().numpy().astype(np.float32)
        Wb = W[:, b0:b1].contiguous().cpu().numpy()
        k0 = b0 // BLOCK * codec.bpb
        Qn = np.empty((R, BLOCK), dtype=np.float32)

        def one(r0):
            q = codec.quantize(np.ascontiguousarray(Wb[r0:r0 + step]), imp)
            out[r0:r0 + step, k0:k0 + codec.bpb] = q
            Qn[r0:r0 + step] = codec.dequantize(q, BLOCK)

        list(pool.map(one, range(0, R, step)))
        Qb = torch.from_numpy(Qn).to(dev)
        deq[:, b0:b1] = Qb
        if b1 < n:
            Err = torch.linalg.solve_triangular(Uss, W[:, b0:b1] - Qb, upper=True, left=False)
            W[:, b1:] -= Err @ U[b0:b1, b1:]
    return out, deq


def rtn(W: torch.Tensor, imat: np.ndarray, codec: Codec, pool):
    """Plain imatrix rounding of a whole matrix (llama.cpp's way) -> (bytes, dequantized)."""
    Wn = W.contiguous().cpu().numpy()
    R, n = Wn.shape
    step = max(64, -(-R // (4 * pool._max_workers)))
    out = np.empty((R, n // BLOCK * codec.bpb), dtype=np.uint8)
    deq = np.empty((R, n), dtype=np.float32)
    im = np.ascontiguousarray(imat.astype(np.float32))

    def one(r0):
        q = codec.quantize(np.ascontiguousarray(Wn[r0:r0 + step]), im)
        out[r0:r0 + step] = q
        deq[r0:r0 + step] = codec.dequantize(q, n)

    list(pool.map(one, range(0, R, step)))
    return out, torch.from_numpy(deq).to(W.device)


def template_types(path):
    """{(layer, 'gate'|'up'|'down'): ggml type id} from every split of a template GGUF."""
    first = pathlib.Path(path)
    stem = first.name.split("-00001-of-")[0]
    types = {}
    for f in sorted(first.parent.glob(stem + "-0*-of-*.gguf")):
        for t in gguf.GGUFReader(str(f)).tensors:
            p = t.name.split(".")
            if len(p) == 4 and p[2] in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                types[(int(p[1]), p[2].split("_")[1])] = int(t.tensor_type)
    return types


class Capture(torch.nn.Module):
    """Stands in for the MoE block: keeps its (normed) input and contributes nothing, so the layer returns h_mid."""

    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, h):
        self.x = h.detach()
        return torch.zeros_like(h)


def swiglu(gate, up, limit, alpha):
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1.0) * (gate * torch.sigmoid(gate * alpha))


def main():
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--template", help="GGUF whose expert types to use")
    ap.add_argument("--types-json", help='{"layer.gate|up|down": ggml type} instead of --template (remote_types.py)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-59")
    ap.add_argument("--n-seq", type=int, default=384)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--prior", type=float, default=16384.0)
    ap.add_argument("--damp", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--no-gptq", action="store_true", help="plain imatrix rounding (the baseline, same pipeline)")
    ap.add_argument("--xpath", help="where the layer-input memmap lives (default <out>/X.bin; ~10 GB at 384 seqs)")
    ap.add_argument("--gptq-down", action="store_true", help="GPTQ the down projections too (default: imatrix rounding)")
    ap.add_argument("--gptq-down-min", type=int, default=256, help="fewest routed tokens for a down GPTQ")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    lo, hi = (int(x) for x in a.layers.split("-"))
    ids = np.load(a.calib)[: a.n_seq]
    N, T = ids.shape

    ref = model_ref.ReferenceModel(a.model, dtype=torch.bfloat16, device=a.dev)
    cfg = ref.config
    Hd = cfg.hidden_size
    lib = load_ggml()
    codecs = {}
    if a.types_json:
        types = {(int(k.split(".")[0]), k.split(".")[1]): v for k, v in json.loads(open(a.types_json).read()).items()}
    else:
        types = template_types(a.template)
    pool = cf.ThreadPoolExecutor(a.threads)

    xpath = pathlib.Path(a.xpath) if a.xpath else out / "X.bin"   # the current layer's input, bf16 [N, T, H] (resume point)
    state_p = out / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"next_layer": 0}
    X = np.memmap(xpath, dtype=np.uint16, mode="r+" if xpath.exists() else "w+", shape=(N, T, Hd))
    if state["next_layer"] == 0:
        for s in range(N):
            e = ref.embed_ids(torch.from_numpy(ids[s:s + 1].astype(np.int64)))
            X[s] = e[0].to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
        X.flush()
    pos = torch.arange(T, dtype=torch.long, device=dev).unsqueeze(0)

    for L in range(max(lo, state["next_layer"]), hi + 1):
        t0 = time.time()
        layer = ref.make_layer(L)
        sparse = cfg.mlp_layer_types[L] == "sparse"
        if not sparse:
            for s in range(N):
                h = torch.from_numpy(np.array(X[s])).view(torch.bfloat16).to(dev).unsqueeze(0)
                y = ref.run_layer(layer, h, pos)
                X[s] = y[0].view(torch.int16).cpu().numpy().view(np.uint16)
            X.flush()
            state["next_layer"] = L + 1
            state_p.write_text(json.dumps(state))
            print(f"layer {L:2d} dense  {time.time() - t0:6.0f}s", flush=True)
            del layer
            continue

        mlp = layer.mlp
        cap = Capture()
        layer.mlp = cap
        xm = torch.empty(N * T, Hd, dtype=torch.bfloat16)          # MoE input (normed), CPU
        hmid = torch.empty(N * T, Hd, dtype=torch.bfloat16)        # residual after attention, CPU
        for s in range(N):
            h = torch.from_numpy(np.array(X[s])).view(torch.bfloat16).to(dev).unsqueeze(0)
            y = ref.run_layer(layer, h, pos)
            hmid[s * T:(s + 1) * T] = y[0].cpu()
            xm[s * T:(s + 1) * T] = cap.x[0].cpu()
        t_attn = time.time() - t0

        # routing (fp32 router, sigmoid + correction bias, top-k, normalized) and the layer-wide prior Hessian
        gw = mlp.gate.weight.float().to(dev)
        gb = mlp.gate.e_score_correction_bias.float().to(dev)
        E, K = cfg.num_local_experts, cfg.num_experts_per_tok
        idx_l, w_l = [], []
        Hbar = torch.zeros(Hd, Hd, device=dev)
        acc = torch.zeros(N * T, Hd, dtype=torch.float32)          # routed + shared output, CPU fp32
        sh = mlp.shared_experts.to(dev)
        for c0 in range(0, N * T, 32768):
            x = xm[c0:c0 + 32768].to(dev)
            sc = F.linear(x.float(), gw).sigmoid()
            i = torch.topk(sc + gb, K, dim=-1, sorted=False)[1]
            w = sc.gather(1, i)
            w = w / w.sum(-1, keepdim=True)
            idx_l.append(i.cpu())
            w_l.append(w.cpu())
            xf = x.float()
            Hbar += xf.t() @ xf
            with torch.no_grad():
                acc[c0:c0 + 32768] = sh(x).float().cpu()
        Hbar /= N * T
        idx, wts = torch.cat(idx_l), torch.cat(w_l)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=E).tolist()
        starts = np.concatenate([[0], np.cumsum(counts)]).tolist()
        tok_all, w_all = order // K, wts.reshape(-1)[order]
        scale = float(cfg.routed_scaling_factor)

        tq = {k: types[(L, k)] for k in ("gate", "up", "down")}
        for t in tq.values():
            if t not in codecs:
                codecs[t] = Codec(lib, t)
        cg, cu, cd = codecs[tq["gate"]], codecs[tq["up"]], codecs[tq["down"]]
        I = cfg.intermediate_size
        files = {k: open(out / f"L{L:02d}_{k}.bin", "wb") for k in ("gate", "up", "down")}
        err_num = err_den = 0.0
        provider = mlp.experts.provider
        for e in range(E):
            w1, w3, w2 = (t.float().to(dev) for t in provider(e))
            a_, b_ = starts[e], starts[e + 1]
            tk, ww = tok_all[a_:b_], w_all[a_:b_].to(dev)
            nt = len(tk)
            CH = 16384                                             # tokens per chunk (hot experts get 100K+)
            chunks = [(c0, min(c0 + CH, nt)) for c0 in range(0, nt, CH)]

            def xs(c0, c1):
                return xm[tk[c0:c1]].to(dev).float()

            if a.no_gptq or nt == 0:
                img = torch.ones(Hd, device=dev) if nt == 0 else sum(xs(c0, c1).square().sum(0) for c0, c1 in chunks) / nt
                qg, dg = rtn(w1, img.cpu().numpy(), cg, pool)
                qu, du = rtn(w3, img.cpu().numpy(), cu, pool)
            else:
                H = a.prior * float(ww.square().mean()) * Hbar
                for c0, c1 in chunks:
                    xw = xs(c0, c1) * ww[c0:c1, None]
                    H += xw.t() @ xw
                del xw
                qg, dg = gptq(w1.clone(), H.clone(), cg, pool, a.damp)
                qu, du = gptq(w3.clone(), H, cu, pool, a.damp)
            # --gptq-down: down gets GPTQ too, against the Hessian of its real input (the SwiGLU of the *rounded*
            # gate/up, routing-weighted), for experts with at least --gptq-down-min tokens; others keep imatrix rounding.
            use_gd = a.gptq_down and nt >= a.gptq_down_min and not a.no_gptq
            imd = torch.zeros(I, device=dev)
            Hdn = torch.zeros(I, I, device=dev) if use_gd else None
            for c0, c1 in chunks:
                x = xs(c0, c1)
                h = swiglu(x @ dg.t(), x @ du.t(), cfg.swiglu_limit, cfg.swiglu_alpha) * ww[c0:c1, None]
                imd += h.square().sum(0)
                if use_gd:
                    Hdn += h.t() @ h
                del h
            imd = (imd / nt).cpu().numpy() if nt else np.ones(I, np.float32)
            if use_gd:
                qd, dd = gptq(w2.clone(), Hdn, cd, pool, a.damp)
                del Hdn
            else:
                qd, dd = rtn(w2, imd, cd, pool)
            for k, q in (("gate", qg), ("up", qu), ("down", qd)):
                files[k].write(q.tobytes())
            for c0, c1 in chunks:
                x, wc = xs(c0, c1), ww[c0:c1, None]
                yq = (swiglu(x @ dg.t(), x @ du.t(), cfg.swiglu_limit, cfg.swiglu_alpha) @ dd.t()) * wc
                yr = (swiglu(x @ w1.t(), x @ w3.t(), cfg.swiglu_limit, cfg.swiglu_alpha) @ w2.t()) * wc
                err_num += float((yq - yr).square().sum())
                err_den += float(yr.square().sum())
                acc.index_add_(0, tk[c0:c1], (yq * scale).cpu())
                del x, yq, yr
        for f in files.values():
            f.close()

        del xm
        for s in range(N):                                         # per sequence: a whole-set fp32 temp is 19 GB
            r = slice(s * T, (s + 1) * T)
            X[s] = (hmid[r].float() + acc[r]).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        X.flush()
        state["next_layer"] = L + 1
        state.setdefault("err", {})[str(L)] = err_num / max(err_den, 1e-30)
        state_p.write_text(json.dumps(state))
        print(f"layer {L:2d} moe  attn {t_attn:5.0f}s  total {time.time() - t0:6.0f}s  "
              f"rel expert err {err_num / max(err_den, 1e-30):.4f}  types {tq}", flush=True)
        del layer, mlp, hmid, acc
        torch.cuda.empty_cache()
    print("SEQ_GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
