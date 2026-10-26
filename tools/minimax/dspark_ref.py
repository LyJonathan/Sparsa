"""tools/minimax/dspark_ref.py - a numpy reading of MiniMax-M3.1-Flash's DSpark draft head.

There is no reference drafter implementation in the checkpoint, only weights and a config, so this script is the
second opinion the engine is checked against: it recomputes the drafter's forward from the checkpoint's own tensors
(MXFP8/BF16 decoded the way tools/minimax/dspark_pack.py does) on a block of mask tokens.

The block, in the convention the verify pass uses (src/minimax/engine.cpp, decode_spec):
  row 0            the anchor token (the target's own next token) at position p0
  rows 1..block    the mask/noise token at p0+1 .. p0+block
  logits of row j  predict the token at p0+j+1, so draft j comes from row j's logits (j = 0 .. block-1)
  the Markov head's `prev` for row j is the token at p0+j: the anchor for j = 0, draft j-1 otherwise
This script scores exactly that, and the shifted reading (draft j from row j+1) beside it, because the shift is what
the engine did before this script existed.

The score is the greedy acceptance against the FP32 reference run's own top-1 (`top1_all.npy`): draft j is compared
with the target's argmax at position p0+j, which IS the token the target would emit at p0+j+1.  Row 0's draft is
the exact single-draft case (both sides share the history up to p0).

Conventions, all from the checkpoint's config.json (text_config):
  hidden 6144, 5 layers, 64 q heads / 4 kv heads (GQA), head_dim 128, partial RoPE over the first 64 dims with
  theta 1e7 (half-split pairs, HF rotate_half), per-head q/k RMSNorm, use_gemma_norm -> every RMSNorm multiplies by
  (1 + w), dense FFN 12288 with swigluoai(alpha 1.702, limit 7), non-causal attention across the block, the target's
  own lm_head, then the rank-256 Markov head.

    python tools/minimax/dspark_ref.py --positions 24 [--dumps <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dspark_pack as dp  # noqa: E402  (the checkpoint decoder the packer uses, and its own tests cover)

PREFIX = "language_model.model.dspark."
ROOT = HERE.parents[1]


def rms_gemma(x, w, eps=1e-6):
    """Gemma RMSNorm: x / sqrt(mean(x^2) + eps) * (1 + w)."""
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * inv) * (1.0 + w)


def qk_head_norm(x, w, heads, eps=1e-6):
    """Per-head RMSNorm over head_dim with (1 + w), on a [rows][heads * head_dim] row."""
    r = x.shape[0]
    hd = w.shape[0]
    y = x.reshape(r, heads, hd)
    inv = 1.0 / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + eps)
    return ((y * inv) * (1.0 + w)).reshape(r, heads * hd)


def rope(x, pos, heads, head_dim, n_rot, theta):
    """HF partial RoPE: rotate_half pairs (j, j + n_rot/2), on the first n_rot dims of every head."""
    half = n_rot // 2
    inv = theta ** (-np.arange(half, dtype=np.float64) * 2.0 / n_rot)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    y = x.reshape(x.shape[0], heads, head_dim).copy()
    a = y[:, :, :half].copy()
    b = y[:, :, half:n_rot].copy()
    y[:, :, :half] = a * c[:, None, :] - b * s[:, None, :]
    y[:, :, half:n_rot] = b * c[:, None, :] + a * s[:, None, :]
    return y.reshape(x.shape[0], heads * head_dim)


def swigluoai(g, u, alpha, limit):
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    return (u + 1.0) * g * (1.0 / (1.0 + np.exp(-alpha * g)))


class Drafter:
    def __init__(self, ck, cfg):
        self.ck = ck
        tc = cfg.get("text_config", cfg)
        self.E = int(tc["hidden_size"])
        self.L = int(tc["num_hidden_layers"])
        self.H = int(tc["num_attention_heads"])
        self.HK = int(tc["num_key_value_heads"])
        self.D = int(tc["head_dim"])
        self.n_rot = int(tc.get("rotary_dim", self.D))
        self.I = int(tc.get("dense_intermediate_size", tc.get("dspark_ffn_hidden_size")))
        self.block = int(tc["dspark_block_size"])
        self.mask = int(tc["dspark_noise_token_id"])
        self.alpha = float(tc.get("swiglu_alpha", 1.702))
        self.limit = float(tc.get("swiglu_limit", 7.0))
        self.theta = float(tc.get("rope_theta", 1e7))
        self.eps = float(tc.get("rms_norm_eps", 1e-6))
        self.taps = [int(x) for x in tc["dspark_target_layer_ids"]]

    def g(self, name):
        return self.ck.get(PREFIX + name + ".weight")

    def load(self):
        self.fc = self.g("fc")                                     # [E][n_in]
        self.hidden_norm = self.g("hidden_norm")
        self.final_norm = self.g("final_norm")
        self.mw1 = self.g("markov_head.markov_w1")                 # [vocab][rank]
        self.mw2 = self.g("markov_head.markov_w2")
        self.layers = []
        for l in range(self.L):
            p = f"layers.{l}.decoder_layer."
            self.layers.append({
                "in_norm": self.g(p + "input_layernorm"),
                "post_norm": self.g(p + "post_attention_layernorm"),
                "q": self.g(p + "self_attn.q_proj"),
                "k": self.g(p + "self_attn.k_proj"),
                "v": self.g(p + "self_attn.v_proj"),
                "o": self.g(p + "self_attn.o_proj"),
                "q_norm": self.g(p + "self_attn.q_norm"),
                "k_norm": self.g(p + "self_attn.k_norm"),
                "gate": self.g(p + "mlp.gate_proj"),
                "up": self.g(p + "mlp.up_proj"),
                "down": self.g(p + "mlp.down_proj"),
            })

    def context(self, taps):
        """The fused context (hidden_norm(fc(taps))), once, shared by every drafter layer."""
        return rms_gemma(taps @ self.fc.T, self.hidden_norm, self.eps)

    def context_kv(self, ctx, head):
        rows = ctx.shape[0]
        k = qk_head_norm(ctx @ head["k"].T, head["k_norm"], self.HK, self.eps)
        k = rope(k, np.arange(rows), self.HK, self.D, self.n_rot, self.theta)
        v = ctx @ head["v"].T
        return k.reshape(rows, self.HK, self.D), v.reshape(rows, self.HK, self.D)

    def run_block(self, emb, ring_k, ring_v, pos0):
        """One drafter pass over the block rows; returns the final-normed hidden of every row."""
        H, HK, D = self.H, self.HK, self.D
        rows = emb.shape[0]
        pos = np.arange(pos0, pos0 + rows)
        x = emb.copy()
        for l in range(self.L):
            L = self.layers[l]
            h = rms_gemma(x, L["in_norm"], self.eps)
            q = rope(qk_head_norm(h @ L["q"].T, L["q_norm"], H, self.eps), pos, H, D, self.n_rot, self.theta)
            k = rope(qk_head_norm(h @ L["k"].T, L["k_norm"], HK, self.eps), pos, HK, D, self.n_rot, self.theta)
            v = h @ L["v"].T
            k = np.concatenate([ring_k[l], k.reshape(rows, HK, D)], axis=0)
            v = np.concatenate([ring_v[l], v.reshape(rows, HK, D)], axis=0)
            q = q.reshape(rows, H, D)
            kv = np.repeat(np.arange(HK), H // HK)                 # head -> its kv head
            s = np.einsum("thd,khd->thk", q, k[:, kv, :]) / np.sqrt(D)
            s = s - s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=-1, keepdims=True)
            o = np.einsum("thk,khd->thd", w, v[:, kv, :]).reshape(rows, H * D)
            x = x + o @ L["o"].T
            h = rms_gemma(x, L["post_norm"], self.eps)
            h = swigluoai(h @ L["gate"].T, h @ L["up"].T, self.alpha, self.limit)
            x = x + h @ L["down"].T
        return rms_gemma(x, self.final_norm, self.eps)

    def run_context(self, ctx):
        """The context stream through all five layers (causal, the way a KV cache is built): this returns the
        per-layer K/V rows the block attends to, one set per position."""
        H, HK, D = self.H, self.HK, self.D
        rows = ctx.shape[0]
        pos = np.arange(rows)
        kv = np.repeat(np.arange(HK), H // HK)
        x = ctx.copy()
        out = []
        for l in range(self.L):
            L = self.layers[l]
            h = rms_gemma(x, L["in_norm"], self.eps)
            q = rope(qk_head_norm(h @ L["q"].T, L["q_norm"], H, self.eps), pos, H, D, self.n_rot, self.theta)
            k = rope(qk_head_norm(h @ L["k"].T, L["k_norm"], HK, self.eps), pos, HK, D, self.n_rot, self.theta)
            v = h @ L["v"].T
            kb = k.reshape(rows, HK, D)
            vb = v.reshape(rows, HK, D)
            out.append((kb, vb))
            q = q.reshape(rows, H, D)
            s = np.einsum("thd,khd->thk", q, kb[:, kv, :]) / np.sqrt(D)
            s = np.where(np.tril(np.ones((rows, rows), bool))[:, None, :], s, -np.inf)   # causal
            s = s - s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=-1, keepdims=True)
            o = np.einsum("thk,khd->thd", w, vb[:, kv, :]).reshape(rows, H * D)
            x = x + o @ L["o"].T
            h = rms_gemma(x, L["post_norm"], self.eps)
            h = swigluoai(h @ L["gate"].T, h @ L["up"].T, self.alpha, self.limit)
            x = x + h @ L["down"].T
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(ROOT / "Strata-MiniMax" / "_local" / "ref_out" / "dspark_taps"))
    ap.add_argument("--ckpt", default=r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash")
    ap.add_argument("--dumps", default=r"E:\Jon's Projects\AI\Strata Fork Minimax\wt-stdin\dump_taps")
    ap.add_argument("--positions", type=int, default=24, help="how many anchors to score")
    ap.add_argument("--drafts", type=int, default=4, help="draft positions compared per anchor")
    a = ap.parse_args()

    ref = pathlib.Path(a.ref)
    ckpt = pathlib.Path(a.ckpt)
    config = json.loads((ckpt / "dspark" / "config.json").read_text(encoding="utf-8"))
    ck = dp.Checkpoint(ckpt / "dspark")
    dr = Drafter(ck, config)
    dr.load()
    print(f"drafter: {dr.L} layers, E={dr.E}, H={dr.H}/{dr.HK}, D={dr.D} rot={dr.n_rot}, ffn={dr.I}, "
          f"block={dr.block}, mask={dr.mask}, taps={dr.taps}", flush=True)

    ids = np.load(ref / "token_ids.npy").astype(np.int64)
    top1 = np.load(ref / "top1_all.npy").astype(np.int64)
    taps_ref = np.concatenate([np.load(ref / f"layer{t:02d}_enter.npy") for t in dr.taps], axis=1)
    print(f"ref taps {taps_ref.shape}, ids {ids.shape}, top1 {top1.shape}", flush=True)

    # the target's own head + embedding, straight out of the base checkpoint
    main_ck = dp.Checkpoint(ckpt)
    emb = main_ck.get("language_model.model.embed_tokens.weight")       # [vocab][E]
    head_w = main_ck.get("language_model.lm_head.weight")               # [vocab][E]
    print(f"embed {emb.shape} head {head_w.shape}", flush=True)

    def logits_of(fn):
        out = np.empty((fn.shape[0], head_w.shape[0]), np.float32)
        for r in range(fn.shape[0]):
            for v0 in range(0, head_w.shape[0], 16384):
                v1 = min(v0 + 16384, head_w.shape[0])
                out[r, v0:v1] = head_w[v0:v1] @ fn[r]
        return out

    # sanity: the ref's final hidden through the target head reproduces the ref's logits of the last position
    fh = np.load(ref / "final_hidden.npy")
    lg = logits_of(fh[-1:])
    rl = np.load(ref / "logits_last.npy").reshape(1, -1)
    cos = float(lg.ravel() @ rl.ravel() / (np.linalg.norm(lg) * np.linalg.norm(rl)))
    print(f"check: final_hidden -> head vs ref logits_last cos={cos:.6f} argmax {int(lg.argmax())} vs "
          f"{int(rl.argmax())}", flush=True)

    def run(taps, label, n_anchor):
        ctx = dr.context(taps)
        kvs = [dr.context_kv(ctx, L) for L in dr.layers]          # every layer's K/V from the fused context
        R = dr.block                                               # queries = anchor + (block-1) masks, one draft per row
        anchors = list(range(8, len(ids)))
        step = max(1, len(anchors) // n_anchor)
        picked = anchors[::step][:n_anchor]
        hits = {j: 0 for j in range(a.drafts)}
        for p0 in picked:
            ring_k = [kv[0][:p0] for kv in kvs]
            ring_v = [kv[1][:p0] for kv in kvs]
            anchor = int(top1[p0 - 1])
            cols = np.concatenate([[anchor], np.full(R - 1, dr.mask)]).astype(np.int64)
            fn = dr.run_block(emb[cols].copy(), ring_k, ring_v, p0)
            lg = logits_of(fn)
            # the Markov head, walked row by row: prev is the token AT that row's position
            prev = anchor
            for j in range(R):
                bias = dr.mw2 @ dr.mw1[prev]   # bias[v] = <w2[v], w1[prev]>
                pick = int(np.argmax(lg[j] + bias))
                if j < a.drafts and p0 + j < len(top1):
                    hits[j] += int(pick == int(top1[p0 + j]))
                prev = pick
        n = len(picked)
        s = ", ".join(f"draft {j}: {hits[j]}/{n} ({100.0 * hits[j] / n:.1f} %)" for j in range(a.drafts))
        print(f"{label} | rows={R} row j -> draft j: {s}", flush=True)

    run(taps_ref, "FP32 ref taps", a.positions)

    if a.dumps and os.path.isdir(a.dumps):
        alt = [np.load(pathlib.Path(a.dumps) / f"dump-L{t}" / "residual_in.npy") for t in dr.taps]
        run(np.concatenate(alt, axis=1), "engine IQ2_XS taps", a.positions)


if __name__ == "__main__":
    main()
