"""FP32/bf16 CPU reference forward for MiniMax-M3.1-Flash (text tower only).

Design
------
The checkpoint cannot be loaded by ``from_pretrained`` (see spec D.10: the
``mxfp8`` quantizer expects fp32 scales with 128x128 blocks and the ``nvfp4``
quantizer refuses pre-quantized files).  Instead the model is rebuilt from an
explicitly constructed ``MiniMaxM3VLTextConfig`` and the transformers v5.20.2
``minimax_m3_vl`` module classes, with one substitution:

    ``MiniMaxM3VLExperts`` is replaced by :class:`StreamingExperts`, which
    dequantizes *one routed expert at a time* from the safetensors shards and
    applies it to all tokens routed to it.  The class computes exactly the
    same expression as the upstream forward (which also loops per hit expert),
    but never holds the 29 GB fp32 expert bank of a layer in RAM.

Everything else (attention, indexer, router, shared expert, dense MLP, norms,
RoPE) is the upstream module, unmodified.  Layers are built one at a time and
freed after use, so a full 60-layer pass keeps only one layer's dense weights
(about 0.7 GB fp32) plus one dequantized routed expert (0.23 GB) resident.

The only intentional behavioural deviation from the HF eager path is that we
pass ``attention_mask=None`` into the decoder layer: every layer of this
checkpoint is sparse (``layer_types == ["minimax_m3_sparse"] * 60``) and
``MiniMaxM3VLIndexer.build_block_mask`` rebuilds the token-level causal mask
itself when no mask is supplied -- identical for an unpadded, batch-1 prefill,
and it avoids materialising a second [B, 1, T, T] mask.  Padding is not
supported here (all prompts are run independently, batch 1).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dequant import dequant_mxfp8, dequant_nvfp4

DEFAULT_MODEL_DIR = os.environ.get(
    "MINIMAX_MODEL_DIR", r"E:\Jon's Projects\AI\Strata Fork Minimax\models\MiniMax-M3.1-Flash"
)
DEFAULT_INDEX = os.environ.get(
    "MINIMAX_INDEX",
    r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\model.safetensors.index.json",
)
DEFAULT_CONFIG = os.environ.get(
    "MINIMAX_CONFIG", r"E:\Jon's Projects\AI\Strata Fork Minimax\_ref\minimax\config.json"
)

LAYER_PREFIX = "language_model.model.layers."


def build_text_config(config_path: str = DEFAULT_CONFIG):
    """Build MiniMaxM3VLTextConfig from the checkpoint's ``text_config``."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    text = dict(raw["text_config"])
    rope_theta = text.pop("rope_theta")
    partial = text.pop("partial_rotary_factor")
    # The checkpoint keeps rope_theta / partial_rotary_factor at the top level;
    # the native config reads config.rope_parameters.
    text["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial,
    }
    cfg = MiniMaxM3VLTextConfig(**text)
    cfg._attn_implementation = "eager"
    cfg.attention_dropout = 0.0
    return cfg


class ShardStore:
    """Read tensors by checkpoint name through the safetensors index."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, index_path: str | None = None):
        self.model_dir = Path(model_dir)
        index_path = Path(index_path) if index_path else self.model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            # the checkpoint dir may still be downloading its metadata files;
            # the copy in _ref was fetched from the same repo revision.
            index_path = Path(DEFAULT_INDEX)
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}")
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self._handles = {}
        self.missing_shards = set()
        for name in self.weight_map.values():
            if not (self.model_dir / name).exists():
                self.missing_shards.add(name)

    def shard_of(self, name: str) -> str:
        return self.weight_map[name]

    def available(self, name: str) -> bool:
        return name in self.weight_map and self.shard_of(name) not in self.missing_shards

    def _handle(self, shard: str):
        from safetensors import safe_open

        h = self._handles.get(shard)
        if h is None:
            path = self.model_dir / shard
            h = safe_open(str(path), framework="pt", device="cpu")
            self._handles[shard] = h
        return h

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.shard_of(name)).get_tensor(name)

    # -- dequantizing helpers -------------------------------------------------

    def mxfp8(self, name: str, dtype=torch.float32) -> torch.Tensor:
        w = self.get(name)
        s = self.get(name[: -len("weight")] + "weight_scale_inv")
        return dequant_mxfp8(w, s, dtype)

    def nvfp4(self, base: str, dtype=torch.float32) -> torch.Tensor:
        """``base`` is the projection prefix, e.g. ``...experts.3.w1.``."""
        return dequant_nvfp4(
            self.get(base + "weight_packed"),
            self.get(base + "weight_scale"),
            self.get(base + "weight_global_scale"),
            dtype,
        )

    def close(self):
        self._handles.clear()


# ---------------------------------------------------------------------------
# transformers patches


class RSSMonitor:
    """Poll this process's RSS so the report can quote a real peak."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = False
        self._thread = None

    def _run(self):
        import time

        try:
            import psutil

            proc = psutil.Process()
        except Exception:
            return
        while not self._stop:
            self.peak_bytes = max(self.peak_bytes, proc.memory_info().rss)
            time.sleep(self.interval)

    def start(self):
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        return round(self.peak_bytes / 1e9, 2)


class DumpContext:
    """Per-forward capture buffer, filled by the patched modules."""

    active: "DumpContext | None" = None

    def __init__(self):
        self.data = {}

    def put(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach()
        self.data[key] = value


class StreamingExperts(torch.nn.Module):
    """Drop-in for MiniMaxM3VLExperts that dequantizes experts on demand.

    Forward is a copy of the upstream ``MiniMaxM3VLExperts.forward``
    (per-hit-expert loop, one_hot mask, index_add_), with the two 3-D parameter
    banks replaced by ``provider(expert_idx) -> (gate, up, down)``.
    """

    provider = None  # set by the loader (class-level default, overridden per instance)
    layer_idx = -1
    imatrix = None
    imatrix_names = None

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.limit = config.swiglu_limit
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        dump = DumpContext.active
        for expert_idx in hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            x = hidden_states[token_idx]
            if self.imatrix is not None:
                self.imatrix.add_expert(self.imatrix_names[0], expert_idx, x)
                self.imatrix.add_expert(self.imatrix_names[1], expert_idx, x)
            w1, w3, w2 = self.provider(expert_idx)
            gate_up_w = torch.cat([w1, w3], dim=0)
            gate_up = F.linear(x, gate_up_w)
            current = self._apply_gate(gate_up)
            if dump is not None:
                dump.put(f"expert_{expert_idx:03d}_gate_up", gate_up)
                dump.put(f"expert_{expert_idx:03d}_hidden", current)
            if self.imatrix is not None:
                self.imatrix.add_expert(self.imatrix_names[2], expert_idx, current)
            current = F.linear(current, w2) * top_k_weights[token_idx, top_k_pos, None]
            if dump is not None:
                dump.put(f"expert_{expert_idx:03d}_out", current)
                dump.put(f"expert_{expert_idx:03d}_token_idx", token_idx)
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return (up + 1.0) * glu


def install_patches():
    """Patch the transformers module so layer construction uses our classes."""
    import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl as mm

    if getattr(mm, "_strata_ref_patched", False):
        return mm

    mm._strata_ref_experts_orig = mm.MiniMaxM3VLExperts
    mm.MiniMaxM3VLExperts = StreamingExperts
    mm._strata_ref_indexer_orig = mm.MiniMaxM3VLIndexer
    mm.MiniMaxM3VLIndexer = make_ref_indexer(mm)

    orig_eager = mm.eager_attention_forward
    mm._strata_ref_eager_orig = orig_eager

    def eager_capture(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out, weights = orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        if DumpContext.active is not None:
            d = DumpContext.active
            d.put("attn_core_out", out)
        return out, weights

    mm.eager_attention_forward = eager_capture
    mm.ALL_ATTENTION_FUNCTIONS["eager"] = eager_capture

    orig_rope = mm.apply_rotary_pos_emb

    def rope_capture(q, k, cos, sin, unsqueeze_dim=1):
        qe, ke = orig_rope(q, k, cos, sin, unsqueeze_dim)
        if DumpContext.active is not None:
            d = DumpContext.active
            if q.shape[1] == module_num_heads[0]:
                d.put("q_after_rope", qe)
                d.put("k_after_rope", ke)
            else:
                d.put("index_q_after_rope", qe)
                d.put("index_k_after_rope", ke)
        return qe, ke

    mm.apply_rotary_pos_emb = rope_capture
    mm._strata_ref_rope_orig = orig_rope
    mm._strata_ref_patched = True
    return mm


# the rope patch needs to know how many heads the *main* branch has
module_num_heads = [64]


def make_ref_indexer(mm):
    class RefIndexer(mm._strata_ref_indexer_orig):
        """Upstream indexer forward, plus capture of the block scores."""

        def forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
            batch, q_len, _ = hidden_states.shape
            idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
            idx_q = self.q_norm(idx_q).transpose(1, 2)
            idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
            idx_k = self.k_norm(idx_k).transpose(1, 2)
            cos, sin = position_embeddings
            idx_q, idx_k = mm.apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])

            if past_key_values is not None:
                idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)

            k_len = idx_k.shape[2]
            num_key_blocks = -(-k_len // self.block_size)
            pad = num_key_blocks * self.block_size - k_len

            scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
            k_positions = torch.arange(k_len, device=idx_q.device)
            token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
            scores = scores.masked_fill(token_future, float("-inf"))
            if pad:
                scores = F.pad(scores, (0, pad), value=float("-inf"))
            scores_blocks = scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size)
            block_scores = scores_blocks.amax(dim=-1)

            q_block = position_ids // self.block_size
            if self.local_blocks > 0:
                local = torch.arange(self.local_blocks, device=idx_q.device)
                local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
                local_idx = local_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                block_scores = block_scores.clone()
                block_scores.scatter_(-1, local_idx, float("inf"))

            topk = min(self.topk_blocks, num_key_blocks)
            topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
            out = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
            if DumpContext.active is not None:
                d = DumpContext.active
                d.put("index_scores", scores)
                d.put("index_block_scores", block_scores)
                d.put("index_block_topk_scores", topk_scores)
                d.put("index_block_indices", out)
            return out

    RefIndexer.__name__ = "MiniMaxM3VLIndexer"
    return RefIndexer


# ---------------------------------------------------------------------------
# layer construction / weight loading


def _module_suffixes(layer_idx, is_moe):
    """(checkpoint suffix, module attribute path, kind) for one layer."""
    rows = [
        ("input_layernorm.weight", "input_layernorm.weight", "plain"),
        ("post_attention_layernorm.weight", "post_attention_layernorm.weight", "plain"),
        ("self_attn.q_proj.weight", "self_attn.q_proj.weight", "mxfp8"),
        ("self_attn.k_proj.weight", "self_attn.k_proj.weight", "mxfp8"),
        ("self_attn.v_proj.weight", "self_attn.v_proj.weight", "mxfp8"),
        ("self_attn.o_proj.weight", "self_attn.o_proj.weight", "mxfp8"),
        ("self_attn.q_norm.weight", "self_attn.q_norm.weight", "plain"),
        ("self_attn.k_norm.weight", "self_attn.k_norm.weight", "plain"),
        ("self_attn.index_q_proj.weight", "self_attn.indexer.q_proj.weight", "mxfp8"),
        ("self_attn.index_k_proj.weight", "self_attn.indexer.k_proj.weight", "mxfp8"),
        ("self_attn.index_q_norm.weight", "self_attn.indexer.q_norm.weight", "plain"),
        ("self_attn.index_k_norm.weight", "self_attn.indexer.k_norm.weight", "plain"),
    ]
    if not is_moe:
        rows += [
            ("mlp.gate_proj.weight", "mlp.gate_up_proj.weight[0:inter]", "mxfp8"),
            ("mlp.up_proj.weight", "mlp.gate_up_proj.weight[inter:2*inter]", "mxfp8"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight", "mxfp8"),
        ]
    else:
        rows += [
            ("block_sparse_moe.gate.weight", "mlp.gate.weight", "plain_f32"),
            ("block_sparse_moe.e_score_correction_bias", "mlp.gate.e_score_correction_bias", "plain_f32"),
            ("block_sparse_moe.shared_experts.gate_proj.weight", "mlp.shared_experts.gate_up_proj.weight[0:inter]", "mxfp8"),
            ("block_sparse_moe.shared_experts.up_proj.weight", "mlp.shared_experts.gate_up_proj.weight[inter:2*inter]", "mxfp8"),
            ("block_sparse_moe.shared_experts.down_proj.weight", "mlp.shared_experts.down_proj.weight", "mxfp8"),
        ]
    return rows


def _get_submodule(root, path):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def load_layer_weights(layer, layer_idx, store: ShardStore, dtype=torch.float32, config=None):
    cfg = config if config is not None else layer.config
    is_moe = cfg.mlp_layer_types[layer_idx] == "sparse"
    prefix = f"{LAYER_PREFIX}{layer_idx}."
    missing = []

    def deq(kind, key):
        if kind == "mxfp8":
            return store.mxfp8(key, dtype)
        t = store.get(key)
        return t.to(torch.float32) if kind == "plain_f32" else t.to(dtype)

    def set_param(param, value):
        with torch.no_grad():
            param.data = value

    inter = cfg.dense_intermediate_size
    for ckpt_suffix, module_path, kind in _module_suffixes(layer_idx, is_moe):
        key = prefix + ckpt_suffix
        if not store.available(key):
            missing.append(key)
            continue
        value = deq(kind, key)
        if module_path.endswith("]"):
            path, rng = module_path.split("[")
            lo_s, hi_s = rng[:-1].split(":")

            def bound(s, default):
                if s == "":
                    return default
                if s == "inter":
                    return inter
                if s == "2*inter":
                    return 2 * inter
                return int(s)

            lo, hi = bound(lo_s, 0), bound(hi_s, 2 * inter)
            path = path[: -len(".weight")] if path.endswith(".weight") else path
            inter_size = (
                cfg.shared_intermediate_size if "shared_experts" in path else cfg.dense_intermediate_size
            )

            def bound(s, default):
                if s == "":
                    return default
                if s == "inter":
                    return inter_size
                if s == "2*inter":
                    return 2 * inter_size
                return int(s)

            lo, hi = bound(lo_s, 0), bound(hi_s, 2 * inter_size)
            target = _get_submodule(layer, path)
            with torch.no_grad():
                target.weight[lo:hi].copy_(value)
        else:
            target = _get_submodule(layer, module_path)
            set_param(target, value)

    if is_moe:
        # routed experts stay on disk; provide the per-expert dequantizer
        base = prefix + "block_sparse_moe.experts."
        experts = layer.mlp.experts

        def provider(expert_idx, _base=base):
            p = f"{_base}{expert_idx}."
            return (
                store.nvfp4(p + "w1.", dtype),
                store.nvfp4(p + "w3.", dtype),
                store.nvfp4(p + "w2.", dtype),
            )

        experts.provider = provider
        experts.layer_idx = layer_idx

    if missing:
        raise FileNotFoundError(f"layer {layer_idx}: {len(missing)} tensors unavailable, first: {missing[0]}")
    return layer


def build_layer(config, layer_idx):
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDecoderLayer

    module_num_heads[0] = config.num_attention_heads
    return MiniMaxM3VLDecoderLayer(config, layer_idx)


def make_experts_resident(layer, layer_idx, store, dtype=torch.float32, verbose=True):
    """Dequantize all 128 routed experts of one layer into RAM (~29 GB fp32).

    Only for modes that reuse the layer across many chunks (imatrix).  The
    layer's ``StreamingExperts`` provider is swapped for a dict lookup.
    """
    base = f"{LAYER_PREFIX}{layer_idx}.block_sparse_moe.experts."
    bank = {}
    for e in range(layer.mlp.experts.num_experts):
        p = f"{base}{e}."
        bank[e] = (
            store.nvfp4(p + "w1.", dtype),
            store.nvfp4(p + "w3.", dtype),
            store.nvfp4(p + "w2.", dtype),
        )
        if verbose and (e + 1) % 32 == 0:
            print(f"  resident experts {layer_idx}: {e + 1}/128", flush=True)
    layer.mlp.experts.provider = bank.__getitem__
    return bank


# ---------------------------------------------------------------------------
# high level reference model


class ReferenceModel:
    """Streaming, layer-at-a-time text reference."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, dtype=torch.float32, config_path: str | None = None):
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("dtype must be float32 or bfloat16")
        if config_path is None:
            local = Path(model_dir) / "config.json"
            config_path = os.environ.get("MINIMAX_CONFIG") or (str(local) if local.exists() else DEFAULT_CONFIG)
        torch.set_num_threads(int(os.environ.get("MINIMAX_THREADS", os.cpu_count() or 1)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.dtype = dtype
        self.config_path = config_path
        self.config = build_text_config(config_path)
        self.store = ShardStore(model_dir)
        self.mm = install_patches()
        module_num_heads[0] = self.config.num_attention_heads

        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding

        self.rotary = MiniMaxM3VLRotaryEmbedding(config=self.config)
        self._embed = None
        self._lm_head = None
        self._final_norm = None

    # -- embeddings / head ----------------------------------------------------

    @property
    def embed(self):
        if self._embed is None:
            from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRMSNorm

            w = self.store.get("language_model.model.embed_tokens.weight").to(self.dtype)
            self._embed = torch.nn.Embedding.from_pretrained(w, freeze=True)
            self._final_norm = MiniMaxM3VLRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            with torch.no_grad():
                self._final_norm.weight.copy_(
                    self.store.get("language_model.model.norm.weight").to(torch.float32)
                )
        return self._embed

    @property
    def final_norm(self):
        if self._final_norm is None:
            _ = self.embed
        return self._final_norm

    @property
    def lm_head(self):
        if self._lm_head is None:
            w = self.store.get("language_model.lm_head.weight").to(self.dtype)
            self._lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            with torch.no_grad():
                self._lm_head.weight.copy_(w)
        return self._lm_head

    # -- forward --------------------------------------------------------------

    def make_layer(self, layer_idx):
        layer = build_layer(self.config, layer_idx)
        if self.dtype != torch.float32:
            layer.to(self.dtype)
            if self.config.mlp_layer_types[layer_idx] == "sparse":
                # spec D.7: the router gate and bias stay fp32 (checkpoint F32)
                layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
                layer.mlp.gate.e_score_correction_bias.data = (
                    layer.mlp.gate.e_score_correction_bias.data.float()
                )
        load_layer_weights(layer, layer_idx, self.store, self.dtype, self.config)
        layer.eval()
        return layer

    def run_layer(self, layer, hidden, position_ids, dump: DumpContext | None = None):
        DumpContext.active = dump
        hooks = []
        if dump is not None:
            def pre_hook(module, args, kwargs):
                dump.put("attn_input", kwargs.get("hidden_states", args[0] if args else None))
                return None

            def post_hook(module, args, kwargs, output):
                dump.put("attn_out", output[0] if isinstance(output, tuple) else output)
                return None

            def router_hook(module, args, kwargs, output):
                dump.put("router_logits", output[0])
                dump.put("router_weights", output[1])
                dump.put("router_topk_ids", output[2])
                return None

            def shared_hook(module, args, kwargs, output):
                dump.put("shared_expert_out", output)
                return None

            hooks.append(layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
            hooks.append(layer.self_attn.register_forward_hook(post_hook, with_kwargs=True))
            if hasattr(layer.mlp, "gate"):
                hooks.append(layer.mlp.gate.register_forward_hook(router_hook, with_kwargs=True))
                if hasattr(layer.mlp, "shared_experts"):
                    hooks.append(
                        layer.mlp.shared_experts.register_forward_hook(shared_hook, with_kwargs=True)
                    )
                if hasattr(layer.mlp, "shared_experts"):
                    hooks.append(
                        layer.mlp.shared_experts.register_forward_hook(
                            lambda m, a, o, d=dump: d.put("shared_expert_out", o)
                        )
                    )
        try:
            with torch.no_grad():
                pos_emb = self.rotary(hidden, position_ids=position_ids)
                return layer(
                    hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=pos_emb,
                )
        finally:
            DumpContext.active = None
            for h in hooks:
                h.remove()

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embed(input_ids)

    def final_hidden(self, hidden):
        with torch.no_grad():
            return self.final_norm(hidden)

    def logits(self, hidden, chunk=64):
        """Full-vocabulary logits for every row, computed in row chunks."""
        out = []
        with torch.no_grad():
            for i in range(0, hidden.shape[0], chunk):
                out.append(self.lm_head(hidden[i : i + chunk]).to(torch.float32))
        return torch.cat(out, 0) if out else hidden.new_zeros((0, self.config.vocab_size))

    def forward_ids(self, input_ids: torch.Tensor, layers=None, dump_layer=None, dump_prompt=""):
        """Run a single unpadded sequence through the model.

        input_ids: LongTensor [1, T]
        Returns dict with last_hidden, logits (fp32, all rows).
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("forward_ids expects [1, T]")
        layers = range(self.config.num_hidden_layers) if layers is None else layers
        hidden = self.embed_ids(input_ids)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        dumps = {}
        for L in layers:
            layer = self.make_layer(L)
            dump = DumpContext() if (dump_layer is not None and L == dump_layer) else None
            t0 = time.time()
            hidden = self.run_layer(layer, hidden, position_ids, dump)
            if dump is not None:
                dumps[L] = dump.data
            print(f"  layer {L:2d} {time.time() - t0:7.2f}s", flush=True)
            del layer
            gc.collect()
        out = {"last_hidden": hidden}
        if layers and (list(layers)[-1] == self.config.num_hidden_layers - 1):
            out["final_hidden"] = self.final_hidden(hidden)
            out["logits"] = self.logits(out["final_hidden"][0])
        if dumps:
            out["dumps"] = dumps
        return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="MiniMax-M3.1-Flash CPU reference forward")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--layers", default=None, help="e.g. 0-2 or 0,1,5")
    ap.add_argument("--dump-layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", default=None, help="comma separated token ids (skips the tokenizer)")
    args = ap.parse_args()

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        tok_path = args.tokenizer or str(Path(args.model_dir) / "tokenizer.json")
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(tok_path)
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
    model = ReferenceModel(args.model_dir, torch.bfloat16 if args.dtype == "bf16" else torch.float32, args.config)
    layers = None
    if args.layers:
        layers = []
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                layers += list(range(int(a), int(b) + 1))
            else:
                layers.append(int(part))
    t0 = time.time()
    out = model.forward_ids(torch.tensor([ids]), layers=layers, dump_layer=args.dump_layer)
    print(f"done in {time.time() - t0:.1f}s")
    if "logits" in out:
        lg = out["logits"][-1]
        top = torch.topk(lg, 10)
        print("last-position top-10:", list(zip(top.indices.tolist(), [round(v, 4) for v in top.values.tolist()])))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        torch.save(out, Path(args.out) / "forward.pt")


if __name__ == "__main__":
    main()
