# MiniMax-M3.1-Flash — CPU reference (tools/minimax/ref)

This directory is the R1 reference from `docs/MINIMAX_PLAN.md` §3: a
PyTorch/transformers forward pass for the text tower of
`MiniMaxAI/MiniMax-M3.1-Flash`, runnable on CPU by streaming one decoder layer
at a time out of the 250 GB checkpoint. The numerical rules are pinned in
`_ref/MINIMAX_REFERENCE_SPEC.md` (sections A–F, outside git).

Nothing here is part of the engine; it exists to produce logits, activations
and an imatrix for parity work (gates G0–G3).

## Files

| file | purpose |
|---|---|
| `dequant.py` | NVFP4 and MXFP8 dequantization, exact per spec A/B; self-test + real-tensor stats |
| `model_ref.py` | builds `MiniMaxM3VLTextConfig` from `config.json`, per-layer weight loading, forward, dumps |
| `stream_run.py` | layer-streaming runner: prompt set logits, perplexity, greedy decode, dump mode |
| `imatrix.py` | calibration pass and GGUF imatrix writer (llama.cpp format, spec F) |
| `merge_imatrix.py` | sums partial imatrix GGUFs (chunks are independent) and writes the final file |
| `chat.py`, `make_prompts.py`, `make_calibration.py` | prompt/calibration corpus generation |
| `prompts.jsonl` | the 8-prompt set (two > 2,500 tokens) |
| `calibration.txt` | ~23.7K-token calibration corpus for the imatrix (12 chunks of 2048) |
| `test_dequant.py` | hand-built unit tests for both formats |

## Environment

```
uv venv Strata-MiniMax/_local/venv-ref --python 3.11
uv pip install --python .../venv-ref/Scripts/python.exe torch \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .../venv-ref/Scripts/python.exe \
    transformers==5.20.2 safetensors numpy tokenizers gguf pytest psutil
```

Verified: Python 3.11.9, torch 2.14.1+cpu, transformers 5.20.2 (contains
`transformers.models.minimax_m3_vl`), safetensors 0.8.0, tokenizers 0.23.3,
gguf 0.19.0 (llama.cpp gguf-py).

## Design notes / deviations

1. **Modules are transformers 5.20.2 `minimax_m3_vl` classes**, unmodified,
   except `MiniMaxM3VLExperts`, which is replaced by `StreamingExperts`
   (same math, but dequantizes one routed expert at a time from the shards
   instead of holding a 29 GB fp32 expert bank). The upstream forward already
   loops over hit experts, so this is numerically the same expression with the
   same accumulation.
2. **Layers are built and freed one at a time**, exactly as planned in R1.
   A full pass keeps one layer's dense weights (~0.7 GB fp32) plus one routed
   expert (~0.23 GB fp32) resident.
3. **Config is constructed explicitly** from `text_config` (spec D.9/D.10).
   `rope_parameters` is assembled from the checkpoint's top-level `rope_theta`
   and `partial_rotary_factor`; `sparse_attention_config` and `moe_layer_freq`
   are passed as kwargs so `__post_init__` derives `layer_types` (60 sparse
   attention layers) and `mlp_layer_types` (3 dense + 57 MoE).
4. **`attention_mask=None` into the decoder layer.** Every layer is sparse, and
   `MiniMaxM3VLIndexer.build_block_mask` rebuilds the token-level causal mask
   itself when no mask is supplied. Identical for an unpadded batch-1 prefill;
   padding is not supported by this reference.
5. **Router stays fp32** in bf16 mode too (`--dtype bf16` is only for the
   fp32-vs-bf16 noise floor); the checkpoint stores the router gate in F32 and
   spec D.7 makes it exact. Norms, as upstream, always normalize in fp32.
6. Additions that do not alter math: the indexer subclass `RefIndexer` copies
   the upstream forward and records block scores, and small module wrappers
   record activations for `--dump-layer`.
7. `--greedy` uses a *resident* model (all 60 layers in RAM, bf16 by default)
   because per-token layer streaming would re-dequantize every dense weight
   per token. Routed experts are still streamed.

## Usage

```sh
PY="E:/Jon's Projects/AI/Strata Fork Minimax/Strata-MiniMax/_local/venv-ref/Scripts/python.exe"
cd tools/minimax/ref

# unit tests
$PY -m pytest test_dequant.py -q
$PY dequant.py --self-test
$PY dequant.py --stats --tensors language_model.model.layers.32.self_attn.q_proj.weight

# full prompt set (fp32), outputs to _local/ref_out/prompts-fp32
$PY stream_run.py --out "E:/.../Strata-MiniMax/_local/ref_out/prompts-fp32"

# on a CUDA box: add --device cuda (public-domain reference output used this)
$PY stream_run.py --device cuda --dtype fp32 --out .../logits

# 3-layer smoke test
$PY stream_run.py --layers 30-32 --limit 2 --out .../smoke

# perplexity on a ~2K-token text
$PY stream_run.py --ppl ppl_text.txt --out .../ppl

# greedy continuation
$PY stream_run.py --greedy "The capital of France is" --max-new 32 --dtype bf16 --out .../greedy

# imatrix: modest 12-chunk corpus, split across two machines
$PY imatrix.py --estimate-only --layers 32-33 --max-chunks 2 --report .../imatrix_estimate.json
$PY imatrix.py --shard-index 0 --shard-count 2 --out .../imatrix-part0.gguf --report .../imatrix_part0.json
# (on the other machine) --shard-index 1 --shard-count 2 -> imatrix-part1.gguf
$PY merge_imatrix.py --out "E:/.../Strata-MiniMax/_local/ref_out/imatrix-minimax-m3.1.gguf" \
    .../imatrix-part0.gguf .../imatrix-part1.gguf

# per-op activations for layer L (chosen prompt)
$PY stream_run.py --dump-layer 32 --dump-prompt code_lru --out .../dump
```

Large outputs go to `Strata-MiniMax/_local/ref_out/` which is git-ignored.

## Dump contents (`--dump-layer L`)

`dump-L{n}/` + `index.json` (shapes/dtypes). Files:

- `q_after_rope.npy`, `k_after_rope.npy` — main branch after per-head Gemma
  RMSNorm + partial RoPE, `[1, 64 or 4, T, 128]`
- `index_q_after_rope.npy`, `index_k_after_rope.npy` — indexer branch
- `index_scores.npy` — raw indexer dot products `[1, 4, T, Tk]` (fp32)
- `index_block_scores.npy` — max-pooled per-block scores (post local boost)
- `index_block_topk_scores.npy`, `index_block_indices.npy` — top-16 selection,
  `-1` = unused slot
- `attn_core_out.npy` — attention output before `o_proj`
- `expert_{e}_gate_up.npy`, `expert_{e}_hidden.npy`, `expert_{e}_out.npy`,
  `expert_{e}_token_idx.npy` — each routed expert (only experts actually hit)
- the layer output itself is the next layer's input (not duplicated)

Router logits / selected ids / weights and the shared-expert output are
recorded through module hooks when dumping (see `model_ref.install_patches`).

## Measured results

Filled in from the runs on an A100-SXM4-80GB (Thunder Compute, 16 vCPU, 128 GB,
torch 2.12.1+cu130) and the PC (Ryzen 9 5900X, 24 threads, 128 GB):

- dequant sanity on real layer-3 tensors:
  - `q_proj` MXFP8: mean 1.1e-06, std 0.0322, absmax 0.234, zeros 6.5e-06,
    no NaN/inf
  - `w1` NVFP4 [3072,6144]: mean -9.3e-06, std 0.0274, absmax 0.178,
    zeros 7.10%, no NaN/inf
  - `w2` NVFP4 [6144,3072]: std 0.0283, absmax 0.165, zeros 7.18%
  - `w3` NVFP4: std 0.0272, absmax 0.165, zeros 7.10%
  - shared `gate_proj` MXFP8: std 0.0261, absmax 0.406, zeros 6.6e-06
- **NVFP4 divisor cross-check** (layer 53, expert 0): `2688 / weight_global_scale`
  reproduces the decoded tensor's absmax for `w2` exactly
  (`2688/6062.80 = 0.4433594`, decoded absmax `0.4433594`) and for `w3`
  (`2688/15123.69 = 0.1777344` vs `0.1777344`). If the global scale were a
  multiplier these would be off by `G²`. `w1`'s computed max is
  `2496/15123.69 = 0.16504` because its largest E4M3 block scale is 416, not
  448 — also consistent.
- MXFP8 `weight_scale_inv` bytes on layer 53 are 112…120 (E8M0), i.e.
  2^-15…2^-7; NVFP4 E4M3 block scales are 98…125.
- layer 53 (MoE), 2048 tokens, 24 threads on the PC: 0.8 s load, 38.3 s to
  dequantize all 128 experts to fp32 (once), 9.0 s per forward with the expert
  bank resident; 43 s per forward when experts are streamed from disk.
- imatrix, A100 bf16, 23,738 tokens, 12 chunks, all 60 layers:
  **603.5 s total, 10.1 s/layer** (1.2 s load + 2.0 s for 12 chunks + expert
  dequant). 711/711 expected entries, 12 chunks, counts sum to
  23,738 × 4 = 94,952 routed-token rows per expert tensor.
- A100 fp32 full prompt set (8 prompts, 10,910 tokens, last-128 fp32 logits +
  all-position top-1 + final hidden): **2,116 s (35.3 min)**, 33–36 s/layer.
- A100 perplexity, 2,048 tokens of *Alice in Wonderland* (raw text, no chat
  template): **ppl 39.69 fp32** (mean NLL 3.681), **ppl 34.50 bf16**
  (mean NLL 3.541) — the bf16-vs-fp32 noise floor is ~13% in PPL.
- A100 greedy, chat-templated "The capital of France is": **"Paris."** (EOS
  after two tokens).
- Top-1 agreement with the true next token over the prompt text: 27.2% for the
  4,093-token prose prompt, 30.2% for the 5,536-token code prompt, 36.7% for
  the math prompt.  (Next-token exact match is a strict measure; this is
  consistent with a working forward pass.)

**Caveat on raw-text perplexity.** 39.7 is higher than a base LM would score
on this text. The checkpoint is a chat/reasoning model: untemplated prose and
code are off-distribution (the chat-templated greedy answer is correct), and
per-position NLL on untemplated text is dominated by a few very confident
mistakes. Treat 39.69/34.50 as the measured reference numbers and use the
teacher-forced logits (not PPL) for engine parity. A llama.cpp cross-check on
the same GGUF (gate G3) is the right follow-up once `MM-IQ2_XS` exists.

Run artefacts are in `Strata-MiniMax/_local/ref_out/`:
`logits/` (8 prompts), `dumps/short` and `dumps/long` (layers 0, 3, 40),
`ppl_a100_fp32.json`, `ppl_a100_bf16.json`, `greedy_a100_bf16.json`,
`imatrix-minimax-m3.1.gguf`, `imatrix.json`.
