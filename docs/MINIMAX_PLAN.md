# MiniMax-M3.1-Flash engine for Strata: plan

Status: **draft for review, no engine code yet.** Branch `feat/minimax-engine`.

Everything below is sourced from files that were actually read: upstream Strata
(`5c8dd322`, v0.2.3), Strata-DS (`94130fa9`), and the Hugging Face repo
`MiniMaxAI/MiniMax-M3.1-Flash` at sha `54e4e69d` (config.json,
model.safetensors.index.json, safetensors headers, processors, chat template).
Memory figures are **arithmetic from tensor shapes**, not measurements. There
are no speed figures in this document, because nothing has been run yet.

---

## 1. What the model really is

Two corrections to the original brief:

- **It is 427B total, not 220B.** The HF metadata says 220.5B because the
  routed experts are packed NVFP4 (two 4-bit values per byte stored as U8).
  The logical count from tensor shapes is 427.05B, and the README says "~428B".
  **Active per token: about 23.5B.**
- **There is no BF16 release.** Routed experts ship as QAT NVFP4. Attention,
  dense MLP and shared experts ship as MXFP8 (E4M3 with 1x32 block scales).
  Every quant we produce derives from these weights.

| Item | Value (source: config.json / tensor headers) |
|---|---|
| Layers | 60. Layers 0-2 dense MLP (12288), layers 3-59 MoE |
| Hidden / vocab | 6144 / 200,064, untied `lm_head` |
| Attention | **MSA (MiniMax Sparse Attention) in all 60 layers.** GQA 64 Q heads / 4 KV heads, head_dim 128. No linear or lightning attention, no MLA, no sliding window |
| Sparse indexer | 4 index heads x 128 dim, 1 index-key head; blocks of 128 tokens; per query top-16 blocks plus 1 local block; max-pool block score; index-value path disabled on all layers |
| Norms / RoPE | Gemma RMSNorm (w+1), eps 1e-6; per-head QK-norm; partial RoPE 64 of 128 dims, theta 1e7 |
| MoE | **128 routed experts, top-4**, expert FF 3072 (w1 = gate, w3 = up, w2 = down); **1 shared expert** (FF 3072), always on |
| Router | **sigmoid** + `e_score_correction_bias` used for selection only; weights = unbiased sigmoid scores, renormalized; routed sum x 2.0 |
| Activation | `swigluoai`: gate clamped to <=7, up to +-7, out = (up+1)·gate·sigmoid(1.702·gate) |
| MTP | **None.** It is replaced by a separate **DSpark** draft model (`dspark/`, 1.96B params, 5 dense layers, block 7) |
| Context | `max_position_embeddings` 1,048,576. No RoPE-scaling key |
| Vision | CLIP-style ViT, 32 layers, width 1280, Conv3d patch 14 (temporal 2), 3D RoPE; projector 1280->6144->6144, then 2x2 patch-merge MLP 24576->6144->6144. A 672x672 image becomes 576 tokens. Video supported. No audio |
| Chat | `<mm:think>` reasoning tags, `reasoning_effort` low to max, tool calls as `<tool_call><invoke name=...>`, bos 200019, eos 200020 |

Size of the parts:

| Part | Params | Released format | Bytes |
|---|---|---|---|
| Routed experts (57 x 128 x 3 matrices) | 413.12B | NVFP4 + E4M3 scales | 232.4 GB |
| Attention + indexer (60 layers) | 6.65B | MXFP8 | |
| Shared experts (57) | 3.23B | MXFP8 | |
| Dense MLP (layers 0-2) | 0.68B | MXFP8 | 10.9 GB for these three rows |
| embed_tokens / lm_head | 1.23B each | BF16 | 2.46 GB each |
| Vision tower + projectors | ~0.87B | BF16 | 1.74 GB |
| DSpark draft | 1.96B | mixed | 2.30 GB |

One expert (one layer, all three matrices) is 56.6M params. **Per token the
engine must read 4 x 57 = 228 experts, 12.91B params.** That is 3.6 GB at
2.25 bpw and 7.3 GB at 4.5 bpw, plus about 10.6B dense params.

### Ecosystem status (checked 2026-10-23)

- **llama.cpp:** the `minimax-m3` arch has been merged since 2026-07, including
  vision (`PROJECTOR_TYPE_MINIMAX_M3`). Its converter does **not** handle the
  NVFP4/MXFP8 release. The dedicated MSA kernel PR was closed unmerged, so long
  contexts most likely run through a generic path (to be confirmed). M3.1
  needs an arch alias.
- **GGUFs of M3.1-Flash:** only `webmp3/Sakura-MiniMax-M3.1-Flash-Q2_K-GGUF`
  (155.9 GB + mmproj). The Unsloth repo is empty so far. Several M3 (base)
  GGUFs exist, but they are different weights.
- **transformers v5.20.2** has `minimax_m3_vl` (readable reference) but no
  NVFP4 loading. **vLLM / SGLang** support M3.1 only through open PRs, and only
  on datacenter Blackwell or MI355X.

---

## 2. The base we build on

**Upstream Strata, not Strata-DS.** Upstream runs on Windows and Linux,
supports CUDA sm_75/86/89/120 and HIP, and is current. Strata-DS is Linux- and
NVIDIA-only, frozen at upstream v0.1.40.2, and its tier I/O is Linux-only. We
**borrow designs and code** from DS; we do not merge it.

Integration follows the Qwen3.6 precedent (`src/qwen35/`). The new engine
lives in `src/minimax/`, built as static lib `strata_minimax` and linked into
`strata`. The family switch in `generate.cpp` picks it when GGUF
`general.architecture = minimax-m3`. Qwen3.8-Flash-Next and Qwen3.6 code paths
are not edited, apart from about 13 registration points (family enum, arch
string, `--families`, CMake).

### Reuse

| Need | Reuse from | Notes |
|---|---|---|
| Family plumbing, GGUF reader | `model_family.hpp`, `gguf_reader`, `family_layout.hpp` | add `MinimaxM3` |
| Server <-> engine protocol | `serve/server.py` line protocol (`GEN`/`T`/`DONE`/`PP`/`ERR`, optional `BGEN`) | engine-side handler modelled on q36's |
| OpenAI + Anthropic API | `serve/server.py` (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) | unchanged |
| Expert placement (VRAM slots + pinned arena + CPU pool + PCIe stream) | q36 `place_experts()` pattern; `K::cpu::ExpertPool` | the pool is shape-agnostic |
| Expert residency profile | `expert_cache.hpp` STRP format, `tools/make_profile.py` | 57 x 128 instead of 48 x 512 |
| SSD tier I/O | upstream `direct_file.cpp` + `FileExpertSource` (Windows overlapped + `NO_BUFFERING`, Linux `O_DIRECT`) | upstream is the only one with Windows support |
| Adaptive RAM tier, next-layer prefetch, 4 KiB-aligned expert pack | DS `host_experts.cpp`, `expert_prefetch.cu`, `tools/ds41/pack.py` | port to the upstream I/O layer |
| Biased-selection router | DS `k8_router.cu` | MiniMax uses sigmoid, not sqrt-softplus |
| FP8 E4M3 + E8M0 dense GEMV/GEMM | DS `fp8_gemv.cu`, `k2_fp8_gemm.cu` | DS uses 32x32 scale blocks; ours are 1x32 |
| NVFP4 expert kernels (GPU W4A8, W4A4 on sm_120a; CPU AVX2/AVX-512) | upstream `iq_kernels.cu` NVFP4 path, `nvfp4_avx2.cpp` | must be made shape-generic (see risks) |
| IQ/K-quant grouped expert kernels | upstream `iq_kernels.cu` (`STRATA_GU_FMTS`/`STRATA_D_FMTS`), `iq_avx2.cpp` | same |
| Sparse-attention top-k selection | upstream QSA (`qsa_select.cu`, `qsa_decode_attn.cu`): 4 indexer heads x 128 dim, top-k cells | very close to MSA; differs in block size, scoring and pooling |
| Speculative decoding | upstream `spec/controller.cpp`, `spec_prob.hpp`, `suffix_drafter.cpp`; DS `dspark.cu` | DS already implements a DSpark drafter for DeepSeek |
| Logit/KL tooling, bench | `tools/quant_quality.py`, `tools/qwen35/kl.py`, `tools/strata_bench.py`, `bench/bench_vs_llama.py` | |
| Hardware fit | `strata_control/gpu_fit.py`, `vram_fit.py`, setup.py recipes | |

### New work

1. **MSA attention (decode and prefill):** indexer projections and norms,
   block max-pool scores, top-16 + local block selection, GQA attention over
   the selected blocks. Start from QSA, and match the reference exactly
   (index-head to KV-head mapping, masking, score scaling).
2. **Router:** sigmoid scores, plus bias for selection, top-4 of 128,
   renormalize, x 2.0.
3. **swigluoai activation** fused into the GPU and CPU expert kernels. The
   upstream kernels hard-wire SiLU.
4. **Shape-generic expert kernels** for H = 6144, FF = 3072. Upstream
   `kCompiledGeometry` is 2560 x 640.
5. **Dense path:** MXFP8 native on sm_89+ (FP8 tensor cores). Pre-Ada GPUs
   (sm_75/86) and small-VRAM cards use a requantized Q8_0 or Q4_K copy.
6. **Converter/quantizer** (section 4).
7. **Layer split across GPUs inside the new engine.** q36 has no multi-GPU
   support at all, so this is written fresh, following the main engine's
   `--layer-split` semantics. The two 5070s have **no P2P** (`topo -p2p`
   reports GNS), so the per-token hidden state (12 KB) hops through pinned
   host memory, which is negligible.
8. **Long-context KV placement:** full K/V in pinned RAM, indexer keys in VRAM,
   and only the selected blocks fetched per token (section 5.3).
9. **Server `format: "minimax_m3"` front end:** tokenizer from
   `tokenizer.json` (DS pattern, no HF runtime dependency), chat template,
   `<mm:think>`, tool-call parser, EOS ids, `reasoning_effort` mapping.
10. **Vision:** ViT + projector + patch merge, served through the existing
    `GENI` image path.
11. **DSpark speculative decoding** (port DS's drafter to this model).

---

## 3. Correctness first: the reference

GPU frameworks (vLLM/SGLang) need datacenter Blackwell, and the full model in
BF16 is about 854 GB, so neither can run here. Plan:

- **R1. Layer-streaming PyTorch reference** (`tools/minimax/ref/`). It uses
  transformers v5.20.2 `minimax_m3_vl` modules in a separate venv (system
  Python stays on 5.12). Layers are loaded **one at a time** from the
  safetensors shards, NVFP4/MXFP8 is dequantized to BF16/FP32 (one MoE layer is
  about 14.8 GB), the whole prompt batch runs through that layer, then the
  layer is freed. This gives full 60-layer logits on CPU with 128 GB RAM.
  Throughput is irrelevant here.
- **R2. Validate the dequant formula.** The NVFP4 global-scale convention and
  the MXFP8 `weight_scale_inv` encoding are not stated in the repo. Cross-check
  against the vLLM/SGLang loader source, then sanity-check: perplexity on plain
  text, and top-1 agreement with llama.cpp running our `MM-IQ2_XS` once P1
  produces it.
- **Prompt set:** 8 prompts (code, English, Chinese, tool-call, chat
  template). Two prompts must be **longer than 2,176 tokens** (17 blocks x
  128); below that, MSA selects every block and the sparse path never runs.
- **Gates (in this order, before any speed tuning):**
  - G0: our C++ dequant equals the Python dequant **bit-for-bit**, every tensor
    type.
  - G1: per-op parity on captured activations (router ids and weights,
    selected blocks, attention out, expert out, shared expert).
  - G2: end-to-end teacher-forced logits, our engine with **native NVFP4 +
    MXFP8** weights vs R1: top-1 agreement and KL mean and p99. The threshold
    is set from the measured FP32-vs-BF16 noise floor of R1 itself.
  - G3: each quant variant, our engine vs llama.cpp on the **same GGUF**, both
    scored against R1, using Strata's `quant_quality.py` method.

---

## 4. Quant plan: Qwen-Flash-style tiers, each a separate download

Strata's Qwen Flash menu (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, UD-IQ4_XS … Q8_0,
NVFP4, Gyro) is made of **pre-made GGUFs that Strata repacks**. For M3.1-Flash
those GGUFs mostly don't exist, so we produce them ourselves.

**Format: GGUF using llama.cpp's `minimax-m3` tensor naming.** Our engine reads
the GGUF in place, as q36 does. An optional native pack adds 4 KiB-aligned
expert slots for the SSD tier (DS layout). This is what makes the requested
**"same quant" llama.cpp comparison** possible.

**Converter `strata-mm-quant` (C++, linked to `third_party/ggml`):**
- Reads one safetensors shard at a time and dequantizes NVFP4/MXFP8 per tensor
  in memory.
- Quantizes straight to the target ggml type with an imatrix and writes GGUF.
  **No BF16 intermediate:** one would be about 854 GB, more than this machine's
  free disk.
- The imatrix is collected by the R1 reference (per-column activation
  statistics on calibration text) and written in llama.cpp's imatrix format.
- An **NVFP4** recipe exists and is **disabled**. `block_nvfp4` has no
  global-scale field, and this checkpoint's `weight_scale / weight_global_scale`
  is not an exact E4M3 fold (layer 3 expert 0 `w1`, G=15123.69, scale byte 113).
  `strata-mm-quant` rejects `MM-NVFP4` instead of writing a lossy file under a
  lossless name.
- Q2_0 (ISTA GSQ format) is **not** in v1: Strata has no quantizer for it, only
  kernels.

Variants. Expert bpw are ggml nominal values. Pack size includes dense
(10.9 GB MXFP8), embeddings, `lm_head`, and the vision tower.

| Variant | Experts | Expert bytes | Pack approx. | Intended for |
|---|---|---|---|---|
| `MM-IQ1_S` | IQ1_S (1.56) | 81 GB | ~96 GB | smallest tier; 64 GB RAM machines; **needs a new IQ1_S expert kernel** (Strata has IQ1_M, not IQ1_S) |
| `MM-IQ1_M` | IQ1_M (1.75) | 90 GB | ~105 GB | 64-96 GB RAM machines |
| `MM-IQ2_XXS` | IQ2_XXS (2.06) | 106 GB | ~121 GB | 96-128 GB RAM, more headroom for long context |
| `MM-IQ2_S` | IQ2_S (2.5) | 129 GB | ~144 GB | between IQ2_XS and IQ3_XXS |
| **`MM-IQ2_XS`** (proposed default for 128 GB) | IQ2_XS (2.31) mix, IQ3/IQ4_NL down | 119 GB | ~134 GB | 128 GB RAM, nearly all resident |
| `MM-IQ3_XXS` | IQ3_XXS (3.06) | 158 GB | ~173 GB | 128 GB + SSD tier, or 192 GB |
| `MM-IQ3_S` | IQ3_S (3.44) | 178 GB | ~192 GB | 192 GB+ |
| `MM-IQ4_XS` | IQ4_XS (4.25) | 220 GB | ~234 GB | 256 GB |
| `MM-NVFP4` | disabled | 232 GB measured if it were legal | ~247 GB | not shipped; the fold into `GGML_TYPE_NVFP4` is not exact |

The low tiers (IQ1_S, IQ1_M, IQ2_XXS) use a **dynamic mix**, like Unsloth's
UD quants: `ffn_down` and the most-routed experts by profile get one step more
bits. The GGUF name keeps the tier label, and the exact recipe is stored in the
GGUF metadata. Every tier gets a KL/top-1 row (gate G3) so its quality cost is
measured.

The only tier being built now is `MM-IQ2_XS`. The other recipe files stay in
the tree and are not scheduled. `MM-IQ2_XS` is written as 13 final
`llama-gguf-split` parts (`split.no` / `split.count` / `split.tensors.count`,
KV metadata only in part 0). rig-2, the PC, and rig-3 each write the parts in
`tools/minimax/quant/run_tier.py`. Rigs download their safetensors from
Hugging Face. Nothing is copied over the 1 MB/s LAN except the quantizer
bundle and, later, the imatrix. `run_tier.py --plan` is the layer table and
the projected wall-clock. The low tiers and `MM-NVFP4` wait.

Dense sub-variants, chosen automatically by GPU:
- `fp8`: MXFP8 native, 10.9 GB, sm_89+.
- `q8`: Q8_0, 11.3 GB, sm_75/86 and HIP.
- `q4`: Q4_K, 6.0 GB, for GPUs with 16 GB or less.

**Disk (decided: everything on E:, 513 GB free):**

- Source safetensors (250 GB) go to `E:\...\Strata Fork Minimax\models\`,
  outside the repo.
- That leaves about 263 GB: room for `MM-IQ2_XS` (~134 GB) plus working
  space, but not `MM-IQ3_XXS` (~173 GB) as well.
- To fit, the Sakura Q2_K baseline download (156 GB) is **dropped**. The
  imatrix is collected by the R1 reference instead: per-column activation
  statistics from the source-precision weights, which is better than
  bootstrapping it from a Q2_K.
- Before `MM-IQ3_XXS` or `MM-NVFP4` are built, space has to be freed. Options:
  delete the source after its reference logits and imatrix are saved (it can be
  re-downloaded), or free space on E:. Decide when P1 reaches that point.

---

## 5. Memory budget

Your rig is 2x RTX 5070 (12 GB each, sm_120), Ryzen 9 5900X (AVX2, no
AVX-512), 128 GB DDR4, one NVMe drive.

### 5.1 VRAM, 2 x 12 GB

Both cards drive displays: about 2.0 GB and 1.6 GB are already in use at idle,
leaving ~20.8 GB usable. Minus a 0.7 GB reserve per card, that is **~19.4 GB**.

| Item | 32K | 128K | 256K |
|---|---|---|---|
| Dense weights MXFP8, split 30/30 layers | 10.9 | 10.9 | 10.9 |
| `lm_head` Q6_K (last card) | 1.0 | 1.0 | 1.0 |
| CUDA context + workspaces (estimate, to measure) | 2.0 | 2.0 | 2.0 |
| KV in VRAM, int8 | 2.4 | — | — |
| KV in RAM; indexer keys int8 in VRAM | — | 1.0 | 2.0 |
| **Left for hot-expert cache** | **~3.1** | **~4.5** | **~3.5** |

KV per token = 60 layers x (4x128 K + 4x128 V + 128 index-K) = 69,120 values
(f16 138 KB, int8 73 KB). At int8 that is 2.4 / 9.6 / 19.3 GB at
32K / 128K / 256K. At 128K and above, full KV does not fit next to FP8 dense,
so it moves to RAM. Only the selected 17 blocks per KV head are fetched each
step.

### 5.2 RAM, 128 GB

| Item | 32K | 128K | 256K |
|---|---|---|---|
| OS + apps reserve (Windows) | 12 | 12 | 12 |
| Embeddings Q8_0 + pinned staging | 2.3 | 2.3 | 2.3 |
| Vision tower (when enabled) | 1.8 | 1.8 | 1.8 |
| KV (int8) | — | 9.6 | 19.3 |
| **Left for experts** | **~112** | **~102** | **~93** |

### 5.3 Experts across the tiers (VRAM + RAM + SSD)

Coldest experts go to SSD, by profile rank. Percentages are **share of expert
bytes on SSD**. Per-token SSD reads will be lower, because the least-routed
experts are the ones placed there; the real figure is measured in the bench.

| Variant | 32K | 128K | 256K |
|---|---|---|---|
| MM-IQ1_M (90 GB) | 0% | 0% | 0% |
| MM-IQ2_XXS (106) | 0% | ~0% | ~10% |
| **MM-IQ2_XS (119)** | **~4%** | **~10%** | **~19%** |
| MM-IQ3_XXS (158) | ~27% | ~32% | ~39% |
| MM-IQ3_S (178) | ~35% | ~40% | ~46% |
| MM-IQ4_XS (220) | ~48% | ~51% | ~56% |
| MM-NVFP4 (232) | ~51% | ~54% | ~59% |

**Ceiling arithmetic (an upper bound, not a prediction):**
- Expert reads per token = 12.91B x bpw / 8.
- tok/s <= DDR4 bandwidth / RAM-resident bytes per token. Dual-channel
  DDR4-3200 is 51.2 GB/s theoretical; we will measure it.
- If SSD misses dominate: tok/s <= SSD GB/s / SSD bytes per token.
- These bounds are why IQ3/IQ4 on this rig depend on how skewed the routing is.
  The bench will show it.

### 5.4 Other setups (auto-sized at start; flags override)

The engine reads free VRAM and RAM at startup and fills VRAM, then RAM, then
SSD, keeping the reserves above. It refuses to start only if dense weights plus
minimum KV don't fit. The setup recipe recommends a variant from this table.

| Setup | Dense | KV | Recommended variant | Expected class |
|---|---|---|---|---|
| 1x 24 GB + 256 GB | fp8 on GPU | VRAM to ~64K, then RAM | MM-NVFP4 / IQ4_XS, all resident | RAM-bandwidth bound |
| 2x 12 GB or 1x 24 GB + 128 GB (yours) | fp8 | as in 5.1 | MM-IQ2_XS (IQ3_XXS with SSD) | RAM-bound, some SSD |
| 1x 16 GB + 128 GB | q4 (or fp8 with KV in RAM) | RAM above 32K | MM-IQ2_XS | same |
| 1x 12 GB + 64 GB | q4 | RAM | MM-IQ1_M (~45% on SSD) | SSD-bound |
| 1x 8 GB + 64 GB | q4; shared experts on CPU | RAM | MM-IQ1_M | SSD-bound, experimental |
| 32 GB RAM, any GPU | q4 | RAM, max 32K | MM-IQ1_M (~80% on SSD) | heavily SSD-bound; needs fast NVMe |
| 16 GB RAM | — | — | not supported for interactive use; boots but almost all reads hit SSD | — |
| CPU-only | ggml-cpu dense | RAM | MM-IQ1_M / IQ2_XS | **upstream has no CPU-only path**; this is a later phase |

**GPU support:**

| Backend / GPUs | Status |
|---|---|
| NVIDIA sm_120 (Blackwell) | primary target; FP8 and NVFP4 W4A4 tensor cores |
| NVIDIA sm_89 (Ada) | FP8 dense; NVFP4 via W4A8 |
| NVIDIA sm_86 / sm_75 (Ampere / Turing) | Q8_0/Q4_K dense |
| AMD HIP (RDNA3/4) | kernels written HIP-compatible like upstream, but **untested** (no AMD card here) |
| Metal / SYCL | out of scope for v1 |
| Mixed-VRAM multi-GPU | supported via `--layer-split` with uneven splits |

---

## 6. Phases, owners, deliverables

Work is split about 40% Claude / 30% Grok 4.7 High / 30% DeepSeek V4.1 Flash
High. The planner (Claude) reviews every result, and the correctness gates
G0-G2 always stay with Claude.

| # | Phase | Owner | Done when |
|---|---|---|---|
| P0 | Venv (`_local/`); download source to E:; dequant module; R1 reference; reference logits + imatrix; llama.cpp copy in `_baseline/` with M3.1 support | Claude (+ Grok for llama.cpp, DeepSeek for dequant/MSA spec) | R1 sane (R2), reference logits saved |
| P1 | `strata-mm-quant` + imatrix; GGUFs `MM-IQ2_XS`, then low tiers `MM-IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_S`, then `MM-IQ3_XXS`/`IQ3_S`/`IQ4_XS`/`NVFP4` as disk allows; IQ1_S expert kernel (GPU + AVX2) | Grok | G0 passes; llama.cpp runs our GGUFs |
| P2 | `src/minimax/` single-GPU engine: GGUF load, MSA, router, swigluoai experts, shared/dense, sampling, `--serve` protocol. Experts in RAM/CPU first, then GPU kernels | Claude (core) + DeepSeek (kernels) | **G1 + G2 pass** |
| P3 | Tiers: STRP profile, VRAM cache + adaptive swaps, pinned RAM arena, SSD tier (Windows + Linux I/O), DS adaptive RAM tier + next-layer prefetch, auto-sizing | DeepSeek | G2 still passes; tier stats reported |
| P4 | Layer split over 2 GPUs (no-P2P host hop); long-context KV in RAM | Claude | G2 passes on 2 GPUs; 256K runs |
| P5 | Server `format: minimax_m3`: tokenizer, template, think tags, tools, EOS; OpenAI + Anthropic endpoints on localhost | Grok | chat-golden tests pass; tool call round-trip |
| P6 | Vision: ViT + projectors + patch merge; `GENI` path | Grok | image-token embeddings match R1 per-tensor |
| P7 | DSpark spec decoding (port DS drafter), output-identity test | DeepSeek | greedy output identical with spec on/off |
| P8 | Bench (section 7) | Claude runs, Grok scripts | numbers in `bench/results/minimax/` |
| P9 | setup.py recipes, hardware-fit table, CPU-only mode | Grok + DeepSeek | each section 5.4 row boots or refuses cleanly |

Text-only chat through the API is usable after P2+P5. P3/P4 are what make it
fast, and P6/P7 are additive.

### P6 measured (2026-10-24)

- **mmproj:** `mmproj-MiniMax-M3.1-Flash-F16.gguf`, 524 tensors, 1.73 GB. The repo's converter
  (`tools/minimax/convert_mmproj.py`) produces byte-identical tensors to llama.cpp's
  `convert_hf_to_gguf.py --mmproj` for this checkpoint.
- **ViT + projector parity, fp32 CPU against transformers 5.20.2**, on the processor's own `pixel_values` (so
  preprocessing is excluded): per-token cosine min 0.9999998 / 0.9999934 / 0.9999977 (mean 1.00000000) for a red
  circle on white (391 tokens), a photograph (391) and a text card (450). Max-abs 0.13 / 0.096 / 0.29 on values up
  to 727 / 1494 / 543.
- **Preprocessing:** the engine's `smart_resize` + bicubic + mean/std matches the processor's `pixel_values` to
  max-abs 0.24 (red circle, hard edges) and 0.015 (text card); end-to-end embedding cosine on the red circle is
  min 0.9846, mean 0.99973, and 0.99996 on the text card. The residual is the resize at one hard-edge token.
- **CPU cost:** 64-80 s per 640x480 picture on 16 cores; 3.5 GB fp32 weights. The GPU path and video are not done.

### P4 measured (2026-10-24)

- **G1 on real weights** (Q8_0 dense and experts, F32 norms/router, truncated GGUFs from the real checkpoint,
  CPU placement): layer 0 from the embedding, layer 3 through layers 0-3, and layers 3 and 40 seeded with the
  reference's own input, on the 114-token and 4093-token prompts. Layer-out relative error 7.1e-4 to 2.4e-3,
  attention out 1.4e-3 to 1.3e-2, router top-4 agreement 96.5-99.1%, MSA block sets 98.4-99.6% (Jaccard 0.9974-
  0.9993) on the long prompt, expert weights by id 7.3e-4 to 3.0e-3. Every disagreement seen was a 4th/5th-place
  tie inside the Q8_0 noise floor.
- **Layer split**: `--gpus 0,1 --layer-split auto|K` with per-layer ownership of dense weights, KV and VRAM
  expert slots; the hidden state crosses through pinned host memory (no P2P). On the synthetic model the split is
  bit-identical to one GPU in both KV placements.
- **KV in VRAM / pinned RAM**: `--minimax-kv vram|ram|auto`; `ram` keeps K/V page-locked and copies only the
  selected blocks per step (<= 2048 positions per KV head) to the owning device, with the indexer keys in VRAM.
  RAM vs VRAM is bit-identical; the device attention path is 6.7e-5 max-abs from the host path.
- **Not run**: full-model benchmarks (the PC was busy with a quantization job) and an unequal-VRAM split (both
  cards here are 12 GB).

---

## 7. Benchmark protocol

- **Metrics:** decode tok/s, prefill tok/s, TTFT, peak VRAM per GPU (NVML),
  and RAM (working set + pinned + file cache) at **32K, 128K, 256K**. Also
  tier hit rates and SSD bytes per token.
- **Method:** medians of 3 runs, fixed prompts from `bench/prompts`
  (context-filler + question), greedy, spec decoding reported both off and on.
  Windows native first, then Linux (WSL `LLM-Ubuntu` needs the CUDA toolkit
  installed; weights on the Linux filesystem, not `/mnt/e`).
- **llama.cpp comparison:**
  - Same GGUF file, using the best existing build in `E:\llm` (mainline or
    ik_llama.cpp).
  - Settings: `-ot exps=CPU` with the hot layers' experts on GPU, flash-attn
    on, `-ctk/-ctv q8_0`, tensor split across both cards, the same context.
  - If llama.cpp cannot reach 256K on this RAM, we report that rather than
    work around it.
- Every result lists commit, build flags, driver, CUDA version and the exact
  command line. **No number in a report without its raw log.**

---

## 8. Risks / open items

1. **Dequant conventions** (NVFP4 global scale, MXFP8 `weight_scale_inv`
   encoding) are not documented in the repo. G0/R2 must settle them before
   anything else.
2. **Requantizing from QAT NVFP4.** IQ2/IQ3 from already-4-bit weights may lose
   more than from BF16 would. Measured by G3, not assumed.
3. **MSA details** (head mapping, score scale, causal block masking during
   prefill) come from transformers code, not the paper. The reference is the
   arbiter.
4. **Upstream kernels hard-code Qwen geometry.** Generalizing `iq_kernels.cu` /
   NVFP4 / AVX2 expert code to 6144 x 3072 is real work. Q5_K/Q6_K downs need
   FF divisible by 256 (3072 is, unlike Qwen's 640).
5. **No P2P between the 5070s** and a **Gen3 x16** link on GPU0. PCIe expert
   streaming will be slower on GPU0.
6. **Display VRAM:** ~3.6 GB is used by the desktops. Moving the monitors to
   one card frees about 2 GB on the other.
7. **CUDA toolchain:** `CUDA_PATH` points to 13.1 but nvcc is 13.3. The build
   pins 13.3 explicitly. The known-bad nvcc 13.2.0/13.2.1 are avoided.
8. **llama.cpp needs an M3.1 alias patch.** We write a minimal one in a
   separate llama.cpp copy and do not run the third-party `runtime.patch`.
9. **License:** MiniMax community license, not yet analyzed. It matters only if
   quants are ever published. The repo stays private.

## 8b. Publishing (decided 2026-10-23)

- **Quants go on Hugging Face, public,** under `LyJonathan`. The CLI is
  already logged in; the token is read from the HF cache and is never printed
  or committed.
- **Repo:** `LyJonathan/MiniMax-M3.1-Flash-GGUF`, one folder per tier,
  split with `llama-gguf-split` so no file is over 50 GB.
- **A tier is uploaded only after its quality gate (G3) runs.** Its model-card
  row gives measured KL / top-1 vs the reference, the exact recipe, file sizes
  and the llama.cpp commit and patch needed.
- **License compliance:** the MiniMax M3.1-Flash license allows publishing and
  distributing derivatives, provided the copyright and permission notice is
  included. Its `LICENSE` file ships in the HF repo (`license: other`,
  `license_name: minimax-community`, with a link to the original). The card
  restates the Commercial-Use conditions ("Built with MiniMax M3.1-Flash"
  attribution; notice or authorization to MiniMax) and the prohibited-use
  appendix.
- **Disk:** once a tier is uploaded and its upload is checked (file hashes),
  the local copy can be deleted. This is how E: makes room for the IQ3/IQ4
  tiers.
- **GitHub repo goes public at the end,** after:
  - a secrets scan of the full history (gitleaks or trufflehog)
  - a check that `_baseline/`, `_local/` and model files are untracked
  - a README section on MiniMax
  - your final OK on the checklist

## 9. Decisions (approved 2026-10-23)

1. Default quant for 128 GB is **`MM-IQ2_XS`**. **`MM-IQ3_XXS`** is the
   quality tier (with the SSD tier).
2. All data goes on **E:** (see the disk plan in section 4).
3. Order: text first (P0-P5), then vision (P6), then DSpark (P7). CPU-only
   comes later, in P9.
4. The llama.cpp baseline is a minimally patched copy in
   **`_baseline/llama.cpp/`** inside this repo, which is git-ignored (as is
   `_local/` for the venv and dumps).
