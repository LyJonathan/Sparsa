# GSQ + RCO on MiniMax-M3.1-Flash: feasibility, cost, and what to do instead

Research note, 2026-10-26. No engine runs, no PC lock. Sources are the cloned upstream
repos under `_ref/upstream/` (`gsq`, `rco`, `gptq-gguf-toolkit`), the ISTA HF card and its
allocation artifacts, the local MiniMax checkpoint, and this repo's own quantizer/reference
work.

**Bottom line.** The *method* transfers cleanly; the *released code* does not. GSQ as
published emits `compressed-tensors` (and, per its README only, K-quant GGUF) — it has no
IQ1/IQ2/IQ3 grid anywhere in the tree, and the GGUF path that produced the DeepSeek-V4.1
IQ builds is not public. RCO's assignment stage, by contrast, is trivially portable and is
worth adopting *now* at zero GPU cost. A faithful GSQ-on-IQ run is ~1,000–1,800 H100-hours
(≈$2.4k–$6.5k) plus 2–4 TB of scratch and 2–6 weeks of new quantizer code. It is not the
best next dollar.

---

## 0. What the ISTA release actually is (and a caveat on its numbers)

The card (`huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`, `README.md`,
§"Quantization procedure") describes a three-step pipeline:

1. per-tensor database: each weight tensor quantized at *every* candidate GGUF type with GSQ;
2. RCO search on the backbone's average bit-width;
3. assembly into a standard GGUF.

Dense tensors draw on Q5_K/Q6_K/Q8_0; routed experts on IQ1_S…IQ3_S. The search teacher is
"the released checkpoint itself, with the routed experts in their native FP4".

The shipped artifact is `tensor-allocation/…-IQ2_XXS.rco-allocation.txt`: 1008 tensors, with
histogram `BF16=71, F32=489, I32=1, I64=3, IQ1_M=60, IQ2_XS=18, IQ2_XXS=42, Q5_K=244, Q8_0=80`,
target bpw 2.00, measured 2.0038. Two facts fall out of that file that matter to us:

* **Only three expert types were actually used** (IQ1_M, IQ2_XXS, IQ2_XS) across the 120
  expert tensors (40 MoE layers × 3). The IQ1_S/IQ3_S rungs exist in the ladder but the
  search never selected them at this budget. A DB of K=3 is sufficient in practice.
* **Every dense tensor has the same type in every layer** (attn_q_a always Q8_0, attn_kv
  always Q8_0, attn_output_* always Q5_K, shexp always Q5_K). The freedom that the search
  exercised is essentially the expert ladder; the dense ladder looks fixed or collapsed.

**Caveat on "98.7–100% of base".** Their base is DeepSeek-V4.1-Flash *as released*: FP8
dense, FP4 routed experts (card §Results). So the recovery number is "2 bpw matches an
already-4-bit model", and the eval (AIME25/GPQA/LCB at reasoning effort 50) is close to
saturated for that base (AIME25 = 100.00 for base and all three builds). It is **not**
comparable to our G3 top-1-vs-FP32 numbers, where the reference is the FP32 dequant of the
checkpoint and there is real headroom. Do not import their percentage as an expected gain.

---

## 1. Does GSQ (or the code behind that card) produce llama.cpp I-quant formats?

**No — not publicly.**

`IST-DASLab/GSQ` (cloned at `_ref/upstream/gsq`, HEAD of `main`, Apache-2.0):

* Output format is `compressed-tensors` `pack-quantized`, group size 128, symmetric integer
  codes. `save_model.py:165-205` writes `quant_format: pack-quantized`,
  `quant_method: compressed-tensors`, `quantization_status: compressed`; README:61,406.
* The quantizer alphabets are fixed small scalar grids. `src/quantization/gumbel_quantizer_2bit.py:12`
  is literally `torch.tensor([-2, -1, 0, 1])`; `:13` is the 1-D group index
  (`arange(cols) // groupsize`); `:24-25` are the learnable `quant_logits` (shape
  `[4, rows, cols]`) and per-group `scales`. The ternary/1-bit modules use sign and mask
  logits. There is no codebook, no lattice, no imatrix.
* `grep -ri gguf` over the entire GSQ tree hits **only `README.md`** (lines 14, 30-34, 72).
  The README claims "can also refine publicly released GGUF K-Quant checkpoints in-format"
  (README:14) and reports the Qwen3-8B Q2_K result 50.03 → 56.28 (README:30-34), but **no
  script, module or config in the repo performs it**. There is no GGUF reader/writer, no
  `llama-quantize` hook, no IQ table.
* Supported models are wrappers in `src/models/` for LLaMA, Qwen3(-MoE), Qwen3.5(-MoE),
  Gemma-4, Kimi-K2/K2.5 (README:86-100). No MiniMax, no DeepSeek-V4.

The nearest *public* thing to the missing GGUF stage is a sibling repo the card does not
link: **`IST-DASLab/gptq-gguf-toolkit`** ("Efficient non-uniform quantization with GPTQ for
GGUF", cloned at `_ref/upstream/gptq-gguf-toolkit`). It is the ancestor of the card's
pipeline and it is honest about its scope:

* Quantization: GPTQ with direct K-quant export, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K only
  (README:22; `quant/gptq/src/quant_utils.py:12-16,21-25,98-109` — the `else` branch raises
  `Unsupported quantization type`).
* DB + assembly: `mapper/build_ep_database.sh`, `mapper/gguf_splitter.py`,
  `mapper/gguf_stitcher.py`, plus `config_converter.py`. The splitter's type table *does*
  know IQ1_S 1.5625, IQ1_M 1.75, IQ2_XXS 2.0625, IQ2_XS 2.3125, IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_NL, IQ4_XS (`gguf_splitter.py:46-48,80-90`), because it just slices already-quantized
  GGUF blobs. So the *DB/stitch* half works for IQ; the *quantize* half does not.
* Search: EvoPress evolutionary search with KL fitness (README:8,225-250) — RCO's
  predecessor, not RCO.
* **MoE support is listed as future work** (README:547-550). The toolkit as shipped is
  dense-model only.

So: the ISTA DeepSeek IQ1_M/IQ2_XXS/IQ2_XS files were produced by a pipeline that is **not
in either public repo**. Only the search *result* is published (the allocation txt), not the
search.

**What IQ would require technically.** IQ types are *not* a per-weight scalar alphabet:

* IQ1_S (`_baseline/llama.cpp/ggml/src/ggml-quants.c:4507-4690`) works on 32-weight blocks:
  an exhaustive 2-threshold search picks a 3-level scale (`besti1/besti2`, `best_shift`), then
  each 8-weight ternary pattern `L ∈ {0,1,2}^8` is mapped through `kmap_q2xs` to an index
  into the `kgrid_q2xs` codebook (or the nearest valid neighbour via
  `iq1_find_best_neighbour2`, `NGRID_IQ1S`). Signs are packed separately.
* IQ2_XXS is an E8-lattice codebook on 8-weight groups with per-32 4-bit sub-scales and an
  imatrix-weighted nearest-codebook search (`quantize_row_iq2_xxs_impl`, same file :3295).

So the GSQ idea ("learn the discrete grid assignment") *is* structurally applicable — the
grid index is already the discrete variable, and the sign pattern is already a binary
variable. But it needs a new quantizer module: a categorical logit over the codebook per
8-weight group + binary sign logits + the existing per-32 scale logits, with a differentiable
decode of the IQ lattice. Nothing like that exists upstream. Rough size: a few hundred lines
mirroring `gumbel_quantizer_ternary.py`, plus a differentiable/STE version of the lattice
decode, plus a bridge that emits the actual packed GGUF rows (our `strata-mm-quant` calls
`ggml_quantize_chunk`, which does argmin-RTN and cannot accept GSQ's chosen codes).

**Cheap approximation, if one insists:** run GSQ in its native 2-bit grid, dequantize to bf16,
then pack to IQ2_XXS/IQ2_XS with our existing imatrix quantizer. That projects the refined
weights back onto the IQ grid by nearest-neighbour, discarding the *joint* optimization that
is GSQ's entire value. Expect a small fraction of the reported gain, at full GSQ cost. Not
recommended.

---

## 2. What MiniMax support would take

Much less than feared, because we already have a working reference forward.

**HF modeling exists.** `transformers` 5.20.2 ships `transformers/models/minimax_m3_vl/`
(`modeling_minimax_m3_vl.py`, 74 KB), verified in `_local/venv-ref`. Relevant pieces:

* `MiniMaxM3VLExperts` (`modeling_minimax_m3_vl.py:182-191`): fused weights
  `gate_up_proj: [num_experts, 2*intermediate, hidden]` and
  `down_proj: [num_experts, hidden, intermediate]`. That is the same fused layout GSQ's MoE
  wrappers already handle (`src/models/qwen3_moe_dist.py:16-51` shards experts by
  `ExpertSharder` and all-to-alls tokens).
* `MiniMaxM3VLSparseMoeBlock` (:243-257), `MiniMaxM3VLTopKRouter` (:222-229, sigmoid scoring
  + `e_score_correction_bias`), `MiniMaxM3VLIndexer` (:481-600, block-sparse top-k indexer),
  `MiniMaxM3VLAttention` (:397), and `MiniMaxM3SparseForConditionalGeneration` (:1454).

**We already run it.** `tools/minimax/ref/` on branch `feat/minimax-ref` (README, `model_ref.py`,
`stream_run.py`, `dequant.py`, `imatrix.py`) builds the text tower from the *upstream*
transformers classes, streams one decoder layer at a time out of the 250 GB checkpoint,
replaces `MiniMaxM3VLExperts` with `StreamingExperts`, and produced everything in
`_local/ref_out/`: the 23,738-token imatrix, the FP32 logits for 8 prompts, and per-op dumps.
Measured on the A100 (Thunder Compute): **10.1 s/layer** for the imatrix pass and **33–36
s/layer** for full logits (`_local/ref_out/RESULTS.md`). That is exactly the harness GSQ
needs — GSQ only wants (a) layer input activations for calibration and (b) a layer-wise
forward/backward.

**Starting from NVFP4 / MXFP8.** `config.json` shows `quant_method: mxfp8` (block 1×32) for
dense/attention/shared experts and `moe_quant_algo: NVFP4, moe_quant_format:
nvfp4-pack-quantized` for routed experts. `tools/minimax/ref/dequant.py` already implements
both exactly (spec A/B), with unit tests. The GSQ "teacher" would be the dequantized bf16
checkpoint, which is also what our FP32 reference uses — consistent with the ISTA card's
"teacher is the released checkpoint with experts in native FP4".

**What is missing, concretely:**

| Piece | Status | Effort |
|---|---|---|
| Layer-wise fwd/bwd with MSA + indexer + sparse MoE | exists (`tools/minimax/ref`, upstream classes) | 0 |
| GSQ wrapper for MiniMax (q/k/v/o, indexer, shexp, routed experts, fused 3-D experts) | new `src/models/minimax_m3.py` mirroring `qwen3_moe*.py` | 2–4 days |
| Expert sharding for a 7.24 B-param MoE layer | `ExpertSharder` + all-to-all, already generic | 1–2 days |
| GSQ→IQ grid quantizer + GGUF emitter | does not exist anywhere | 2–6 weeks |
| RCO on GGUF groups | rewrite of grouping/store to read packed GGUF layers | 1 week |

Sizing per MoE layer (from `config.json`: hidden 6144, expert intermediate 3072, 128 experts):
`gate_up` 128×6144×6144 = 4.83 B params, `down` 128×6144×3072 = 2.42 B → **7.25 B per layer**,
413 B over 57 layers (matches `tools/minimax/quant/README.md`: 1.652 TB logical f32).

So GSQ's memory per layer, 2-bit quantizer: weights bf16 14.5 GB + logits `[4,·]` bf16 58 GB
+ grads 58 GB + Lion fp32 moment 116 GB ≈ **250 GB**, i.e. 4–8 H100s with expert sharding
(16 experts/GPU ≈ 31 GB of logits each). For a 1–2-logit (ternary/IQ1-like) quantizer it
halves to ~125 GB → 2–4 GPUs. There is no need for multi-node: MiniMax's expert block is 3×
smaller than Kimi-K2.5's per layer (384 experts × 44 M ≈ 16.9 B).

---

## 3. RCO on MiniMax: inputs, DB, compute, and the cheap proxy

**Inputs.** RCO's assignment stage needs only: (i) a group list, (ii) one cost/bit-width per
option, (iii) a sensitivity or loss response per (group, option). Its search
(`rco_search_quant.py`) takes `--layer-groups` regexes or `--moe-per-expert`, `--bitwidths`,
`--target-avg-bits`, `--objective kl|ce`, `--n-steps 300` (line 62), `--calibration-samples
256`, `--batch-size 4`, `--calibration-seq-length 2048` (lines 80-82). The model side is
`InterpolatedModel` (`src/search/quant.py:118-220`), which materializes per-group per-option
**deltas** `W_bw − W_ref` and forms `W = W_ref + Σ p_b Δ_b` — a full-model forward/backward
per step.

**Group count and DB size for us.** With GGUF's per-tensor type constraint — "A GGUF tensor
has one ggml type, so mixed types inside one stacked expert tensor are not possible"
(`tools/minimax/quant/README.md:106`) — the finest expressible granularity is *per tensor*,
i.e. per (layer, projection): 57 × 3 = **171 expert groups**. `--moe-per-expert` (RCO
`grouping.py:110`) gives 57 × 128 = 7,296 groups but GGUF cannot express the result; it is
only useful for pruning, not type assignment.

DB in RCO's native form (dequantized bf16 deltas per option):

| K expert types | fake-quant bf16 | + qparams sidecars | packed IQ (equivalent) |
|---:|---:|---:|---:|
| 3 (IQ1_M/IQ2_XXS/IQ2_XS, what ISTA shipped) | **2.48 TB** | ~0.4 TB | ~330 GB |
| 4 (+IQ1_S or IQ3_XXS) | **3.31 TB** | ~0.5 TB | ~440 GB |
| 5 | 4.13 TB | ~0.7 TB | ~550 GB |

The alpha state itself is tiny (171 × K floats), and the exact DP
(`src/search/quant.py:362 round_with_budget_dp`) is O(N·K·B) — microseconds. **Disk, not the
search, is the constraint.**

**RCO's own compute.** 300–400 steps × one 4×2048-token batch forward+backward through the
full 427 B model, plus one reference-log-prob pass over 256×2048 tokens. On 8×H100 that is
**~30–60 minutes** (MiniMax activates ~23 B params/token). Negligible next to the DB.

**Can our G3 KL harness replace the gradient search? Yes — and it is the right call.**
G3 (`tools/minimax/g3_sparsa.py`, `_local/g3_iq2xs/tools/compare.py`) gives KL64, full-vocab
KL and top-1 against the FP32 reference per prompt. But a full 60-layer pass over the 679
useful positions takes minutes-to-tens-of-minutes on CPU, and you would need N×K such passes
(171 × 3 = 513). Two cheaper, already-built substitutions:

1. **Per-tensor output error.** `strata-mm-quant --exp-layer N` already reports
   `plain_rel`, imatrix-`weighted_rel`, and `output_rel = ||Wx−Qx||/||Wx||` over real routed
   tokens, per expert matrix (`tools/minimax/quant/README.md` §§"Layer experiment"). It is a
   *local* proxy, measured on CPU in minutes per layer, and is exactly the kind of per-group
   signal RCO's `--init sensitivity` uses (`delta²/n_params`, `src/search/quant.py:271-287`).
2. **A sparse set of full-model confirmations.** Run G3 only for the final allocation and for
   3–5 candidate allocations, not per group.

Then solve the assignment *exactly* with the DP under the size budget. That is RCO minus the
Riemannian gradient refinement. The theoretical loss: the true objective is non-decomposable
(README, "What RCO solves"), so per-group proxies are an approximation — but our proxies are
*measured output* errors, not delta norms, so they are strictly better than RCO's own init.

Expected quality: the hand sweep is already doing this by intuition. Moving to a measured
table + exact DP should recover the remaining allocation headroom. Given the spread in our
own tiers at fixed bpw (IQ2_XXS 52.0% at 2.06 bpw vs IQ2_KT 56.3% at 2.125), a 1–3 pp top-1
gain at equal size is a realistic target; more than that would require the grid work.

---

## 4. Compute and cost for a full GSQ+RCO run

**Anchor.** GSQ's own scaling guidance for Qwen3-235B-A22B (128 experts, 94 MoE layers, 2.42 B
params/layer) is `configs/qwen3/qwen3_235B_A22B.yaml:9-12`:

```
2 nodes ( 8 GPUs) - 16 experts/GPU, ~24h for 94 layers
4 nodes (16 GPUs) -  8 experts/GPU, ~12h for 94 layers
8 nodes (32 GPUs) -  4 experts/GPU, ~6h  for 94 layers
```

That is **2.04 GPU-hours per layer, constant in GPU count** (192 GPU-h total). Scaling by
parameters per layer (MiniMax 7.25 B vs Qwen 2.42 B = 3.0×) and discounting slightly for
top-4 vs top-8 routing:

**≈ 6 GPU-h per MiniMax MoE layer → 340 GPU-h for the 57 MoE layers** (+ ~5–10 for the 3
dense). For a 1–2-logit (IQ1-like) grid, roughly half: ~170 GPU-h.

| Scope | GPU-hours | Wall clock, 8×H100 | Wall clock, 16×H100 |
|---|---:|---:|---:|
| One expert type, 2-bit-style grid | 300–400 | 38–50 h | 19–25 h |
| One expert type, IQ1-style grid | 150–220 | 19–28 h | 10–14 h |
| **DB of 3 types** (1× IQ1 + 2× IQ2) | **900–1,200** | 113–150 h | 56–75 h |
| DB of 4 types | 1,200–1,600 | 150–200 h | 75–100 h |
| RCO search (fwd/bwd over the assembled DB) | 10–50 | 2–7 h | 1–4 h |
| Packing 413 B params × K to IQ on CPU | — | 3–9 h/type on one 5900X (0.14–0.5 GB/s) | — |

Uncertainty is ±2×: the anchor is a config comment, not a measured log, and grid-refinement
work per weight differs between alphabets.

**Rental options (checked 2026-10-26).**

| Provider | H100 80GB | A100 80GB | Notes |
|---|---:|---:|---|
| Thunder Compute (`tnr` 2.1.0, installed) | **$3.20/GPU-h** on-demand | **$1.09** | Instance disk max 500 GB, 32–128 GB RAM, 4–16 vCPU. **Too small for a 2.5–4 TB DB.** Good for the RCO search and for ≤1-type runs. |
| Verda (the lab's own sponsor) | $4.606 on-demand / **$2.303 spot** | $2.16 / $1.08 | Instant clusters up to 144 GPUs, InfiniBand, large NVMe, storage $0.20/GiB-month. Best fit. |
| RunPod / Lambda | ~$3.3–4.0 | ~$1.3–1.9 | Comparable; big local NVMe on 8× nodes. |

**Total cost for the full GSQ+RCO-on-IQ program:**

* Compute: 900–1,600 GPU-h × $2.30 (Verda spot) = **$2,100–3,700**; at on-demand H100
  ($3.20–4.61) = **$2,900–7,400**.
* Storage while the DB lives: 2.5–4 TB × $0.20/GiB-mo = **$500–800/month** (or run one type
  at a time and delete: 0.8 TB peak, ~$170/month).
* Wall clock end-to-end, 8–16 H100s: **1.5–2.5 weeks**, assuming the IQ GSQ quantizer already
  exists. Add **2–6 weeks of engineering** before the first useful GPU-hour.
* On A100 ($1.09, Thunder) the job is 2–3× slower: 2,000–4,500 A100-h ≈ **$2,200–4,900** —
  no real saving, and Thunder's disk cap rules it out for the DB.

**Verdict: ~$2.5k–$6.5k in compute, dominated by building the per-type DB, not by RCO.** RCO
itself is a rounding error. The real cost is the quantizer code and the disk.

---

## 5. What is cheap to adopt now, without big compute

**(a) Exact-budget allocation over measurements we already have — zero GPU, highest value.**
Build the table `Δ(group, type)` from `--exp-layer` `output_rel` (57 layers × 3 projections ×
K types), cost `c(type)` from `ggml_row_size`, weight `w` from parameter count, and solve the
multiple-choice knapsack exactly (`rco/src/search/quant.py:362`). We already have per-matrix
output error for layer 40 across IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_KT/IQ3_* and the per-layer
G3 numbers in `tools/minimax/g3_results.md`. This replaces the hand sweep with a reproducible,
provably budget-exact allocation. Caveat: the objective is non-decomposable, so confirm the
final 3 candidates with G3.

**(b) Build a per-layer DB out of the uniform tiers we already ship and stitch.**
`gptq-gguf-toolkit/mapper/gguf_splitter.py` + `gguf_stitcher.py` already understand IQ type
ids and bpw (lines 46-48, 80-90) and can splice GGUF layers. If we assemble an IQ1_M DB, an
IQ2_XXS DB and an IQ2_XS DB from existing tier files (all produced from the *same* imatrix
and calibration), each layer gets 3 options with no requantization at all. Then (a) runs on
real variants. Disk is ~830 GB for the three tiers; can be pruned to the expert tensors only.

**(c) GPTQ-style rounding for IQ types (the "is it worth it" experiment).**
Feasible, and the GPU side is cheap: the Hessian per layer is 6144² fp32 = 151 MB (8.6 GB for
57 layers), and the GPTQ column loop with blocksize 128 is seconds per matrix. But GPTQ must
call the *IQ quantizer* per column with an error-feedback correction, and our IQ search is
CPU-only and scalar (`ggml_quantize_chunk`), so the implementation is a C++ change to the row
loop (accept precomputed corrections from a GPU/H sidecar) — **2–4 weeks**. The published
evidence for GPTQ-vs-imatrix at ~2 bpw is mixed: the toolkit's own table shows GPTQ-2 at 2.56
bpw is *worse* than imatrix-based EvoPress-kQuant at 2.71 bpw on C4 PPL (17.94 vs 14.10),
i.e. GPTQ below ~3 bpw underperforms imatrix-aware RTN. **Recommendation: test it on one layer
(layer 40) on an A100/A6000 (~$20), not on the full model.**

**(d) Per-group scale refinement only.** The cheapest slice of GSQ: keep the IQ codebook
assignment from the existing quantizer, but re-solve the per-32 (IQ1_S) / per-32-sub-scale
(IQ2_XXS) scale under the imatrix-weighted objective, or do a 3-point scale jitter search.
This is a small change inside `quantize.hpp` / the ggml row function, no GPU, no new grids,
and captures the "learned scales" half of GSQ. Do this with (a).

**(e) Do not** run public GSQ as-is: its output is `compressed-tensors`, which neither our
engine nor our GGUF packer consumes, and its 4-level grid is not an IQ grid. The GSQ pass
would cost $1k and end in a format conversion that throws the gain away.

---

## 6. Licenses

* **GSQ**: Apache-2.0 (`_ref/upstream/gsq/LICENSE`). **RCO**: Apache-2.0
  (`_ref/upstream/rco/LICENSE`, also `CITATION.cff`). **gptq-gguf-toolkit**: no LICENSE file
  in the tree at clone time — treat as all-rights-reserved unless the author clarifies; do
  not copy code from it.
* If we vendor or adapt GSQ/RCO code we must: (i) include a copy of Apache-2.0, (ii) retain
  copyright/attribution notices, (iii) **state prominently that we modified the files**, (iv)
  carry any `NOTICE` file (neither repo has one), (v) add entries to our
  `THIRD_PARTY_NOTICES.md` (already used for ik_llama.cpp). Apache-2.0 grants a patent
  license and is *not* copyleft, so our engine stays closed/whatever we like.
* Reimplementing the *method* from the papers is not a license event at all. Citing both
  papers in the model card is expected practice and is what the ISTA card asks for
  (two BibTeX entries, §Citation) — not a legal obligation.
* The HF card's `license: mit` applies to *their released weights*, not to the GSQ/RCO
  tooling (the card says so explicitly under §License). We are not shipping their weights.

---

## 7. Recommendation

**Do not port GSQ to MiniMax now.** The released code cannot emit IQ, the GGUF pipeline
behind the ISTA card is unpublished, and the engineering cost (2–6 weeks) buys an
unquantified gain whose headline evidence is measured against an already-FP4 base.

**Three phases, in order:**

1. **This week, $0.** Implement (5a)+(5b): measured per-tensor table → exact DP → stitch a
   mixed tier at 2.00–2.25 bpw from existing uniform tiers. Validate with G3 on 3 candidate
   allocations. This is RCO's assignment stage with a better proxy, and it directly replaces
   the hand sweep. Success criterion: ≥ IQ2_XS-D3's 59.4% top-1 at ≤133 GB, or equal quality
   at ~10 GB less.
2. **Next, ~$50.** Layer-40 experiments: (5d) scale refinement and (5c) a one-layer GPTQ
   prototype. Measure `output_rel` and a single-layer KL. This tells us whether the residual
   error is in the *scale* (cheap, adopt) or the *grid assignment* (needs GSQ-style search).
3. **Only if phase 2 shows the grid is the bottleneck: ~$3k–$6k, 1.5–2.5 weeks wall.**
   Write the IQ-grid GSQ quantizer (grid-index logits + sign logits + IQ emitter), run the
   MiniMax wrapper on `feat/minimax-ref`, build a K=3 DB (IQ1_M/IQ2_XXS/IQ2_XS) on Verda spot
   H100s, then RCO/DP allocate. Reuse `tools/minimax/ref` for calibration activations and the
   imatrix corpus so the GSQ teacher and our packer see the same distribution.

The single highest-leverage fact in this note: **the allocation half of the ISTA result is
reproducible today with files we already have**, and the quantization half is not published.

---

### Source index

| Claim | Source |
|---|---|
| GSQ emits compressed-tensors only | `_ref/upstream/gsq/save_model.py:165-205`; README:61,406 |
| GSQ grids are per-weight scalars | `_ref/upstream/gsq/src/quantization/gumbel_quantizer_2bit.py:12-25` |
| No GGUF code in GSQ | `grep -ri gguf _ref/upstream/gsq` → README only (14,30-34,72) |
| Public GGUF toolkit is K-quant only, dense-only | `_ref/upstream/gptq-gguf-toolkit/README.md:22,547-550`; `quant/gptq/src/quant_utils.py:12-16` |
| Toolkit splitter knows IQ types | `mapper/gguf_splitter.py:46-48,80-90` |
| ISTA allocation result (3 expert types) | HF card `tensor-allocation/…-IQ2_XXS.rco-allocation.txt` |
| ISTA base is FP8/FP4, eval saturated | HF card §Results, §Overview |
| IQ1_S/IQ2_XXS are codebook formats | `_baseline/llama.cpp/ggml/src/ggml-quants.c:4507-4690`, `:3295` |
| MiniMax HF modeling | `_local/venv-ref/.../transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py:182-191,222-257,481` |
| Our reference forward exists | branch `feat/minimax-ref`, `tools/minimax/ref/README.md`; `_local/ref_out/RESULTS.md` |
| Checkpoint is MXFP8 + NVFP4 | `models/MiniMax-M3.1-Flash/config.json` |
| GSQ scaling anchor | `_ref/upstream/gsq/configs/qwen3/qwen3_235B_A22B.yaml:9-12` |
| RCO search cost/knobs | `_ref/upstream/rco/rco_search_quant.py:60-86`; `src/search/quant.py:118-220,362` |
| Per-tensor type is the GGUF limit | `tools/minimax/quant/README.md:106` |
| Our per-layer error harness | `tools/minimax/quant/README.md` §"Layer experiment" |
| G3 harness | `tools/minimax/g3_results.md`; `tools/minimax/g3_sparsa.py:305-311` |
| GPU pricing | thundercompute.com/pricing, verda.com/pricing, 2026-10-26 |
