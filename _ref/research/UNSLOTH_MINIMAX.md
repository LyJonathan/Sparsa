# Unsloth MiniMax-M3.1-Flash vs Sparsa D3

Research note, 2026-10-26. No engine runs and no model downloads were used: the
tensor tables come from the GGUF headers of the Hub shards (HTTP Range, 1-32 MB
each) and from the local D3 pack's headers; the scores are re-scores of the
existing G3 dumps. Items marked **(inferred)** were not measured.

## TL;DR

* The 58% vs 91.5% gap is **mostly the metric, not the quant**. Ours is
  teacher-forced top-1 on *prompt* positions against an FP32 reference. Theirs
  is per-token top-1 over *greedy-generated* tokens against a Q8 file that is
  itself only 95.5-99.1% away from their reference. On the slice of our dumps that
  looks like generated text (FP32 reference's own greedy path and confident,
  n = 125) **D3 scores 96.0%**, between their Q2 (91.5%) and Q4 (97.3%).
* On all 935 prompt positions D3 is 53.8%; on all 232 positions that are on the
  FP32 model's greedy path it is 84.5%. The reference's own top-1 against the
  true next token is only 25.0%, i.e. these prompt positions are mostly
  ambiguous and a 2-bit quant flips many near-ties.
* The engine is not the story. llama.cpp and Sparsa agree on D3 to 0.15 pp
  (59.20% vs 59.35% on the six short prompts).
* Their tensor allocation differs in a useful way: they spend 4 GB less than D3
  on attention/shared/embedding/output, keep the **indexer in F32**, and put the
  savings into **IQ4_XS down experts on layers 52/58/59** and **IQ3_XXS gate/up
  on layer 59**. D3 protects none of the late layers.
* Our engine **will load their target GGUF**, but with one real bug: Unsloth omits
  `minimax-m3.attention.indexer.first_layer`, and our default is the dense prefix
  (3), so layers 0-2 would run without MSA. Their metadata also spells
  `swiglu_alpha`/`swiglu_limit` while we read `swiglu.alpha`/`swiglu.limit`
  (harmless today: the values equal the defaults).
* Their DSpark **draft GGUF cannot be loaded by us** (different arch keys, tensor
  names and dtypes). The architecture itself matches ours on taps (3,17,31,45,59),
  mask token 200058 and Markov orientation; the differences are draft count
  (they run `n_max` rows, we always run 7) and where the Markov bias is computed.
  Their n-max 1 is fastest because the verify window, not the drafter, is the
  cost: our own measurement is 3.58 s for 2.11 verified rows vs 333 ms for one
  plain token.

## 1. What the two "top-1" numbers actually are

### Ours (`tools/minimax/g3_sparsa.py`, copy of `_local/g3_iq2xs/tools/compare.py`)

| item | value |
|---|---|
| reference | MiniMax-M3.1-Flash, **FP32 weights** on an A100, `_local/ref_out/logits` |
| positions | **prompt positions**: every position of the short prompts; the last 128 of the long ones. Row *i* predicts `ids[start+i+1]` |
| teacher forcing | yes, the reference's own `token_ids.npy` are fed verbatim |
| top-1 | `argmax(ref) == argmax(quant)` at the same position, **not** against the true next token |
| KL64 | KL(ref‖quant) over the reference's **top-64** ids, both renormalized to those 64 |
| prompt set | 6 short prompts = 679 positions; +2 long = 935; held-out 14 = 53.46% / 1.4999 (given) |
| reference's own top-1 vs true next | **25.02%** on the 935 positions |

The prompt set is chat-templated text with no assistant answer, so most
positions are open-ended. The FP32 model itself picks the actual next token only
a quarter of the time. Position 0 (no context) agrees 37.5% of the time; dropping
the first 32 positions raises D3 from 53.80% to 56.70% only. Early positions are
therefore a 2.9 pp effect, not the gap.

### Theirs (Unsloth docs, Dynamic 3.0 page, PR #31192)

| item | value |
|---|---|
| reference | `UD-Q8_K_XL`, described as "a direct lossless conversion from the original safetensors … left at **NVFP4 + BF16**". Verified from its header: routed experts `NVFP4` + F32 per-expert scale, dense/attention `BF16` |
| positions | **greedy-generated tokens**: "300 held out examples … greedy argmax decoding for **32 tokens** for BF16 vs all quants" (Dynamic 3.0, Divergence-300 @32) |
| teacher forcing | no; both sides generate their own trajectory |
| top-1 | per-token agreement along those continuations |
| PR #31192's own check | the native NVFP4 GGUF agrees with "a reference GPU implementation on **95.5% to 99.1%** of top-1 tokens **over its own greedy continuations**" |
| counts | 300 prompts × 32 tokens per category **(inferred** for the MiniMax table; the page gives no count) |
| headline | 91.5% = a weighted average over seven categories, weights not published; per-category 88.5% (chat) and 95.7% (math) |

Their own scale therefore saturates at 97.8-99.3% (Q8) and 98.0% (Q4), not 100%:
the reference file itself differs from their GPU reference by 1-4%. The useful
dynamic range is 91.5 → 97.8, i.e. 6.3 points.

The MiniMax page does not say "teacher forced" anywhere. The only published
methodology is the 32-token generation harness. Whether the 91.5% number is that
exact harness is **(inferred)**; it is certainly not a prompt-position metric.

### Mapping D3 onto their regime (re-score of existing dumps, no engine)

| slice | n | D3 top-1 | D3 KL64 |
|---|---:|---:|---:|
| all 935 prompt positions | 935 | 53.80% | 1.578 |
| six short prompts (llama.cpp / Sparsa) | 679 | 59.20% / 59.35% | 1.345 / 1.211 |
| on the FP32 model's greedy path (`argmax(ref) == true next`) | 232 | 84.48% | 0.462 |
| on path **and** ref p(max) ≥ 0.90 | 125 | **96.00%** | – |
| ref p(max) ≥ 0.99 | 121 | 89.34% | 0.424 |
| drop the first 32 positions | 679 | 56.70% | – |
| IQ2_XS (no D3 down upgrade), all 935 | 935 | 46.84% | 1.916 |

Generated tokens are, by construction, mostly on the model's own greedy path and
confident. On the closest analogue we can build (on-path + p ≥ 0.90) D3 is
**96.0%**, i.e. at their Q4 level and above their Q2. This is a proxy, n = 125
from 8 prompts, not their 300-prompt harness; treat it as strong evidence that
58% ≠ their 91.5%, not as a replacement measurement.

**Did the earlier llama.cpp runs agree?** Yes. `_local/g3_iq2xs/results_d3.md`
has IQ2_XS 46.84% / KL64 1.916 and D3 53.80% / 1.578 over 935 positions, and the
Sparsa engine's `tools/minimax/g3_results.md` has D3 59.35% / 1.211 over 679.
Two independent engines, same quant, same positions: the metric is engine
independent.

**Verdict.** The metrics are **not comparable**. If one forces the comparison,
D3 at 58% is far behind their 91.5%, but that comparison charges D3 for
ambiguous prompt positions their harness never scores. On their kind of slice
D3 is ~96%. What remains is a real but modest allocation gap (section 2), not a
33-point quality collapse.

## 2. Where the bits go (header parse, exact bytes)

Types and decimal GB. "bpw" = payload bits per weight for that group. Their three
2-bit-class files share an identical down-expert, attention, indexer and shared
layout; gate/up changes, and Q2_K_XL alone raises the embedding to Q5_K.

| group | D3 | UD-IQ2_XXS | UD-Q2_K_XL | UD-IQ1_M |
|---|---:|---:|---:|---:|
| attention q/k/v/o | 6.82 GB, Q8_0, 8.50 bpw | 4.43, Q5_K/Q6_K, 5.52 | 4.48, Q5_K/Q6_K/Q8_0, 5.58 | 4.43, 5.52 |
| MSA indexer | 0.25, **Q8_0, 8.50** | 0.94, **F32, 32.0** | 0.94, **F32, 32.0** | 0.94, **F32, 32.0** |
| shared expert | 3.61, Q8_0, 8.50 | 2.37, Q5_K/Q6_K, 5.88 | 2.37, 5.88 | 2.37, 5.88 |
| dense FFN (3 layers) | 0.72, Q8_0, 8.50 | 0.53, Q5_K/Q6_K, 6.28 | 0.53, 6.28 | 0.53, 6.28 |
| router + bias | 0.18, F32 | 0.18, F32 | 0.18, F32 | 0.18, F32 |
| routed gate/up | 79.61, IQ2_XS, 2.31 | 71.31, IQ2_XXS (56 L) + IQ2_S (L59), 2.07 | 80.07, IQ2_XS (56 L) + IQ3_XXS (L59), 2.33 | 65.72, IQ1_M (28 L) + IQ2_XXS (29 L), 1.91 |
| routed down | 52.72, IQ3_XXS, 3.06 | 53.79, IQ3_XXS (54 L) + IQ4_XS (L52/58/59), 3.12 | 53.79, same | 53.79, same |
| token_embd | 1.31, Q8_0, 8.50 | 0.69, Q4_K, 4.50 | 0.85, Q5_K, 5.50 | 0.69, Q4_K |
| output | 1.01, Q6_K, 6.56 | 0.69, Q4_K, 4.50 | 0.69, Q4_K | 0.69, Q4_K |
| norms | 0.09, F32 | 0.09, F32 | 0.09, F32 | 0.09, F32 |
| **total** | **146.05 GB / 136.02 GiB** | **134.94 / 125.7** | **143.90 / 134.0** | **129.35 / 120.5** |
| experts | 132.33 | 125.10 | 133.86 | 119.51 |
| non-expert | 13.72 | 9.84 | 10.04 | 9.84 |

(Our local D3 pack measures 136.02 GiB / 146.05 GB, not the 142.9 GiB in the
brief. The 132.3 GB experts figure matches to 0.03 GB, so this is the pack; the
142.9 may include a different tier or an older copy.)

**Where they spend bits we do not**

1. **MSA indexer at F32, 32 bpw** (+0.69 GB). They refuse to quantize it; our
   Q8_0 indexer is the single biggest architectural difference in the dense path.
2. **Down experts on layers 52, 58, 59 at IQ4_XS** instead of IQ3_XXS (+1.07 GB).
3. **Gate/up on layer 59 at IQ3_XXS** instead of IQ2_XS (+0.03 GB, Q2_K_XL only).
4. IQ1_M additionally gives **29 of 57 gate/up layers IQ2_XXS** while 28 layers
   drop to IQ1_M — a per-layer, not per-tensor, mix.

**Where we spend bits they do not** — about 4 GB that does not touch the routed
experts, which are the error source:

* attention Q8_0 vs Q5_K/Q6_K: **+2.34 GB**
* shared expert Q8_0 vs Q5_K/Q6_K: **+1.24 GB**
* token_embd Q8_0 vs Q4_K/Q5_K: **+0.46 to +0.62 GB**
* output Q6_K vs Q4_K: **+0.32 GB**
* dense FFN Q8_0 vs Q5_K/Q6_K: **+0.19 GB**
* indexer Q8_0 vs F32: **−0.69 GB**

Net: D3 has 2.3 bpw gate/up and 3.06 bpw down — essentially their Q2_K_XL — but
spends 3.7 GB more on non-experts and protects no late layer. Reallocating
those bytes to their choices is the cheapest quality experiment by size; it is
not cheap in time (section 5).

## 3. Will our engine load their files?

Target GGUF: **yes, with one fix and two cosmetic differences.** Architecture is
`minimax-m3`; all 60 layers use the same tensor names and shapes our
`src/minimax/engine.cpp` requires (`attn_q/k/v/output`, `attn_q/k_norm`,
`indexer.q/k_proj`, `indexer.q/k_norm`, `ffn_gate/up/down`, `ffn_gate_inp`,
`exp_probs_b.bias`, `ffn_*_exps`, `ffn_*_shexp`, `token_embd`, `output`,
`output_norm`). Tensor types are all on paths we implement: Q5_K/Q6_K/Q8_0
attention, F32 indexer, Q4_K/Q5_K embeddings, Q4_K output, IQ2_XS/IQ2_XXS/IQ1_M/
IQ2_S/IQ3_XXS/IQ4_XS experts.

| difference | effect |
|---|---|
| `minimax-m3.attention.indexer.first_layer` **absent** (D3 writes 0, their indexer tensors exist on all 60 layers) | **bug**: `config.cpp` defaults it to `leading_dense` = 3, so layers 0-2 would run without MSA. Fix the default to 0 (or infer it from the presence of `blk.0.indexer.*`) |
| `minimax-m3.swiglu_alpha` / `swiglu_limit` (underscore) vs our `swiglu.alpha` / `swiglu.limit` | no effect today: we fall back to 1.702 / 7.0, the same values. A latent trap if a file ever differs; also affects our own D3 |
| no `strata.*` keys, no `general.alignment` | optional; defaults are fine |
| `tokenizer.ggml.add_bos_token` absent, `normalizer.nfc` present | serving/tokenization only; teacher-forced ids unaffected |
| `split.count` 4 vs 13 | fine |
| `mmproj-BF16.gguf` / `mmproj-F32.gguf` | our `vision.cpp` reads only F16/F32, so **mmproj-BF16 fails**; use F32 (1.74 GB) |

Draft GGUF: **no.** It is `general.architecture = dflash`, with
`dflash.*` keys, `blk.N.*` tensors, `conf_proj.*`, `markov_w1/2` in Q8_0 and F32
norms. Our loader (`src/minimax/dspark.cpp`) requires `dspark.*` keys,
`dspark.tap_layers`, `layers.N.*`, `confidence_head.*`, F16 norms and F16 Markov
weights, plus `hidden_norm.weight`. A repack (section 4) is needed; the target
GGUF does not need one.

## 4. DSpark: theirs vs ours

Sources: PR #31192 (`conversion/minimax.py`, `conversion/qwen.py`),
`src/models/dflash.cpp` on master, the draft GGUF header, `docs/P13_DSPARK.md`,
`src/minimax/dspark.cpp`.

| point | llama.cpp PR #31192 | Sparsa |
|---|---|---|
| tap layers | checkpoint config `dspark_target_layer_ids = [3,17,31,45,59]` — **the same ids as ours**; the GGUF stores `dflash.target_layers = [4,18,32,46,60]` because the converter adds 1 for llama.cpp's `target_layers` convention. Tap is the block input (`layer_inp`); our 62% acceptance independently confirms the input convention | **3,17,31,45,59**, input residual; confirmed against the FP32 reference (cosine 0.999997 at L3, 0.982 at L59) |
| mask | `tokenizer.ggml.mask_token_id = 200058`; anchor first, mask rows after; full non-causal attention inside the block; SWA 4096 | same mask id (default 200058), anchor first, non-causal block over the ring; `window` 4096 |
| Markov bias | `bias[v] = <w2[v], w1[prev]>`, added to the logits, chained by `argmax` per position. `w1` `[rank, vocab]`, `w2` `[rank, vocab_draft]`. Computed **in-graph**, full vocab | same orientation after the Round-5 fix, but computed **on the host**, over candidates within 12 nats of the row max (cap 8192), then argmax |
| draft count | `n_block_tokens = n_max` rows (`sample_from_anchor=true`). `--spec-draft-n-max N` runs N rows and yields N drafts; max 7 (`block_size`) | `DsparkDrafter::draft()` has **no n argument**: always `rows() = block = 7`, 7 Markov picks, 7×200k logits copied D2H, then the verifier uses `want+1` |
| confidence | confidence head optional; `--spec-draft-p-min` (their command uses 0 = off) | confidence head required; `--minimax-spec-min-conf 0.9`, `--minimax-spec-n-min 1` |
| acceptance | not published; only "greedy decoding with DSpark is identical to plain greedy" | **13/21 = 61.9%**, 1.68 tokens/round, output token-identical |

**Why n-max 1 is their fastest.** The drafter is not the cost; the verify window
is. Their own `n_block_tokens` shows the draft cost scales with `N`, and the
target verify batch is `N+1` rows. Acceptance per extra draft decays (our first
draft carries most of the 62%). On a memory-bound 2-bit MoE the target reads
each expert for the whole window, so per-row cost stays high. Our measured
accounting (`docs/P13_DSPARK.md`, Round 5):

```
rounds=19 drafts=21 accepted=13 rate=0.619 per_round=1.68
draft_ms=1689.4  verify_ms=68034.2  round_ms=69723.6  ms/token=2178.9  (plain 333)
checked_per_verify=2.11  experts_ratio=0.82
```

That is 3.58 s per 2.11-row window = ~1.70 s/row = **5.1× a plain token per
row**; 2.11 rows for 1.68 emitted tokens is 6.5× slower overall. With that
per-row penalty even N=1 is ~6× slower. On a GPU-resident target the same
2-row verify costs ~1.1 tokens, and N=1 becomes the best setting; N>1 cannot
pay for the extra rows at 60-70% acceptance. So n-max 1 is not a property of
DSpark, it is the only setting that can win when verify is cheap — and on our
box none can win until the batched verify path is fixed.

**What this reveals about ours**

1. **No n-draft control** in `DsparkDrafter::draft` (always 7 rows). Real waste,
   but only ~89 ms/round (2.4% of the round), so it does **not** explain the 6.5×.
2. The 6.5× is the **verify path**, not DSpark: 10.75× window penalty. The
   zero-copy lane is the prerequisite; changing n will not save it.
3. The Markov bias is host-side and candidate-truncated. At cap 8192 it is
   almost never truncated, so it is not a correctness bug in practice, but it is
   serial work and it diverges from the reference's full-vocab bias on very flat
   rows **(inferred)**.
4. The confidence gate can cut the window to 1 draft even when 2-4 would be
   accepted. It matches their n=1 default in practice but adds a head that
   llama.cpp leaves optional.
5. The draft file's names/dtypes are a hard incompatibility. Repack:
   `blk.N.attn_norm` → `layers.N.input_layernorm`,
   `blk.N.ffn_norm` → `layers.N.post_attention_layernorm`,
   `blk.N.attn_{q,k,v}` → `layers.N.self_attn.{q,k,v}_proj`,
   `blk.N.attn_output` → `layers.N.self_attn.o_proj`,
   `blk.N.ffn_{gate,up,down}` → `layers.N.mlp.{gate,up,down}_proj`,
   `blk.N.attn_{q,k}_norm` → `layers.N.self_attn.{q,k}_norm`,
   `output_norm` → `final_norm`, add `hidden_norm` (their `enc.output_norm` is
   the encoder's, not the same tensor), `conf_proj.*` → `confidence_head.*`,
   dequantize `markov_w1/2` Q8_0 → F16 and norms F32 → F16, add
   `dspark.markov_rank=256`, `dspark.tap_layers=[3,17,31,45,59]` (from
   `dflash.target_layers` minus 1), `dspark.mask_token_id` (from
   `tokenizer.ggml.mask_token_id`), `dspark.block_size`,
   `dspark.attention.sliding_window`, `dspark.swiglu_alpha/limit`,
   `dspark.vocab_size`.

Their draft is Q8_0 (2.09 GB) vs our Q4_0 (1.16 GiB); acceptance would need a
re-measure after repacking.

## 5. What to measure, in order, and what it costs

Disk now: **E: 148.1 GiB free, D: 141.2 GiB free.**

| # | step | disk | time | why |
|---|---|---|---:|---|
| 0 | Re-score existing dumps with the conditioned metrics (`on-path`, `p≥0.9`, `p≥0.99`) and report them next to the headline in `g3_sparsa.py` | 0 | minutes | already answers most of the "metric" question; make it permanent so the next run is not misread |
| 1 | Patch `config.cpp`: `idx_first_layer` default **0** when absent (or infer from `blk.0.indexer.*`); accept `swiglu_alpha`/`swiglu_limit`; add a load self-test that asserts indexer on all 60 layers | 0 | ~1 h | without it, any Unsloth G3 run silently disables MSA on layers 0-2 |
| 2 | Download **UD-Q2_K_XL** to E:, verify blob hashes, load-only + 8-token eval smoke | **134.0 GiB** | ~1.5 h at 25 MB/s + 5 min | the exact file behind 91.5%; 2 GiB smaller than the D3 we already run |
| 3 | G3 six short prompts on UD-Q2_K_XL, same flags as the D3 run | +0.6 GB dumps | ~35 min | decisive: if it lands ~60-63%, the metric is the whole story; if ≥80%, our allocation/quant is the problem |
| 4 | Only if step 3 shows a gap: build "D3-UD" (attention Q5_K, shared Q5_K, embd Q4_K, output Q4_K, indexer F32, down L52/58/59 IQ4_XS, gate/up L59 IQ3_XXS). Reuse `mix_layers.py` where a source pack exists, otherwise requant | 250 GB source or an existing tier | ~1 day | isolates allocation from quantizer; expected +? (inferred) |
| 5 | DSpark: repack their 2.09 GB draft, add `n` to `draft()`, run n=1..7 acceptance on D3 | **1.95 GiB** | ~1 h repack + gate | only worth it after the zero-copy verify path; otherwise it stays 6.5× slower |

**Which file fits the 128 GB box best.** `UD-Q2_K_XL` (134.0 GiB) is the right
one to measure: it is smaller than the D3 pack already running on the same
machine (136.0 GiB), so it fits at least as well, and it is the file the 91.5%
refers to. Put it on **E:** (148.1 GiB free, 14.1 GiB left). `UD-IQ2_XXS`
(125.7 GiB, 22.4 GiB left on E:) is the safe daily-driver fallback if 14 GiB of
headroom is too tight for 16K KV + dumps. Do not start with `UD-Q4_K_XL`
(232.3 GiB) or `UD-Q8_K_XL` (241.3 GiB): neither fits RAM+VRAM or disk.

Their own budgets: 1-bit 134 GB, 2-bit 150 GB total (weights + 20-32 GiB), Q2_K_XL
"160 GB+". The box has 128 GB RAM + 24 GB VRAM = 152 GB gross. Q2_K_XL will
therefore run at 8-16K context with one slot and no vision; it is not a
comfortable 32K configuration.

## 6. Sources

* Unsloth model repo (tree + headers): https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF
* Unsloth guide: https://unsloth.ai/docs/models/minimax-m3.1-flash (.md variant fetched)
* Unsloth Dynamic 3.0 / Divergence-300: https://unsloth.ai/docs/basics/unsloth-dynamic-v3.0-gguf
* llama.cpp PR #31192: https://github.com/ggml-org/llama.cpp/pull/31192
* DSpark runtime: https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp
* DSpark spec loop: https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp
* Converter: PR head `danielhanchen/llama.cpp@6b95a990`, `conversion/minimax.py`, `conversion/qwen.py`
* Draft config: https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/raw/main/dspark/config.json
* Draft GGUF header: `unsloth/MiniMax-M3.1-Flash-GGUF/dspark-MiniMax-M3.1-Flash-Q8_0.gguf`
* Ours: `tools/minimax/g3_sparsa.py`, `_local/g3_iq2xs/tools/compare.py`,
  `_local/g3_iq2xs/results_d3.md`, `tools/minimax/g3_results.md`,
  `_local/ref_out/RESULTS.md`, `tools/minimax/quant/recipes/MM-IQ2_XS-D3.json`,
  `src/minimax/config.cpp`, `src/minimax/engine.cpp`,
  `docs/P13_DSPARK.md`, `src/minimax/dspark.cpp`
