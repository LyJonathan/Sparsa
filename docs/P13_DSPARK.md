# P13: DSpark speculative decoding for MiniMax-M3.1-Flash

**Status: opt-in, and it stays opt-in.** The machinery is complete and exact in the sense that it accepts only what
the target agrees with, but the two gates the brief asks for are **not** met on this box:

| gate | result |
| --- | --- |
| greedy output token-identical to non-speculative decode | **no** - the batched verify pass is not bit-identical to the token path (KL 4.1e-3, top-1 93 % at 60 positions; first token difference at index 15 of 32) |
| > +5 % sustained speed (8k document, ~1,500 tokens) | **no** - the verify window is **8x slower** per token on the slice (measured below); the full-model sustained test could not be run at all (the box had 0.3 GiB free RAM while the other task held its own expert tier) |

How the drafter works and how a round is put together is in `tools/minimax/README.md` (§ "P13").

## What was measured

| | |
| --- | --- |
| box | 2 x RTX 5070 12 GB, Ryzen 9 5900X, 128 GB DDR4-3200, Windows 11 |
| slice | `models/quant-d3/p5a-l0-14-iq2xs.gguf` (layers 0-14, 30.7 GB), `--gpus 0 --minimax-dense-layers-gpu 15 --minimax-tiers --minimax-vram-slots-per-layer 0 --minimax-ram-experts-per-layer 128 --minimax-kv vram --cpu-threads 12 --memory-limit-mib 40000 --minimax-prefill-batch 512`, ctx 1024, 60 real-text tokens |
| drafter | `dspark-q4_0.gguf`, 1.16 GiB, `--speculative 4` (the window was cut to 1 draft by the confidence gate) |
| full model | `D:\hf-d3\IQ2_XS-D3\...-00001-of-00013.gguf` + `mm3-d3.bin`: the drafter **loads** (`dev=1 ... vram=1191 MiB taps=3,17,31,45,59`, RAM tier 87.2 experts/layer), but no generation completed (below) |

### The slice, plain vs DSpark (same binary, same placement)

| | plain | DSpark |
| --- | ---: | ---: |
| decode | **15.1 tok/s** (8 tokens, 531 ms, 66 ms/token) | **1.8 tok/s** (8 tokens, 4460 ms, 548 ms/token) |
| per token: attn | 8.1 ms | (verify window) 31-38 ms / window |
| per token: CPU routed experts | 34.7-56.4 ms | - |
| per token: shared expert | 4.2 ms | - |
| per token: lm_head | 2.0 ms | - |
| draft (drafter forward + head) | - | 36 ms / round |
| verify (one batched forward) | - | 512 ms / round (2 rows) |
| acceptance | - | 0/7 drafts |

The acceptance is 0 by construction here: the slice stops at layer 14 while the pack taps layers 3, 17, 31, 45, 59,
so `--tap-layers 2,4,6,8,10` was used and the drafts are meaningless. The slice therefore measures the **cost** of a
window, not its payoff.

### Why the window costs what it costs

* **The CPU expert path re-decodes each weight row per token.** `moe_batch_cpu` reads an expert's bytes once per
  window (that part works: 4 picks x 2 rows = ~7 distinct experts a layer, so ~1.75x one token's bytes) but calls
  `ikq::dot_row` once per (expert, row) - the IQ2_XS block is decoded again for every token. Two rows therefore cost
  ~3.5x one token's CPU work (7 experts x 2 rows against 4 x 1), and the measured 400 ms of FFN time per window is
  exactly that. **This is Maya's `MAYA_GLM_CPU_MT`**: decode a weight row once for up to 8 tokens (bit-identical to
  ggml's one-pair dots). Without it a window can only lose on a RAM/CPU-bound box.
* **The dense projections pay a dequantize pass per window.** `forward_batch` projects through cuBLAS on
  dequantized fp16 tiles; for 2 rows the dequantize is most of the cost (attn 31-38 ms a window against 8.1 ms for a
  token). The token path's `native_mmvq` reads the quantized matrix once and is the cheaper shape for a small window.
* The GPU expert path is much worse still for a small window (`gemm_expert_gpu` dequantizes each expert for a couple
  of rows), so `forward_batch` now takes it only from `kGpuExpertsMinRows = 16` rows up.

### Exactness

The verify pass is `forward_batch`, the batched prompt path. It is not bit-identical to the token path, and with the
same expert placement on both sides (CPU Q8_K experts, i.e. **not** the q8_1 device kernel the brief excludes):

| comparison (60 positions, slice) | max abs | KL mean | KL max | top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| batched forward (CPU experts) vs token path | 1.36 | **4.1e-3** | 3.9e-2 | **93.3 %** (4/60) |
| batched forward (GPU experts) vs token path | 1.47 | 3.6e-3 | 3.9e-2 | 95.0 % (3/60) |

and in the decode test the first differing token is #15 of 32. The drafts are not the cause: `STRATA_DSPARK_CORRUPT=1`
replaces every draft with a wrong token, so every round rejects and every round rolls its K/V (and indexer keys) back,
and that arm is **byte-identical to the ordinary DSpark arm** - the rollback machinery is consistent; the residual
difference from plain decode is the batched forward's own arithmetic (the router's 4th pick flipping on a
last-bit difference, which the 1.36 max-abs is the signature of).

Making the window bit-identical is the work the Qwen family already did in `src/core/verify.cpp`: a multi-token
kernel for every stage whose per-token arithmetic *is* the single-token kernel's (multi-column MMVQ in exact mode, the
per-token attention, the per-token shared expert, and the multi-token CPU expert rows). Until that exists for the
MiniMax engine, the honest statement is the KL above, not identity.

### The full model

The drafter loads and the tiers plan it in (`vram=1191 MiB` on the drafter's device, the RAM tier keeps
87.2 experts/layer). Two attempts to generate did not complete: one reached `READY` and produced no tokens, the other
died with `minimax SSD read: I/O failed` at `avail=0.3 GiB` - the box was out of RAM because the other task on the PC
(`pc-engine.lock` = `maya`) was holding its own pinned expert tier at the same time. The full-model numbers
(exactness, acceptance with the real taps, the 8k-document sustained test) are therefore **not** available yet; the
harness that produces them is `tools/minimax/bench_dspark.py --doc README.md --doc-tokens 8000 --decode 1500
--temperature 0.7 --reps 3`.

## VRAM and expert slots

* Drafter: **1191 MiB** with the Q4_0 pack (1.16 GiB), 1417 MiB with Q5_0, 2083 MiB with Q8_0. The block's ring and
  scratch add ~60 MiB on top (included).
* The head is **not** copied: when the engine already keeps `output.weight` on the device (`output_gpu=1`, which the
  full-model placement does), the drafter reads that copy; otherwise it uploads its own (1.0 GiB at Q6_K).
* 1191 MiB is ~1.4 IQ2_XS experts per layer of the VRAM pool (an expert is 14.6 MB): on GPU1 of the full model the
  free VRAM after the weights went from 3164 MiB to 1973 MiB with the drafter loaded, i.e. the drafter costs about one
  and a half expert slots a layer. The RAM tier was unaffected (87.2 experts/layer with it, the plan is RAM-bound).
* The verify window needs the batched-prompt workspace for the whole decode loop (`PrefillWs`), which the default
  `--minimax-prefill-vram reserve` already sized the slots for.

## Next steps, in the order they pay

1. **Multi-token CPU expert rows** (Maya's `MAYA_GLM_CPU_MT`): decode each IQ2_XS weight row once for up to 8
   tokens. Without it the window's expert cost is `rows x picks` instead of `union(rows x picks)`; with it, the
   measured 400 ms FFN per 2-row window should land near 100 ms and a 4-draft window becomes the cheap shape the
   brief assumes.
2. **Small-window dense projections through `native_mmvq`** (the token path's kernel, one read per window, and
   per-column bit-identical by `native_mmvq_set_multi_exact`), which also removes the first exactness gap.
3. Then: the per-query attention and the per-token shared expert in the window (the remaining exactness gaps), the
   sampled-mode distribution check on the slice, and the full-model sustained calibration.

---

# Round 2: the window's arithmetic

The gates in round 1 said the window was slower and not bit-identical. Both came from the same two places: the CPU
expert pass decoded each weight row once per token, and the batch path projected through cuBLAS/fp16 instead of the
token path's kernels. This round changes both, and the exactness work is what makes the speed work meaningful.

## 1. Multi-token CPU expert rows (`src/kernels/cpu/iq_mt.cpp`)

`gemm_cpu_base` used to call ggml's one-pair `vec_dot` once per (row, token). `mt::dot_rows` decodes a row's IQ2_XS
block ONCE - the 512-entry grid lookups, the sign masks, the scale bytes - and then runs the same integer
multiply-accumulate for each token of the window. The per-column instruction sequence is ggml's, unchanged
(`maddubs` on (grid, signed q8), `madd` with the sub-block scales, `sumi1`/`sumi2`, the per-block `fma` into an 8-lane
float accumulator, one horizontal sum at the end), so a column's result is bit-identical to ggml's own dot.

**Gate (unit, CPU only, no engine and no GPU: it needs no PC lock)** - `build/iq_mt_test.exe` builds random IQ2_XS rows
and Q8_K activations with ggml's own quantizers and compares each column against ggml's `vec_dot` for the same column:

```
iq_mt_test --rows 800
iq_mt_test: 15328 columns over IQ2_XS rows at 3072 and 6144 values, 0 mismatches (0 with a value difference)
PASS (bit-identical to ggml's one-pair vec_dot)
```

The same kernel serves a prompt chunk in groups of 8 columns, so small-batch prefill gets the shared decode too; the
prefill A/B is `tools/minimax/p13_gates.sh` step 4 (`STRATA_IQ_MT=1` against `=0`, same binary).

## 2. Small-window dense projections through `native_mmvq`

For a window of at most 8 rows, `forward_batch` now projects with `quantize_q8_1_rows` + `native_mmvq` (the token
path's own kernel; its multi-column layout is bitwise equal to its single-column one by contract) instead of
dequantize + cuBLAS: query/key/value, the indexer's q/k, the output projection and the head. That is one read of each
matrix per window instead of a dequantize pass, and it is the token path's arithmetic by construction.

## 3. The rest of the window is the token path's own code

* the routed experts accumulate in ROUTE order (`moe()` adds them that way; float addition is not associative),
  through the multi-token row above;
* the router takes `gemv` one row at a time (the token path's call) instead of a batched CPU GEMM;
* the shared expert takes `mlp()` one row at a time (device gemvs plus the host swigluoai, which the batched
  tensor-core pass does not reproduce);
* a dense layer's FFN takes `mlp()` per row;
* the attention takes `attend()` per row - the token path's block selection, gather and MSA kernel - from f32
  queries, instead of the batched MSA kernels.

## Status of the gates in this round

| gate | state |
| --- | --- |
| multi-token row equals ggml's dot (unit) | **PASS**, 0 mismatches, above |
| window (8 rows) equals the token path, same placement (`gate_logits`, max abs 0.0) | **not run**: `pc-engine.lock` said `maya` for the whole round-2 window (an engine of the other task was resident throughout; available RAM 0.3-1.2 GB), so no engine was started. `tools/minimax/p13_gates.sh` runs it |
| 200-token prefill A/B | **not run**, same reason (step 4 of the script) |
| greedy speculative == plain decode | **not run** (step 5 of the script) |
| full-model sustained test + sampled-mode distribution check | **not run** (`bench_dspark.py --doc README.md --doc-tokens 8000 --decode 1500 --temperature 0.7 --reps 3`, and `p13_sample_dist.py`) |
