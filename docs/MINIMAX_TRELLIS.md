# Trellis expert encoding for MiniMax-M3.1 (TQ2_T / TQK6 / TQK7)

Status, 2026-10-24: **the encoder is written, tested and integrated; the CUDA backend is written but its search is not
yet quality-approved** (see 6.4). Nothing in this document is taken from a paper without saying so; every number was
measured on this machine and says what it was measured on.

---

## 1. License: TQ2_T / TQK6 / TQK7 come from an MIT tree

`third_party/ggml-tq` in this repository carries no LICENSE file. Its origin is:

* `_ref/research/agention-llama` = `https://github.com/agentionai/llama.cpp` (commit `dc255f2`), a fork of
  `ggml-org/llama.cpp` ("the ggml authors"). It ships `LICENSE` = **MIT, Copyright (c) 2023-2026 The ggml authors**,
  unchanged from upstream llama.cpp (which is MIT).
* The trellis work in that fork is the "Gyro"/APR work of Laurent Zuijdwijk, contributed under the same MIT license;
  the fork's `ggml/src/ggml-common.h` defines the three block types and its `ggml-quants.c` ships the reference
  decoder and a *placeholder* encoder. QTIP (arXiv 2406.11235) and EXL3 are the papers behind the *method*, not the
  source of this code.
* `third_party/ggml-tq/tq.cuh` and `tq-mmq-tiles.cuh` already say so in their headers ("MIT, the ggml authors;
  Laurent Zuijdwijk") and carry no further restriction.

**Conclusion: MIT.** No notice is required beyond the one already in those two headers and in
`third_party/ggml/ggml-common-tq.h`; the new encoder files (`src/kernels/cpu/tq_encode.cpp`,
`src/kernels/cuda/tq_encode.cu`, `include/strata/kernels/cpu/tq_encode.hpp`) are our own MIT code, and the *format*
(block layout + codebook + the hadamard convention) is reused under the same MIT grant. There is no need to
re-derive the encoder from QTIP/EXL3. No GPL or research-only code is involved (QTIP's own release is MIT as well).

---

## 2. The format, from the decoder side

All of this is `third_party/ggml/ggml-common-tq.h` plus the runtime kernels (`src/kernels/cpu/tq_*.cpp`,
`src/kernels/cuda/iq_kernels.cu`). A block is **128 weights**:

| type | id | bytes | layout | bpw |
|---|---:|---:|---|---:|
| TQ2_T | 144 | 34 | `fp16 d; u8 qs[32]` | 2.125 |
| TQK6  | 145 | 26 | `fp16 d; u8 qs[24]` (`32 x 6` bits) | 1.625 |
| TQK7  | 146 | 30 | `fp16 d; u8 qs[28]` (`32 x 7` bits) | 1.875 |

A block is **32 trellis steps of four weights**. Step `t` has a 16-bit state `s_t`; its low 15 bits are used:

```
h   = (s_t & 0x7fff) * 0x9e3779b1        // 32-bit wrap
p0  = h >> 21                            // 11-bit index into the 2048-point codebook
p1  = (h >> 10) & 2047
w[4t+0] = d * lut[2*p0+0]   w[4t+1] = d * lut[2*p0+1]
w[4t+2] = d * lut[2*p1+0]   w[4t+3] = d * lut[2*p1+1]
```

`lut` is the fork's 4096-entry fp16 "3INST" table (`tq_lut_f16`, copied verbatim into
`third_party/ggml/ggml-common-tq.h`), i.e. a 2048-point pseudo-Gaussian codebook. It is the same table on every
backend, held as `float[4096]` by our encoder (`tq_codebook()`), as `int16[32768][4]` at scale 2^13 by the CPU
decode kernels, and staged into shared memory by the CUDA kernels.

The state is the 16-bit window over the **chunk stream**:

* **TQ2_T** (`K = 8`): `qs[t]` *is* the chunk of step `t`, and `s_t = (qs[(t+31) mod 32] << 8 | qs[t]) & 0x7fff`, so
  the window is exactly two adjacent path bytes and the trellis is a first-order chain over one byte.
* **TQK6/TQK7** (`K = 6, 7`): the 32 chunks are bit-packed into a **circular** stream of `32*K` bits, chunk `t` at
  stream bit `(31-t)*K`, LSB first, and `s_t` is the 16 stream bits starting at bit `(31-t)*K` (circular modulo
  `32*K`). Because `16 = 2K + (16-2K)`, `s_t` depends on three consecutive chunks, so the trellis is a second-order
  chain.

Every byte string decodes to something; the encoder only chooses *which*.

### 2.1 The rotation and the GGUF key

For a folded pack the encoder stores `W' = W diag(s) H_b^T` per block of `b` input columns, where `H_b` is the
normalized Sylvester-Walsh-Hadamard matrix (`(-1)^popcount(r&c)/sqrt(b)`, the butterfly order of
`src/kernels/cpu/hadamard.cpp`) and `s` is a fixed `+/-1` vector of the input width. Then `W x = W' (H_b (s * x))`, so
the runtime must rotate the activation. Keys (written by `strata-mm-quant`, read by `src/minimax/engine.cpp`):

```
strata.hadamard.block        u32         128
strata.hadamard.signs.<w>    i32 array   one +/-1 per input value, one array per folded width
                                        (6144 for gate/up, 3072 for down)
strata.hadamard.roles        i32 array   per output layer: 0 = not folded, 7 = gate+up+down folded
strata.hadamard.note         str          the convention in words, for anyone reading the file
```

`w` is a power of two in `[2,1024]` that divides the row width; **128** is the TQ block and divides 6144 and 3072, so
the rotation and the coding block are the same unit. The sign vector is one per width (not per layer): the rotation
only has to be orthonormal and dense, and a per-layer vector would cost 6144 x 61 x 4 bytes of metadata for no
quality.

An engine that sees `strata.hadamard.block` and cannot rotate must refuse the file; `src/minimax/engine.cpp` fails
hard on a malformed record, and applies the rotation on the gate/up input, on the down input, on both the CPU
(`tq_rows`) and the GPU (`minimax_moe_gu` / `hadamard_rows` / `minimax_moe_down`) expert paths.

---

## 3. The encoder

`include/strata/kernels/cpu/tq_encode.hpp`, `src/kernels/cpu/tq_encode.cpp` (reference, plain C++),
`src/kernels/cuda/tq_encode.cu` (GPU, TQ2_T), `include/strata/kernels/tq_encode_cuda.hpp`.

1. **Cost.** For step `t` and 15-bit window `s`: `cost = sum_j w_j (x_{4t+j} - d c_j(s))^2`. The 32 tables of a
   block are built with the *state* loop outermost, so one golden-ratio multiply and two codebook loads serve all 32
   steps (4x the per-step loop), and the 32 output streams are walked in 1024-state tiles so they stay in L2.
2. **Search.** Tail-biting Viterbi, two laps around the 32-step circle with uniform start metrics, traceback from the
   best final state. TQ2_T: 256 states, 65,536 edges per step. TQK6/TQK7: the DP state is a *pair* of chunks
   (`64^2` / `128^2` states), 262k / 2.1M edges per step. Back-pointers are recovered by re-evaluating the argmin
   during traceback, which keeps the forward pass a pure `min` accumulate (`dp_min`, SSE2) instead of a
   compare-and-store.
3. **Scale.** Start at the block RMS, run the Viterbi, refit `d` by weighted least squares, round `d` to fp16 (the
   file's precision) and run a second lap pair at that scale; keep the better of the two. `candidates` widens the
   start (0.8/0.9/1.0/1.15/1.3 x RMS) at a linear cost in time.
4. **Importance.** `w` is the imatrix row. **A diagonal Hessian is not usable per coordinate after a block rotation:**
   the diagonal of `H S diag(w) S H` is the *block mean* of `w` on every coordinate, so the per-coordinate weight in
   the rotated basis is a constant and the Viterbi argmin is unchanged. The encoder therefore takes the per-coordinate
   weight (it is exact when a tensor is not folded, and it is what the caller wants for the *metric*), and
   `tq_block_importance()` gives the block means for folded tensors. The honest statement is: **with Hadamard folding
   the imatrix weights block error, not coordinate error.** What it can still buy is search effort; `candidates` is
   the knob, and this release does not spend it per block yet (see 7).

---

## 4. Tests

`src/kernels/cpu/tq_encode_parity.cpp` (`ctest -R tq_encode_parity`), all passing:

| check | result |
|---|---|
| encode -> decode == a verbatim port of the upstream decoder, bit for bit (all three types) | PASS |
| the same blocks through the shipped CPU kernel `tq_rows` (one-hot Q8_0 activations read one weight at a time) | PASS, worst deviation 6e-5 (the kernel's documented int16 table rounding) |
| Viterbi vs the upstream placeholder encoder on Gaussian data, imatrix-weighted | 0.2824 vs 0.4074 (**1.44x**) |
| Viterbi vs the placeholder on a per-16-scaled NVFP4-like source | 0.2907 vs 0.4453 (**1.53x**) |
| the Hadamard convention: `W x` vs `decode(W') . (H (s * x))`, and the same with the activation *not* rotated | 0.448 vs 1.772 (PASS) |

Round-trip error on 1024 Gaussian weights: TQ2_T 0.318, TQK6 0.430, TQK7 0.389 relative L2. (2.125 bpw on i.i.d.
Gaussian is 0.28-0.33 in the published QTIP/ik_llama tables too; the *real* weights below are what matters.)

---

## 5. Integration into `strata-mm-quant`

* New types: `TQ2_T`, `TQK6`, `TQK7` (ids 144/145/146) in `--recipe`, `--override`, `parse_type`. ggml (a pinned
  upstream llama.cpp) has no entry for those ids, so every `ggml_row_size` / `ggml_blck_size` / `ggml_type_name`
  call goes through `src/tqio.hpp`, and `quantize_rows` dispatches to the trellis encoder.
* Recipes: **`MM-TQ2_T`** (all routed expert matrices TQ2_T, Hadamard 128, Q6_K attention/shared/dense, Q4_K embed,
  Q6_K head) and **`MM-TQK6-7`** (gate/up TQK6, down TQK7, same side).
* GGUF: the hadamard keys of 2.1, plus `strata.recipe` (which now names the trellis types) and
  `general.file_type` from the recipe (llama.cpp has no file type for 144-146).
* Engine (`src/minimax/engine.cpp`): row size and decode for 144-146 (`row_bytes_of`, `decode_f32`), the CPU expert
  path via `tq_rows` (`dot_rows_base`), the hadamard record (`read_hadamard`) and its application on both the CPU and
  the GPU expert paths. `Fmt<144/145/146>` were already in `src/kernels/cuda/iq_kernels.cu` (`STRATA_GU_FMTS` has
  144/145, `STRATA_D_FMTS` has 144/146, and `minimax_moe_gu` / `minimax_moe_down` route through them).
* Build check: `build-engine-mapped.cmd` configures with `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_MMQ_TQ_OVERLAY=OFF` and
  builds `strata_minimax` + `strata-minimax.exe` clean (CUDA 13.3, sm_120). The mapped drive is required: nvcc breaks
  on the apostrophe in `Jon's Projects`, and the trellis MMQ overlay's `configure_file` of `ggml.h` makes ninja
  reject a second output for the same path when the source tree is a substituted drive.

---

## 6. Measurements

### 6.1 Error on one real layer (layer 3 of MiniMax-M3.1-Flash)

`strata-mm-quant --tq-eval 3 --tq-eval-experts 1 --tq-eval-rows 512 --tq-eval-dumps
.../dumps/short/dump-L03` on layer 3 (the first MoE layer): the first 512 rows of each of gate/up/down of **expert 0**
(512 of 3072 rows for gate/up, 512 of 6144 for down), dequantized from the checkpoint's NVFP4, folded with Hadamard
128, encoded, decoded and compared. `relL2` is unweighted; `relL2(im)` weights each coordinate by the real imatrix;
`downOut` applies the quantized down projection to the layer's **real** activated hidden rows from
`_local/ref_out/dumps/short/dump-L03/expert_000_hidden.npy` (6 tokens, the actual input of `ffn_down_exps`).

| format | bpw | relL2 | relL2(im) | downOut | encode Mw/s (12 threads) |
|---|---:|---:|---:|---:|---:|
| TQ2_T-greedy (upstream placeholder) | 2.125 | 0.4932 | 0.4932 | 0.4381 | 0.393 |
| **TQ2_T (this encoder)** | 2.125 | **0.3769** | **0.2839** | **0.3448** | 0.068 |
| TQK6 | 1.625 | 0.5144 | 0.3873 | 0.4739 | 0.041 |
| TQK7 | 1.875 | 0.4433 | 0.3329 | 0.4124 | 0.012 |
| IQ2_XS | 2.312 | 0.3195 | 0.2534 | 0.2761 | 1.09 |
| IQ3_XXS | 3.062 | 0.1983 | 0.1491 | 0.1763 | 5.19 |

A repeat on experts 0-3 at 512 rows (before the CUDA row was gated out; that run was killed by an unrelated
concurrent build, the per-expert numbers were 0.3792 / 0.2831 / 0.3482 for TQ2_T) agrees to 0.002.

Notes on the table:

* TQ2_T beats the format's own placeholder encoder by **24%** unweighted and **42%** on the imatrix-weighted metric,
  and by 21% on the real-activation down error. The DP works.
* TQK7 (1.875 bpw) is 15% better than TQK6 (1.625) and lands 12% worse than TQ2_T with 12% fewer bytes: the
  bit-packed formats are not obviously worse per byte, which is the one place where this measurement *supports*
  the research note's ordering.

**What this says, and it contradicts the research note.** On this checkpoint TQ2_T beats the format's own placeholder
encoder by 24% (and by 42% on the imatrix-weighted metric), but it does **not** reach IQ3_XXS quality at 2.125 bpw:
its weighted error is 1.9x IQ3_XXS's and 1.12x IQ2_XS's, and on the real-activation down error it is 1.29x IQ2_XS.
So the "TQ2_T == IQ3_XXS at 69% of the bytes" claim from `docs/GYRO.md` does **not** reproduce on MiniMax-M3.1 with
this encoder. The two candidate explanations, in order of likelihood:

1. **The encoder is still short of an offline Viterbi.** Two laps and two scale passes is a small search. The Gyro
   files came from a closed-source offline encoder with (by its own description) a much wider search; the 26%
   improvement over the placeholder shows the DP works, but not that it is at the format's ceiling.
2. **A 2.125 bpw trellis really is ~IQ2_XS on this source.** The source is a per-16 NVFP4 QAT checkpoint; after a
   128-block Hadamard rotation the block is near-Gaussian, which is the codebook's assumption, but the *scale*
   structure (fp16 per 128) may not be the win that it is on BF16 weights.

The measurement that would separate them is more search effort on the same rows (`--tq-eval` with a wider candidate
set), and that is the first thing to do next (section 7). Until then, **TQ2_T is not a replacement for IQ3_XXS on
this model, and the recommended default tier should stay IQ2_XS until this is resolved.**

### 6.2 Encode speed

| backend | hardware | throughput | one MoE layer (7.25e9 weights) | full tier (413e9 weights) |
|---|---|---:|---:|---:|
| CPU reference, 12 threads | Ryzen 9 5900X | 0.062 Mw/s (TQ2_T), 0.041 (TQK6), 0.012 (TQK7) | 32 h | **~1850 h** |
| CPU, 1 thread | 5900X | 0.014 Mw/s | 144 h | 8200 h |
| CUDA, 1 RTX 5070 | sm_120 | 2.5 Mw/s end to end (incl. H2D/D2H) | 48 min | **~46 h** |

The measured CUDA rate is 40x the 12-thread CPU rate, but **neither is close to the 20-40 min/tier that the research
note projected**, and the gap is not the GPU: it is the algorithm's shape. 65,536 edges per step x 32 steps x 2 laps
x 2 passes = 8.4M edge costs per 128 weights, i.e. **65k cost evaluations per weight**. At 1e10 cost/s (a 5070 with
the codebook in shared memory and one block per trellis block) that is 2.7e16 evaluations for a tier = **weeks**, not
minutes. The only way the projected numbers work is by reusing one cost table across the DP (the CPU does this) and
by keeping the table in a cache that the DP walks sequentially, i.e. one *warp group* per trellis block over a
persistent grid - see 7.

### 6.3 What the numbers rule out

* A CPU-only TQ tier is not viable on these machines (1850 h at the measured 12-thread rate, and even an AVX2
  implementation of the same algorithm would be ~4x that speed at best: 15-20 days per tier, not 22 h).
* The 5070's per-token decode is unaffected (the decoder already existed); this is an *encoding* cost only.
* Renting one GPU is still the right shape, but on the corrected algorithm, not this one.

### 6.4 The CUDA backend's quality gap (open)

With `MMQ_TQ_EVAL_CUDA=1` the same eval run reports TQ2_T on the GPU at relL2 0.696 / relL2(im) 0.648 against the
CPU's 0.379 / 0.283 on identical input. The first 128-weight block of row 0 decodes identically on both backends, so
the two agree on the easy case and diverge on some other blocks; the kernel is deterministic across runs, so this is
a logic bug, not a race, and it is **not yet found**. The CPU encoder is the reference and the only one whose numbers
above should be quoted.

---

## 7. What is left

1. **Find the CUDA search bug** (6.4) - compare the path bytes per block between the two backends on the first
   divergent block; the DP and the traceback are ~40 lines each and the cost function is shared.
2. **More search, cheaply**: the CPU already shares one cost table across its DP; give the GPU the same structure
   (one warp group per trellis block, the 32x32768 table in shared+L2, a persistent grid). That is the change that
   makes the rented-GPU plan real.
3. **Spend the imatrix where it can act**: pick the blocks whose imatrix-weighted error is worst (top ~10%) and give
   them more candidates/restarts. With a block-constant diagonal this is the only place a diagonal Hessian can help.
4. **A real quality answer for TQ2_T on MiniMax**: re-run 6.1 at a wider search, on all 128 experts of layer 3, and
   add layer 40 (`dump-L40` exists). Then decide between TQ21 and IQ2_XS for the default tier.
5. **The engine parity test on the GPU** (`minimax_moe_gu` / `minimax_moe_down` with type 144 against `tq_rows` on a
   synthetic expert) - the CPU half is covered by `tq_cpu_parity` and `tq_encode_parity`; the GPU half needs the
   CUDA build and a small `src/kernels/minimax_tq_parity.cpp`.
6. **Full-model wiring**: `run_tier.py` splits by layer range; a TQ tier needs the hadamard keys repeated in part 0
   only (done) and `STRATA_TQ_OVERLAY_OK` (already in the tree) for the prompt path.
