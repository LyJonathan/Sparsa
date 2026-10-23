# Imatrix value and tail blending: measured on MiniMax-M3.1-Flash layer 40

Experiment B of `_ref/research/QUANT_RESEARCH.md` section 6.4. One MoE layer
(source layer 40), all 128 routed experts, all three matrices
(`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`), measured five ways:

| run | type | importance |
|---|---|---|
| a | IQ2_XS | real, 24,576-token imatrix |
| b | IQ2_XS | uniform (`--write-test-imatrix`) |
| c | IQ3_XXS | real |
| d | IQ2_XS | real + tail blend K=200, every expert (`--imatrix-min-count 0`) |
| e | IQ2_XS | real + tail blend K=200, experts with <500 tokens only |

Each matrix is quantized with `ggml_quantize_chunk`, then dequantized again.
Three errors are reported per matrix kind:

* **plain** - `||W - Q||_F / ||W||_F`.
* **weighted** - per row `sqrt(sum_j m_j (w_ij-q_ij)^2 / sum_j m_j w_ij^2)`,
  averaged over rows and experts, where `m` is the importance actually used.
* **output** - `||W x - Q x|| / ||W x||` over the real routed tokens. `x` comes
  from the `dumps/short/dump-L40` reference dump: `exp_input.py` rebuilds the
  expert input from `attn_input`/`attn_out`/`router_logits` and matches the
  dumped `gate_up` to about 1e-6 relative. `down` uses the dumped
  `expert_XXX_hidden`. 456 routed token instances over 75 experts.

## Results

Layer 40, 128 experts, 456 routed token instances over 75 experts. `plain` and
`weighted` are over all 128 experts; `output` is over the routed tokens.

| run | type | matrix | plain | weighted | output |
|---|---|---|---:|---:|---:|
| **a** | IQ2_XS, real | gate | 0.3079 | **0.2950** | **0.2135** |
| | | up | 0.3078 | **0.2950** | **0.2974** |
| | | down | 0.3380 | **0.2483** | **0.2820** |
| **b** | IQ2_XS, uniform | gate | 0.3048 | 0.3048 | 0.2184 |
| | | up | 0.3048 | 0.3048 | 0.3027 |
| | | down | 0.3051 | 0.3051 | 0.3080 |
| **c** | IQ3_XXS, real | gate | 0.1864 | 0.1781 | 0.1288 |
| | | up | 0.1864 | 0.1782 | 0.1798 |
| | | down | 0.2101 | 0.1451 | 0.1692 |
| **d** | IQ2_XS, real, blend K=200, all | gate | 0.3074 | 0.2962 | 0.2133 |
| | | up | 0.3073 | 0.2962 | 0.2972 |
| | | down | 0.3133 | 0.2807 | 0.2837 |
| **e** | IQ2_XS, real, blend K=200, count<500 | gate | 0.3075 | 0.2959 | 0.2134 |
| | | up | 0.3075 | 0.2959 | 0.2973 |
| | | down | 0.3218 | 0.2707 | 0.2821 |

### Verdict on the imatrix path

The real imatrix wins on both metrics that matter for outputs. Against the
uniform matrix:

| matrix | weighted | output |
|---|---:|---:|
| gate | -3.2% | -2.2% |
| up | -3.2% | -1.8% |
| down | -18.6% | -8.5% |

Plain (unweighted) Frobenius error goes the other way - gate +1.0%, down
+10.8% - which is what a weighted quantizer should do: it spends error where
activations are small. The imatrix path is **not** buggy. The uniform run
confirms it numerically: with weights of 1.0 the weighted metric equals plain
to 7e-6.

Checks made on the file and the reader:

* 711 entries; names `blk.{i}.ffn_{gate,up,down}_exps.weight` match what the
  quantizer asks for (`Item.gguf`, which keeps `.weight`).
* `.in_sum2` is `[128, n_in]` expert-major, `.counts` `[128]`; row 0 and row 1
  differ by 21% in relative norm and every row is finite and positive.
* Normalization is `in_sum2 / counts` per expert, the llama.cpp convention.
* Zero-count experts were falling back to a row of zeros, not 1.0. They now
  take the per-layer mean importance, with or without blending. There were 3
  zero-count experts in the whole file.

Caveat: one prompt (114 tokens, 456 routed instances) and 75 of 128 experts.
The down gain is large enough to be robust; the 2% gate/up gains are small.

### Verdict on tail blending

Blending does not improve layer output. With K=200 on every expert (d) the
output error moves by +0.1% .. -0.1% against (a); restricted to experts with
fewer than 500 tokens (e) it is within 0.05% either way. What it does change is
the error split: plain Frobenius error falls (down -7.3% for d, -4.8% for e)
while the imatrix-weighted error rises (down +13.0% for d, +9.0% for e).

The reason is visible in the token counts: on this prompt the 19 experts with
fewer than 200 calibration tokens carry 1.8% of the routed instances; the 67
experts under 500 carry 20.2%. Blending can only change what those tokens see.
Layer 40 is also less tail-heavy than layer 30 in the study (min 10, p10 141,
median 479, max 5916, no zero-count experts), so there is little headroom.

Recommendation: keep blending off by default (it is), keep the option for
corpora that leave more of the distribution uncalibrated, and re-measure it
with a per-expert output metric and a 250k-token imatrix before using it in a
tier.

### Runtime

One process at a time, 12 threads, on a Ryzen 9 5900X that was running another
quantization job for most of the sweep. 29.0 GB of logical f32 per config
(3 x 128 x 3072 x 6144).

| run | wall | quant only | effective (wall) |
|---|---:|---:|---:|
| a | 23m04s | 1227 s | 0.021 GB/s |
| b | 22m22s | 1200 s | 0.022 GB/s |
| c | 12m01s | 560 s | 0.040 GB/s |
| d | 27m21s | 1412 s | 0.018 GB/s |
| e | 26m26s | 1343 s | 0.018 GB/s |

IQ3_XXS is about twice as fast per logical byte as IQ2_XS in this tree. The
IQ2_XS spread is contention, not the type.

## Tail blending implementation

`--imatrix-min-count N --imatrix-blend K` in `strata-mm-quant`:

```
w = (c * w_e + K * w_mean) / (c + K)
```

`w_mean` is the mean of the expert rows of the same tensor with `c > 0`.
`N = 0` blends every expert; `N > 0` only experts with fewer than `N`
calibration tokens. Layer 40 token counts (gate/up/down are identical):
min 10, p10 141, median 479, max 5916; 67 of 128 experts below 500 tokens,
19 below 200.

