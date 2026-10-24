# MiniMax engine tools

The engine lives in `src/minimax/` and `include/strata/minimax/`. P2b runs a full MiniMax-M3.1-Flash forward
on one GPU plus the CPU and speaks the serve protocol. Routed experts stay on the CPU. Prefill walks the
same per-token path as decode (a batched prefill is later work; the two are the same math).

## Build (Windows, VS 2026 + CUDA 13.3, sm_120)

```powershell
$env:STRATA_GGML_DIR = 'E:\llm\llama-pinned-3cf'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
pwsh -File tools/minimax/build_windows.ps1
```

The script maps the worktree to `M:` because nvcc drops include paths that contain the apostrophe in
`Jon's Projects`. It sets `CUDA_PATH` to 13.3 explicitly (the process environment may point at 13.1), configures
`build/` with `-DCMAKE_CUDA_ARCHITECTURES=120`, and builds `strata` and `strata-minimax`. Set `STRATA_GGML_DIR` to a
llama.cpp checkout at the pinned `third_party/ggml/VERSION.txt` commit; without it CMake fetches that commit. The
MiniMax-patched `_baseline/llama.cpp` is not a Strata build dependency (its MMQ API differs from the pinned one).

`strata-minimax` needs `STRATA_NATIVE_EXPERTS` (the default). That links ggml-cpu for the expert dots. The Qwen
expert pool is not used: its activation is SiLU, and MiniMax's is swigluoai.

## Inspect a GGUF

```powershell
build\strata-minimax.exe --model model.gguf --minimax-inspect
```

Prints every expected `minimax-m3` tensor (name, GGML type, shape, bytes, offset, class) and the byte totals for
dense / experts / embd / output. The inspect path uses the structural check only, so it can describe a file that is
not MiniMax-M3.1-Flash; `--serve` adds the exact M3.1 fingerprint check and refuses anything else. Inspect does
not initialize CUDA.

## Forward pass

Dense tensors (attention, indexer projections, norms, the leading dense MLP, shared experts, the router, the
output head) are uploaded in their GGUF bytes for the first `--minimax-dense-layers-gpu N` layers. `N=0` keeps
every dense matrix on the CPU, which is the numerical gate. The default, `-1`, places as many leading layers as
fit after the KV cache and a 512 MiB reserve. F32 is uploaded as-is. F16 and BF16 are expanded to fp32. Other
GPU-safe quantizations stay packed and are dequantized in tiles of 128 rows. A type or shape the dequant kernel
cannot take stays on the CPU even inside those N layers.

Without the P3 tiers, routed experts always run on the CPU, one ggml type per gate/up pair: `from_float` into the
vec_dot type, then `vec_dot`, then swigluoai (`clamp(gate, max=7)`, `clamp(up, -7, 7)`,
`(up+1)*gate*sigmoid(1.702*gate)`). Expert order is gate = w1, up = w3, down = w2, expert-major, not
interleaved. The router is fp32 sigmoid; the bias is added only while choosing the top experts; the unbiased
weights are renormalized and the routed sum is scaled by `expert_weights_scale` (2 for M3.1). The shared expert is
added unscaled.

Norm weights follow the llama.cpp GGUF convention. `minimax-m3.norm.add1` is authoritative when present
(`true` adds 1 at load). When the key is absent, a mean near 1 is treated as already baked and a mean near 0
is refused, unless `--minimax-allow-nonstandard` is set (the zero synthetic file).

KV, V, and indexer-K are fp16. The attention math in this build reads a host mirror of that cache and also
keeps a device copy. It is not a fused GPU attention kernel. Context length is `--max-context`, clamped to the
GGUF context. If the KV allocation fails, lower `--max-context`.

MSA runs on every indexer layer, including the dense prefix: partial RoPE on the first `n_rot` dimensions,
unscaled fp32 indexer dots, a causal max-pool per block, top-k with the query's own block forced inside the
k, then fp32 softmax over the selected tokens at `1/sqrt(head_dim)`.

## Expert tiers (P3)

The tiers are opt-in; with none of these flags a run is the P2 path above. `--minimax-tiers` turns them on with
automatic sizes. `--minimax-profile PATH` also turns them on and fills the static slots from a profile.

| Tier | Holds | Compute |
|---|---|---|
| VRAM | per-layer slots from the STRP profile (or compulsory first use) | grouped GPU kernel over the selected experts; the IQ types, Q8_0 and the ported ik_llama.cpp formats use `iq_kernels.cu`'s dot products, F32/F16/BF16 have a direct kernel; swigluoai fused into gate/up |
| pinned RAM | next-hottest experts per layer in one `PinnedArena` | the existing CPU `vec_dot` path |
| SSD | everything else, read from the GGUF shard | unbuffered `DirectFile` reads (Windows OVERLAPPED + `NO_BUFFERING`, Linux `O_DIRECT`) into 4 KiB-aligned staging, then the CPU path |

Sizing: `--vram-reserve-mib` (default 700) is kept free after dense weights, KV and workspace. VRAM slots per layer
are then `free / (MoE layers x expert bytes)`. The RAM tier uses available RAM minus a 2 GiB reserve, or
`--memory-limit-mib`; `--minimax-vram-slots-per-layer` and `--minimax-ram-experts-per-layer` override either count
(`0` disables that tier).

The profile is the same STRP file as `include/strata/core/expert_cache.hpp`: `STRP`, `1, n_layers, n_expert,
slots, n_ranked`, the ranked `(layer, expert)` uint16 pairs, then the rank table. MiniMax has one row per **MoE**
layer (57 x 128 for M3.1), not per model layer.

```powershell
# record the routing a prompt exercises, then fill from it
build\strata-minimax.exe --model m.gguf --minimax-eval ids.txt --minimax-dump-logits l.npy `
  --minimax-profile-save profile.bin --minimax-routing-counts counts.txt
python tools/minimax/make_profile.py --n-moe-layers 57 --n-expert 128 --out profile.bin counts.txt
build\strata-minimax.exe --model m.gguf --serve --minimax-profile profile.bin --minimax-stats
```

`--no-minimax-adaptive` keeps the static fill and turns off both the decayed-count VRAM swaps (every 16 tokens,
the most-routed missing expert replaces the least-used resident one, copied on a side stream) and the SSD-miss LRU
in the RAM tier. `--minimax-stats` prints `INFO minimax-stats token=... vram=... ram=... ssd=... ssd_bytes=...` on
stderr for every token and a `total` line after each `GEN` or eval; `--minimax-stats-per-layer` adds one line per
layer. None of it goes to stdout, so the serve protocol is unchanged.

Measured on the synthetic runnable model (one MoE layer, 8 experts, top-4; F32 experts), 40 tokens:
`vram_slots=2 ram_experts=3` gave `vram_hits=47 ram_hits=69 ssd_reads=44 ssd_bytes=8,650,752 swaps=2`, and the
logits were max-abs `7.2e-7`, KL `3.3e-15`, top-1 `1.000` against the all-CPU path. All-VRAM was max-abs `4.8e-7`;
all-RAM and all-SSD were byte-identical (`0.0`) to the all-CPU path. IQ2_XS on the GPU tier (hidden 512, expert
FF 256, so neither dimension is the Qwen geometry) measured max-abs `2.1e-2`, KL `1.3e-5`, top-1 `1.000` against
the CPU IQ2_XS path (GPU iq kernels quantize the activation with q8_1, the CPU path with ggml's Q8_K). Q8_0 matched
to max-abs `3.6e-7`.

## Serve

```powershell
build\strata-minimax.exe --model model.gguf --serve --max-context 32768 --minimax-dense-layers-gpu -1
```

Prints `INFO engine=minimax-m3 ...`, an `INFO` line from the engine (`prefill=token`, how many dense layers
landed on the GPU), and `READY <ctx> stop`. `GEN` streams `T <id>`, `PP <done> <total> <ms> <tok/s>`, and
`DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> 0 0 <reused>`. `finish` is `stop`, `length`,
`cancel`, or `nonfinite`. `STOP` during a request finishes it with `cancel`. `GENI` and `BGEN` return `ERR`
and no `DONE`. Unknown options are ignored in `--serve` mode, because the server's config is shared with the
Qwen families.

Sampling keys on `GEN`: `temperature` (0 is greedy argmax, ties take the lowest id), `top_k`, `top_p`,
`min_p`, `penalty_last_n`, `penalty_repeat`, `penalty_freq`, `penalty_present`, `seed`.

`--minimax-allow-nonstandard` serves a reduced test GGUF; `--serve` otherwise refuses it.

## Teacher-forced logits

```powershell
build\strata-minimax.exe --model runnable.gguf --minimax-allow-nonstandard --minimax-dense-layers-gpu 0 `
  --minimax-eval tokens.txt --minimax-dump-logits logits.npy --minimax-last-k 0 --max-context 512
```

`--minimax-eval` is exclusive of `--serve`. The file is whitespace or comma separated token ids (`#` comments).
The npy is fp32, shape `[rows, vocab]`, one row of next-token logits per input position (or the last K when
`--minimax-last-k` is positive).

## Synthetic GGUF and the parity check

```powershell
python tools/minimax/make_synthetic_gguf.py synthetic.gguf
python tools/minimax/make_synthetic_gguf.py runnable.gguf --runnable
python tools/minimax/make_synthetic_gguf.py runnable-q8.gguf --runnable --quant-experts q8_0
python -m pytest tools/minimax/test_inspect.py tools/minimax/test_forward.py -q
python -m pytest tools/minimax/test_tiers.py -q
python -m pytest serve/test_minimax.py -q
```

The default generator is the zero loader fixture (2 layers). `--runnable` is 4 layers, hidden 256, vocab 128,
8 query heads / 2 KV heads, head dim 32, rotary 16, dense MLP width 64 on layers 0-2, 8 experts top-4 on
layer 3, indexer block 8 with top-16. Twelve tokens keep every block; 200 tokens make the indexer drop
blocks. `tiny_ref.py` is a separate fp32 implementation of the spec on those same bytes. The CPU placement
is the match that must hold. On the runnable F32 file that check measured max abs 3.4e-5 (12 tokens, every
indexer block kept) and 4.5e-5 (200 tokens, sparse), KL under 3e-11, top-1 agreement 1.0. The same short
prompt with all four dense layers on the GPU measured max abs 4.7e-5. Q8_0 routed experts against a
dequantized-weight reference (the engine also quantizes the activation) measured max abs 1.5e-2, top-1 1.0.

## ik_llama.cpp expert types (IQ2_KT, IQ3_K, IQ4_KS, IQ3_KT, IQ2_KL)

`tools/minimax/quant` emits five expert formats ported from
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow (MIT; see `THIRD_PARTY_NOTICES.md` at
the repository root). They are the best quality per byte on the MiniMax-M3.1 routed experts that
`_ref/research/IQK_EXPERIMENT.md` measured, and they beat every mainline type at the same size on layer 40:

| recipe | experts | bpw | projected size | out error on layer 40 (ik's harness) |
|---|---|---:|---:|---:|
| `MM-IQ2_KT` | IQ2_KT | 2.125 | 123.4 GB | 0.2625 |
| `MM-IQ2_KT-IQ3_KT-down` | IQ2_KT gate/up, IQ3_KT down | 2.458 | 140.6 GB | 0.2215 |
| `MM-IQ2_KL` | IQ2_KL | 2.6875 | 152.4 GB | 0.2047 |
| `MM-IQ3_K` | IQ3_K | 3.4375 | 191.2 GB | 0.1156 |
| `MM-IQ4_KS` | IQ4_KS | 4.25 | 233.1 GB | 0.0608 |

The sizes are `51.625 x bpw + 13.7` GB from `IQK_EXPERIMENT.md` (1.652e12 routed-expert values at 1 bpw plus the
dense remainder); `--dry-run` prints the exact tensor bytes (123.811, 141.025, 152.680, 191.230 and 233.547 GB
with the default Q8_0 dense side).

These recipes run in Strata only. Neither mainline llama.cpp nor ik_llama.cpp can read the files: the type ids
are Strata's own (160..164 in `include/strata/kernels/iqk_types.hpp`, chosen because ik's 144/145/146 collide
with the TQ trellis types) and ik_llama.cpp has no MiniMax-M3 architecture to run them in.

```powershell
strata-mm-quant --model <checkpoint> --recipe tools\minimax\quant\recipes\MM-IQ2_KT.json `
  --imatrix <imatrix.gguf> --out mm-iq2-kt.gguf
# "dense-lite": attention / indexer / dense / shared experts as Q6_K instead of Q8_0, which frees VRAM
strata-mm-quant --model <checkpoint> --recipe tools\minimax\quant\recipes\MM-IQ2_KT.json `
  --imatrix <imatrix.gguf> --dense-type Q6_K --out mm-iq2-kt-dl.gguf
```

The port is checked against ik's own quantizer, not just against itself:

```powershell
cmake -S tools/minimax/quant -B build-quant -DSTRATA_LLAMA_CPP_DIR=<_baseline/llama.cpp>
cmake --build build-quant --target strata-mm-quant-ikq-test
.\build-quant\strata-mm-quant-ikq-test.exe --ref <iqk-dump.exe> --bench
```

`iqk-dump` is a five-line harness that links ik_llama.cpp's own ggml (`_ref/research/iqk/dump.cpp`, built with
`_ref/research/iqk/build.cmd`); without it the test still runs the round-trip and the dot-vs-double checks and
says that the reference was skipped. With it, the quantization is byte-identical and the dequantization is
bit-identical to ik's on every one of the 24 cases (six per format), and the test prints the quantize speed:
47 MB/s for IQ2_KT, 17 for IQ3_KT, 482 for IQ3_K, 315 for IQ4_KS and 138 for IQ2_KL at 8 threads on a 5900X.

Runtime support: the CPU expert path (`dot_rows_base` in `src/minimax/engine.cpp`) reads all five formats, and
so does the CUDA path. `minimax_moe_gu` / `minimax_moe_down` have `Fmt<160>`..`Fmt<164>` in
`src/kernels/cuda/iq_kernels.cu`, so these experts can sit in the VRAM tier (`--minimax-tiers`).
`tools/minimax/test_ikq.py` runs the synthetic model per format on the CPU, against `tiny_ref` on the
dequantized weights, and with every expert in VRAM.

## Real-weight parity (P4, gate G1 on the real checkpoint)

The reference dumps now exist (`Strata-MiniMax/_local/ref_out/dumps/`), so the layer 0 / 3 / 40 comparison runs
on truncated GGUFs built from the real checkpoint with near-lossless types. Build them with the converter:

```powershell
strata-mm-quant --model <checkpoint> --recipe tools\minimax\quant\recipes\MM-IQ3_S.json `
  --override experts=Q8_0 --override output=Q8_0 --layers 0,1,2,3 --out p4-l0123-q8.gguf
strata-mm-quant ... --layers 3  --out p4-l3-q8.gguf     # one MoE layer, remapped to blk.0
strata-mm-quant ... --layers 40 --out p4-l40-q8.gguf    # one deep MoE layer
strata-mm-quant ... --layers 0  --out p4-l0-q8.gguf     # one dense layer
```

`--layers` remaps the source layer onto `blk.0`.. and writes `strata.source_layers`, which the probes use for their
directory names, so a remapped file still calls its dump `dump-L40`.

The engine can dump its own activations and can be fed a layer's input instead of the embedding:

| flag | effect |
|---|---|
| `--minimax-dump-dir DIR` | write `DIR/dump-L<source layer>/<name>.npy` for every layer (or the list below) |
| `--minimax-dump-layers 0,3,40` | restrict the dump to those source layers (`all` is the default) |
| `--minimax-seed-hidden FILE.npy` | row t is token t's residual-stream input instead of the embedding |

Names match the reference: `attn_input` (the post-attention-norm input), `residual_in`, `q_after_rope`,
`k_after_rope`, `index_q_after_rope`, `index_k_after_rope`, `index_block_scores`, `index_block_indices`,
`attn_core_out`, `attn_out`, `router_logits`, `router_weights`, `router_topk_ids`, `shared_expert_out`,
`moe_out`, `expert_<id>_{gate_up,hidden,out,token_idx}`, `dense_*`, `layer_out`.

`tools/minimax/make_seed.py` recovers a layer's input from the reference dumps (`layer_out - attn_out -
(2*routed + shared)`), so a one-layer GGUF can be compared stage by stage. `tools/minimax/compare_parity.py`
prints the table (max abs, relative to the reference's peak, cosine, router-id and block-id agreement).

Measured on the two short/long reference prompts (114 and 4093 tokens), Q8_0 dense and experts, F32 norms and
router, CPU placement (`--minimax-dense-layers-gpu 0`):

| layer / prompt | attn_out rel | MSA block ids | router top-4 | experts (matched routing) | layer_out rel |
|---|---|---|---|---|---|
| 0 / short (from the embedding) | 1.4e-3 | — (1 block) | — | — | 1.3e-3 |
| 0 / long (4093 tokens) | 1.9e-3 | 98.41% sets, 0.9974 Jaccard | — | — | 2.0e-3 |
| 3 / short (seeded) | 2.2e-3 | — (1 block) | 97.4% | 2.1e-3 | 7.1e-4 |
| 3 / long (seeded) | 2.6e-3 | 99.55% sets, 0.9993 Jaccard | 99.1% | 1.8e-3 | 9.6e-4 |
| 40 / short (seeded) | 3.5e-3 | — (1 block) | 96.5% | 4.4e-3 | 2.4e-3 |
| 40 / long (4093 tokens, sparse) | 1.3e-2 | 99.51% sets, 0.9992 Jaccard | 98.8% | 7.7e-3 | 1.2e-3 |

One more run starts at the embedding and goes through layers 0-3 (`--layers 0,1,2,3` in one file): layer 0 matches
as above and layer 3 lands at attn_out 8.6e-2, moe_out 7.5e-3 (rows with the same routing) and layer_out 2.1e-2.
That is accumulation, not a per-layer discrepancy: the Q8_0 embedding alone is 9.8e-3 relative, and three flips
of the router's 4th expert turn into large per-token deviations. The seeded runs above isolate the layer math.

The handful of router and block disagreements are 4th/5th-place ties: the router logits agree to 1.4e-3 relative
and the block scores to 3.7e-3, which is the Q8_0 noise floor, and the expert weights agree to 8.2e-4 by id. The
aggregate numbers (`moe_out`, `attn_core_out` over every row) are dominated by those few tokens and are reported
separately in the tool's output.

## Layer split over two GPUs (P4)

```powershell
build\strata-minimax.exe --model m.gguf --serve --gpus 0,1 --layer-split auto --minimax-kv auto
build\strata-minimax.exe --model m.gguf --serve --gpus 0,1 --layer-split 30 --minimax-kv vram
```

`--layer-split K[,K2]` gives layer K to the next GPU (upstream llama.cpp's `layer` split semantics); `auto`
fills each card with dense + KV bytes until that card's own free VRAM (less `--vram-reserve-mib`) runs out, so
unequal cards work (12+16, 8+12). `--gpus` picks the devices. Each layer's dense weights, its KV, and its VRAM
expert slots live on its owner. There is no P2P between these cards: the hidden state hops through page-locked
host memory (`--layer-split` allocates a pinned staging buffer and every device-bound operand goes through it),
and no `cudaMemcpyPeer` is ever issued. `lm_head` goes on the last card. The startup line reports the split and
each card's VRAM in use: `gpus=0,1 split=30 vram_used_mib=10233,10012 kv_space=vram`.

## KV cache in VRAM or pinned RAM (P4)

`--minimax-kv host|vram|ram|auto`. `host` is the P2 default and is unchanged: fp16 K/V in ordinary memory, CPU
attention, bit-identical to the last release. `vram` keeps K/V (and the indexer keys) on the owning device and
runs MSA there: the indexer block scores are a device kernel, the block choice stays on the host, and the selected
slots are gathered on the device. `ram` keeps the full K/V in page-locked host RAM, keeps the indexer keys in
VRAM, and per decode step copies only the selected blocks (at most `top_k` x block_size = 2048 positions per KV
head) across the bus before the same attention kernel. `auto` (the default with `--layer-split`) tries VRAM and
falls back to RAM, with the reason printed.

The device path needs the indexer head count to equal the KV head count (4 and 4 for M3.1); a file that indexes
only from a later layer keeps the host path and says so.

Measured on the synthetic runnable model (4 layers, 200 tokens, 25 indexer blocks of 8, top-16):

| case | max abs vs the baseline |
|---|---|
| KV in RAM vs KV in VRAM | 0.0 (bit-identical) |
| device attention vs the host path (same weights) | 6.7e-5 (1.5e-5 relative) |
| layers 0,1 on GPU0 and 2,3 on GPU1 vs one GPU, VRAM | 0.0 |
| the same split with the KV in RAM | 0.0 |

`python -m pytest tools/minimax/test_p4.py -q` runs those, plus an `auto` split that checks the reported
boundary; the two-GPU cases skip on a one-card machine.


## Batched prefill (P5a)

The prompt no longer walks the decode path. `--minimax-prefill-batch N` processes `N` prompt tokens per layer pass
(0, the default, keeps the token path and is byte-identical to P4):

```powershell
build\strata-minimax.exe --model m.gguf --serve --gpus 1 --minimax-kv vram `
  --minimax-prefill-batch 4096 --minimax-prefill-experts gpu
```

Per chunk and layer:

- Dense projections are one batched GEMM per weight matrix, not one GEMV per token: the weight tile is
  dequantized once and multiplied against every token with cuBLAS. `--minimax-prefill-dense f32` (the default)
  dequantizes the tile to fp32 (SGEMM) so the result matches the per-token GPU GEMV; `f16` halves the weight
  traffic and the GEMM time at ~10^-3 relative accuracy.
- MSA is batched: the indexer block scores are one device kernel over the whole chunk (one block per query token,
  indexer head and 128-token block), the top-k block choice stays on the host (one row per query, as in decode),
  and `msa_attn_batch` attends each query over its selected blocks straight out of the fp16 KV cache (no per-query
  gather), with the scaled scores, causal clamp and softmax of the token path. KV for the chunk is written to the
  cache in one copy per layer.
- Routed experts are grouped: the chunk's `N x top` route entries are sorted by expert, so each expert's bytes are
  read **once per chunk** instead of once per token. With `--minimax-prefill-experts gpu` a non-resident expert is
  streamed over PCIe into a ring slot on the owning card, then its gate/up and down run as batched products over the
  tokens that route to it (the int8 `iq_mmvq` kernel for IQ2_XS/Q8_0/Q2_0/IQ4_NL/NVFP4, or dequantize-to-fp32 +
  cuBLAS for F32/F16/BF16 and the mixed IQ2_XS-gate/F32-down files). MM-NVFP4's per-expert `blk.N.ffn_*_exps.scale`
  divisors are applied to the gate/up rows before `swigluoai` and to the down output, exactly as the token path.
  `--minimax-prefill-experts cpu` runs the same grouping through the ggml CPU dot products.
- The gate, up and down of the shared expert, the dense MLP layers and `lm_head` use the same batched projection
  helper; `lm_head` is tiled by output rows (4096 at a time) so the 200 K vocabulary never needs a full device buffer.
- `--minimax-prefill-batch` also shrinks itself to what each card's free VRAM can hold after dense weights, KV and
  the reserve; the chosen size is printed (`INFO minimax-prefill batch=... experts=...`). Chunks of 512-8192 are
  the useful range: at 4 K every one of the 128 experts of a layer is routed, so the expert bytes stop growing per
  token and the PCIe stream becomes the floor.

Decode overlap (P5b, on by default where it can be measured): the VRAM-tier experts of a token are launched on a
side stream and the CPU/RAM/SSD experts are computed on the pool while the GPU works; the two are joined before the
combine. `--minimax-expert-prefetch N` additionally applies layer `l+1`'s router to layer `l`'s output (which *is*
`l+1`'s input, so the routes are exact) and copies up to `N` of that layer's missing experts into **free** VRAM
slots on the side stream while layer `l` finishes. It only fills empty slots (never evicts) and it is opt-in,
because a card behind a narrow PCIe link should not be sent expert traffic.

Measured on the synthetic runnable model (4 layers, 200 tokens, 25 indexer blocks of 8, top-16). "token" is the P4
path on the same file and placement:

| case | max abs | median row | KL | top-1 |
|---|---|---|---|---|
| batch vs token, all-CPU placement | 0 (1e-7 float32 floor) | 1e-7 | 3e-9 | 1.000 |
| batch vs token, GPU dense+MSA, CPU experts (F32 experts) | 6.1e-2 | 8.3e-4 | 1.2e-6 | 1.000 |
| batch vs token, GPU grouped experts, IQ2_XS gate/up + F32 down | 5.5e-2 | 1.1e-2 | 9.4e-6 | 0.985 |
| decode with vs without `--minimax-expert-prefetch 2` | 9.5e-7 | 2.4e-7 | 0 | 1.000 |

The max-abs column is dominated by the handful of rows where the router's 4th expert flips - the same effect P4
measured on the real checkpoint (its 4th/5th-place ties). `--minimax-prefill-dense f16` is measurably coarser
(2.4e-2 relative on this file) and is not the default.

`python -m pytest tools/minimax/test_p5a.py -q` runs the table, plus an end-to-end serve check that the greedy
continuation after a 40-token prompt is identical with and without `--minimax-prefill-batch`.

### Real weights: a 15-layer slice of MiniMax-M3.1-Flash

No complete local IQ2_XS pack existed (the D3 upload has 3 of 13 shards, and the full 133 GB does not fit next to
the 111 GB free), so the measurement uses a truncated GGUF built from the local checkpoint with
`strata-mm-quant --layers 0..14` (layers 0-2 dense + 12 MoE layers, IQ2_XS experts + a Q8_0 dense path, 30.7 GB).
It is a real-weight slice, not the full model: the per-token expert bytes are 12/57 of the real set, so decode
scales down by ~4.7x on the full model and prefill's PCIe/DFS term by ~4.75x.

**The machine was loaded during these runs** (another task held 99 GB of RAM, GPU0 was at ~30 % and both cards had
7-9 GB in use), so page cache was scarce and every expert read came from the NVMe. Numbers, RTX 5070 (GPU1,
Gen2 x16 idle -> Gen4 x16), 16 CPU threads, `--max-context 8192`, random 200 K-vocabulary ids:

| prompt | engine `--minimax-prefill-batch 4096` | engine token path | llama.cpp `-ot exps=CPU` |
| ---: | ---: | ---: | ---: |
| 512 | 62.4 tok/s (TTFT 8.2 s) | 3.8 tok/s (TTFT 133 s) | 208.8 tok/s (pp512) |
| 2048 | 115.5 tok/s (TTFT 17.7 s) | — (extrapolates to ~9) | — |
| 4096 | 117.8 tok/s (TTFT 34.8 s) | — | — |
| decode, 32 new | 4.2-4.3 tok/s | 4.9 tok/s | 23.1 tok/s (tg32) |

What the numbers say, honestly:

- Batching is worth **16x at 512 prompt tokens** on the same file. The gap grows with prompt length (the token path
  pays the full 786 MB of experts per token; the batched path pays it once per chunk), but this slice is small
  enough that llama.cpp's MMQ CPU path with a warm page cache still wins prefill on a *loaded* box.
- Our CPU expert kernels are the bottleneck for decode: the engine uses ggml's per-row `vec_dot` (P2), llama.cpp
  uses its int8 MMQ/multi-row kernels, and that is a ~5x difference here. Porting `src/kernels/cpu/iq_avx2.cpp`'s
  multi-row IQ2_XS kernel into the P5b pool is the single biggest remaining decode item (SPEED_RESEARCH item 10).
- `STRATA_MM_TIME=1` prints a per-layer `attn`/`ffn` breakdown on stderr. On the slice at 2048 tokens the batched
  MSA kernel is ~0.38 s/layer (about 60 % of the layer) against ~30 ms of cuBLAS projections and 150 ms of
  host-side norm/rope/fp16 work; a KV-head-tiled flash-style kernel is the next prefill item.

## Not in this phase (P6)

CUDA graphs for the per-layer decode work (the engine still round-trips the host for the MSA top-k and the combine,
so a graph would have to be split around those), a measured CPU/GPU cost model per chunk (the switch is
`--minimax-prefill-experts`, not automatic yet), the AVX2 multi-row IQ2_XS CPU kernel, and tensor cores in the MSA
kernel (the P5b kernel is still scalar half math).

## P5b: decode and prefill on the same file as llama.cpp

Three changes, each measured on its own on the 15-layer real-weight IQ2_XS slice
(`p5a-l0-14-iq2xs.gguf`, 12 MoE + 3 dense layers, 30.7 GB), one RTX 5070, 24 CPU threads, all experts on the CPU,
box otherwise idle (103 GiB RAM free, both cards under 1.2 GiB used):

| change | 512 prefill | 4096 prefill | decode |
| --- | ---: | ---: | ---: |
| P5a (start of this branch) | 62.4 tok/s | 117.8 tok/s | 4.3 tok/s |
| + dense GEMV through packed CUDA MMVQ | 62.4 | 117.8 | **13.2** (241 → 76 ms/token) |
| + multi-slot expert ring (overlapped reads) | **97.1** | **121.8** | 13.2 |
| + flash-style MSA kernel | **111.2** | **194.3** | 15.2 |

Final A/B, interleaved (engine run, then llama-bench, three rounds, medians), same file and context, load printed
before every run: 4096-token prompt, 32 new tokens, 24 CPU threads for the engine and `-t 16` for llama.cpp
(`llama-bench -p 4096 -n 32 -r 1 -t 16 -ngl 99 -fa on -ot exps=CPU -dev CUDA1`), 100 GiB RAM free, both cards
under 1.2 GiB used, GPU utilisation 0-3 %:

| | prefill tok/s (3 rounds) | decode tok/s (3 rounds) |
| --- | ---: | ---: |
| strata-minimax | **168.0** (167.9, 168.0, 168.5) | 12.56 (12.59, 12.56, 12.56) |
| llama.cpp | 119.5 (118.9, 120.1, 119.5) | 15.81 (17.27, 15.81, 12.50) |

Prefill is 1.4x llama.cpp's and stable to 0.4 %; decode is 0.79x with a 1.4x spread between llama.cpp's own rounds.
The single-engine numbers above (194.3 / 15.2) are what the engine does when it is not sharing the page cache with a
second engine's mapping. Every number is the engine's own `PP`/`DONE` line (`tools/minimax/bench_p5b.py`).

### Dense projections: `--minimax-dense-mmvq` (default on)

The token path dequantized each dense weight tile to fp32 in a scratch buffer and ran a fp32 GEMV on it: a Q8_0
attention projection moved 8x its own bytes, and the Q6_K vocabulary head did that 49 times per token. It now calls
`native_mmvq` (llama.cpp's CUDA MMVQ, already in the tree) on the packed blocks with a q8_1 activation. Attention
went from 4.4 to 1.0 ms/layer and decode from 241 to 76 ms/token. It is also *closer* to the all-CPU logits than the
path it replaced (KL 7.8e-3 vs 9.1e-2 on the real slice), because that path was handing Q8_0/Q6_K weights to the
i-quant dequantizer, which only knows the i-quant formats. `--minimax-dense-mmvq off` keeps the old path.

### Expert ring: four staging slots, overlapped

The batched MoE stream ran read → copy → compute per expert through one slot, and the "parallel" read was one
thread: 36 pieces is under `pool.run`'s four-pieces-per-thread inline threshold. The chunk's 2.1 GB/layer of expert
bytes therefore moved at ~2.4 GB/s. The ring is now four slots with their own pinned staging buffer, an H2D event on
a copy stream and a compute-done event on the main stream; the read is spread over the whole pool and the copy of
expert *i+1* overlaps the GEMMs of expert *i*. Per-layer read at 4096 tokens: 690 ms → 490 ms; prefill 62 → 97 tok/s
at 512.

### MSA: `msa_attn_flash_kernel`

The P5a kernel kept `4 x (top_k x block) = 43 KB` of per-position scores in shared memory (two blocks per SM) and
streamed the selected positions three times. The new kernel is one block per (query token, KV head) covering every
query head of that KV head (16 for M3.1), Q in shared memory, positions in chunks of 128 with an online softmax:
21 KB of shared memory, one pass. Its output is **bit-identical** to the old kernel (max-abs 0.0 on 8 real-weight
rows) and 3.7x faster: MSA 16.4 → 4.4 s over a 4096-token prompt (73 % → 21 % of prefill).

### What is still slow

- Host-side norm/rope/fp16 work is now 721 ms of a 804 ms attention block at 4096 tokens (~22 % of prefill): it
  still round-trips q/k/v to the host. Moving `rms_heads` + `rope` to the device is the next prefill item.
- Decode experts: 4.1 ms/layer for 65 MB = 16 GB/s from DDR4, against ~41 GB/s peak. The kernel is ggml's single-token
  `vec_dot`; llama.cpp is at ~23 GB/s because its weights are committed RAM rather than our mmap, which pays a soft
  page fault per 4 KiB. `--minimax-tiers --minimax-ram-experts-per-layer 128` does not help by itself (the arena is
  filled on demand, i.e. re-reads the same bytes once more).
- `--minimax-prefill-dense f16` **is fixed** (P5c). It used to produce non-finite logits for any prompt that needed
  more than one prefill chunk. The fault was not the HGEMM: `p.gh`, the fp16 swiglu buffer the f16 path feeds to the
  down GEMM, was sized `rows_cap x n_ff_exp` (1024 x 3072) while the *dense* MLP's hidden is `n_ff_dense` = 12288
  wide - a 4x write past the end of the buffer that landed in whatever was allocated after it. The f32 path never
  touches `p.gh` in the dense MLP (it swiglu's into `p.sh`), which is why only f16 broke. It is now sized by
  `max_dense_n`. On the real slice, 4096-token prefill: f32 270.6 tok/s, f16 294.8/296.4 tok/s (+9 %), and the
  f16 logits against f32 over 768 tokens (two chunks) are finite at KL 1.7e-3, top-1 0.977, median |d| 0.068. It
  stays opt-in: the coarser activation is measurable, and the default is the parity-first f32 path.

## P5c: the norm+rope round trip is gone

`--minimax-gpu-normrope` (default on) runs per-head RMSNorm + partial NEOX RoPE + the fp16 cache write on the
device for both paths. P5b measured 721 ms of a 804 ms attention block at 4096 tokens as host work on q/k/v: the
projections were copied back, normalized and rotated on one core, then converted to fp16 and copied in again.
Now `cu::head_norm_rope` (one block per head row) reads the projection output in place and writes fp16 straight
into the KV cache; the MSA query buffer and the indexer keys are produced by the same kernel, and the token path
reads its query from the device buffer instead of uploading it.

It is **bit-identical** to the host loop, which took two details. The RMS sum is sequential on one thread with
`__fmul_rn`/`__fadd_rn` (nvcc contracts `a*b+c` into an FMA by default, and that last-bit difference was enough to
flip an fp16 K/V rounding and - measured - the router's 4th expert at token 20). The cos/sin table is built from
the same double-precision expression `rope()` evaluates per call (`pos / theta^(2i/n_rot)`), not from
`build_rope_table`'s `pos * theta^(-2i/n_rot)` form. `test_p5c.py` holds both paths to `array_equal` on 200 tokens
of the runnable model.

The table is built once per device up to `min(ctx, 128 K)` positions (256 B per position); a longer context falls
back to the host loop rather than reserving 256 MiB of VRAM on every card.

Measured on the 15-layer real-weight IQ2_XS slice, one RTX 5070, KV in VRAM, 24 threads, box idle:

| prompt | P5b | P5c | gain |
| ---: | ---: | ---: | ---: |
| 512 prefill | 82.2 tok/s | 90.8 | +10.5 % |
| 4096 prefill | 168.2 tok/s | 222.6 | +32.3 % |
| decode (32 new) | 12.63 tok/s | 13.03 | +3.2 % |

`--minimax-gpu-normrope off` is the P5b path, byte for byte.

### Resident experts: `--minimax-expert-resident off|auto|all`

The P3 RAM tier was filled on demand: the first token that routed to an expert copied it out of the mmap into the
pinned arena, so that token paid the fault *and* the copy, and the arena was a cold, partly-empty cache for the
first tokens of every run. `auto` (the default, and only when a RAM tier exists) and `all` (which creates the tier
even with the tiers off, using the device only for dense/KV) now fill every static slot at startup with a parallel
reader (1 MiB pieces over the whole pool), print the achieved rate and the arena backing, and the batched prefill
reads a resident expert from the arena instead of re-faulting it once per chunk.

Large pages are tried first and refused on this box (`VirtualAlloc` error 1314: `SeLockMemoryPrivilege` is not
assigned to a normal account), so the arena is 4 KB pages, `cudaHostRegister`ed and working-set locked. That is
the documented fallback, not an error.

15-layer IQ2_XS slice, RTX 5070, kv=vram, 24 threads, 4096-token prompt, 32 new tokens, page cache **cold**
(45 GiB of anonymous memory written first to evict the mapping):

| arm | prefill tok/s | decode tok/s | ms/token |
| --- | ---: | ---: | ---: |
| `off` (mmap, tier filled on demand) | 131.2 | 6.39 | 156.5 |
| `all` (pinned arena, eager fill) | 104.4 | **12.07** | 82.9 |

With the page cache **warm** (the model's 28.6 GB fit beside the arena) the decode difference disappears:
16.9-17.1 tok/s both arms, and the fill buys prefill at 512 tokens instead (107-126 -> 138-151 tok/s, medians of
three interleaved rounds). The honest reading: pinning does not make DDR4 faster - it stops the *page faults and
SSD reads* when the system cannot keep the whole model cached. On a 128 GB box with a 146 GB model that is the
normal case, not the exception.

Effective CPU expert bandwidth (785 MB of routed experts per token at 512 context, from `STRATA_MM_TIME`): the
expert FFN is 39.2 ms/token = **~20 GB/s**, against 16 GB/s in P5b and 39.5 GB/s for a raw 24-thread stream of the
same RAM. The remaining gap is the single-token IQ2_XS dot kernel, not the memory system: `iq_avx2_parity --bench`
puts Strata's own AVX2 IQ row kernel within 5 % of ggml's `vec_dot` at one token (its win is multi-token), so there
is no cheap port left here.

### Expert skipping: `--minimax-expert-skip <threshold>` (default off)

A routed expert whose renormalized weight is below the threshold is dropped and the survivors are renormalized; a
route whose every weight is below it keeps everything rather than produce a zero vector. The skip applies to the
token path, the tiered path (so a skipped expert is never fetched from VRAM/RAM/SSD) and the batched prefill.

Speed, 512-token prompt, 32 new tokens on the slice (decode):

| threshold | decode tok/s | gain | skipped routes |
| ---: | ---: | ---: | ---: |
| 0 (off) | 18.04 | - | 0 |
| 0.05 | 18.17 | +0.7 % | - |
| 0.10 | 18.62 | +3.2 % | - |
| 0.15 | 19.11 | +5.9 % | - |
| 0.20 | 21.14 | +17.2 % | - |

Quality against the engine's own no-skip logits, the short G3 reference prompts (97-114 tokens), same file and
placement (`tools/minimax/skip_quality.py`):

| threshold | mean KL | mean top-1 | worst median abs |
| ---: | ---: | ---: | ---: |
| 0.05 | 9.3e-4 | 0.958 | 1.7e-1 |
| 0.10 | 1.7e-3 | 0.955 | 2.8e-1 |
| 0.15 | 1.2e-2 | 0.894 | 7.1e-1 |
| 0.20 | 4.9e-2 | 0.769 | 1.6 |

0.10 is the only setting whose KL stays under the IQ2_XS quant noise floor this project measured on the real
checkpoint (0.012), for +3 % decode; 0.15 is at the floor with a visible top-1 cost. The default stays **off**: a
3 % win does not justify changing answers, and the slice's 15 layers understate what a deep-layer skip costs.

### CUDA graphs: `--minimax-graphs off` (default on)

The token path's attention prefix is nine small launches per layer per token: the q8_1 quantize of the activation,
five dense MMVQs (q, k, v, indexer q, indexer k), two fused norm+rope calls and one fp16 cast. The first token of
each layer now captures that prefix into a CUDA graph (`cudaStreamBeginCapture` on the dedicated blocking stream,
instantiated once per layer) and every later token replays it with one launch; the host-side inputs (the activation
h2d and the 4-byte position) stay outside the graph as addresses baked in at capture. The K/V rows land in a fixed
fp16 staging buffer inside the graph, and two 2 KB D2D copies place them in the cache at the token's offset - the
one pointer that changes per token.

The graphs are **bit-identical** to the direct launches (`test_p5c.py`, 200 tokens) - same kernels, same
arguments. 512-token prompt, 96 new tokens, 3 interleaved rounds, same box:

| arm | ms/token | decode tok/s |
| --- | ---: | ---: |
| `--minimax-graphs off` | 57.71 (57.16, 57.71, 57.77) | 17.33 |
| graphs on (default) | **54.56** (54.33, 54.56, 56.64) | **18.33** |

+5.8 %. It buys launch and stream-ordering overhead only: the expert reads, which dominate the token, are on the
CPU and untouched.

## P5c: the full checkpoint on both GPUs (`bench_p5c.py`)

Everything above is the 15-layer IQ2_XS slice. This is the whole `MiniMax-M3.1-Flash-IQ2_XS-D3` pack on the
127.9 GiB box: 60 layers, 57 MoE layers x 128 experts top-4, 136 GiB of tensors (123.2 GiB of routed experts,
10.6 GiB dense, 2.1 GiB embedding and output), two RTX 5070s (12 GiB each, ~10.9 GiB free after the displays).

**The checkpoint does not fit in RAM.** 123.2 GiB of experts against 128 GiB of DIMMs means a large part of the
expert set is read from the NVMe on demand, and that is what the numbers below are measuring. Read them as a
cache study, not as a compute benchmark.

### Harness

`tools/minimax/bench_p5c.py` is the full-model counterpart of `bench_p5b.py`:

- one engine process per context, several `GEN` requests inside it, so the 84 GiB pinned arena is filled once and
  repeat requests measure the deployed (warm) state instead of paying the fill again;
- every repeat moves the last prompt token, so the engine's prefix cache cannot turn a repeat into a hit
  (`DONE ... reused=0` is asserted by the row); `llama-server` gets `cache_prompt: false` for the same reason;
- peak VRAM per GPU (sampled through `nvidia-smi` while the run is in flight), peak process working set and the
  minimum free system RAM are recorded per request;
- `--llama-server` drives llama.cpp over its HTTP API so both engines see the same *text* (see below);
- `--ids FILE` takes comma-separated ids or the G3 dump format (int32 `.bin`).

`tools/minimax/bench_prompt.py` builds matched prompts: the same token ids for the engine and the decoded text for
llama.cpp, from a prompts.jsonl plus the official `tokenizer.json`, and it checks that
`encode(decode(ids)) == ids` so the two engines really see one sequence. Long contexts repeat the corpus *text*
(with a paragraph break) rather than the id list: a repeated id list decodes into text whose re-encoding merges
differently at every join (32 K measured 5619/32768 ids), while a repeated text round-trips exactly.

### The measured arm (uniform-random ids)

Synthetic ids from the `bench_p5b.py` LCG are what `llama-bench` feeds itself, so this arm is apples-to-apples -
but uniform ids route almost uniformly over the 128 experts, which is the worst case for any residency scheme and
makes the routing profile useless. Real-text prompts are the next arm (`bench_prompt.py`), not measured yet.

```
strata-minimax.exe --model MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf --serve --max-context <n+33> \
  --gpus 0,1 --layer-split auto --minimax-kv vram --minimax-dense-layers-gpu -1 \
  --minimax-prefill-batch 4096 --minimax-prefill-experts gpu \
  --minimax-profile mm3-d3.bin --minimax-expert-resident all \
  --minimax-vram-slots-per-layer 0 --memory-limit-mib 86000 --cpu-threads 24 --minimax-stats
```

`--layer-split auto` puts layers 0-27 on GPU0 and 28-59 on GPU1 (dense for all 60 layers on the device, f16 KV in
VRAM); the engine reports `dense_gpu_layers=60 split=28 kv_space=vram`. Decode is 32 new tokens, temperature 0.
Request 1 is the first request after the load, request 2 the next one in the same process.

| context | request | prefill tok/s | TTFT s | decode tok/s | ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 17.7 | 28.85 | 2.40 | 415.9 |
| 512 | 2 | 16.5 | 31.12 | 2.90 | 344.6 |
| 4096 | 1 | 37.4 | 109.64 | 2.40 | 416.3 |
| 4096 | 2 | 48.8 | 83.95 | 2.43 | 412.2 |

Tier counters, from `--minimax-stats` (228 routes per token = 57 MoE layers x top-4):

| context | from pinned RAM | from SSD | from VRAM | SSD bytes/token |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 180-222 | 6-48 | 0 | 108-870 MiB |
| 4096 | 188-218 | 10-40 | 0 | 217-707 MiB |

So 79-97 % of the expert weight comes out of the arena and the rest off the NVMe, at 0.9-2 GB/s effective
(`ssd_bytes` / `ms_per_token`). Decode is 3.9 GiB of expert weight per token (17.3 MiB x 228); at 400 ms that is
~10 GB/s against the ~30-40 GB/s the 5900X's dual-channel DDR4-3200 can stream, so there is room on the token path
too - see the thread hypothesis below.

Load and memory: the 512-token process peaked at 101.9 GiB working set (arena 84 GiB + the mapped pages it
touched) with 2.7 GiB free of 128 GiB; the 4096-token one started from 106.1 GiB free. Peak VRAM per card was not
carried into the surviving rows of this arm (the sampler writes only in the final table/JSON, and the run was
stopped before that); the layout is dense 10.6 GiB split over the two cards plus f16 KV (0.07 GiB at 512,
0.55 GiB at 4096), i.e. ~5.4 GiB per card. GPU utilisation sampled during decode was 10 % on GPU0 and 2 % on GPU1
- the token is waiting on the CPU and the SSD, not on the cards.

### Two things this arm hit, both harness/auto-sizing, both fixed or worked around

- **The auto VRAM expert slots starve the prefill workspace.** With the default `--minimax-vram-slots-per-layer`
  (`-1`), the engine placed 6 slots per layer and reported `vram_used_mib=10287,12198` against 12227 MiB cards -
  18 MiB free on device 1 - and then `batched prefill needs workspace on device 1 but only 18 MiB is free`,
  silently falling back to per-token prefill (a 512-token prompt then prefills at 3-8 tok/s instead of ~17). The
  measured arm therefore runs `--minimax-vram-slots-per-layer 0` and leaves VRAM to dense, KV and workspace. The
  auto slot count should reserve the workspace before it takes slots; that is a P6 item.
- **The harness must close the engine's stdin after the request queue.** `--serve` waits for more input while the
  pipe is open, so a harness that reads `DONE` lines before closing stdin deadlocks (measured: two 20-minute runs
  lost to it before the fix). `bench_p5c.py` writes the whole queue, closes stdin, then reads.

### llama.cpp side (prepared, not run)

The box went to the G3 quality run and then to the decode work, so the llama.cpp column is empty in the table
above. The command is one line and the harness already speaks it:

```
llama-server.exe -m MiniMax-M3.1-Flash-IQ2_XS-D3-00001-of-00013.gguf -c 32801 -ngl 99 -cmoe -fa on -sm layer \
  -t 24 --parallel 1 --host 127.0.0.1 --port 8099
python tools/minimax/bench_p5c.py --model <gguf> --lengths 512,4096,32768 --skip-engine \
  --llama-server <bin>/llama-server.exe --prompts-dir bench-prompts --llama-args '-cmoe'
```

`-cmoe` keeps every MoE expert on the CPU (the counterpart of the engine's RAM tier), `-ngl 99` puts all dense
layers and the KV on the cards with `-sm layer` across both, `-fa on` is flash attention. `-ot exps=CPU` is the
equivalent spelling; `-cmoe` is the documented flag in this build (11837). Each request is
`{"prompt": text, "n_predict": 32, "temperature": 0, "cache_prompt": false}` and the numbers come from the
server's own `timings` (`prompt_per_second`, `predicted_per_second`). This path is untested: `llama-server` never
loaded the pack on this box.

### Thread hypothesis, for the next decode pass

The engine caps its pool at 24 threads and this box is a Ryzen 9 5900X: 12 cores, 24 threads. A latency-bound
expert matvec usually wants one thread per *core*, so `--cpu-threads 12` and `16` are worth two runs against 24;
the slice measurements above all used 24. Nothing has been measured for that yet - it is the first thing to try
when the decode path lands.

### Not measured

32 K context (one 4096-token chunk per ~85-110 s at 4096 extrapolates to ~11-15 min of prefill for 32 K, plus the
decode), 128 K and 256 K (KV fits VRAM, the expert set does not), and the in-distribution arm on real text. All
three are the next run of the same harness.






