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
- `--minimax-prefill-dense f16` is 7 % faster (209.7 tok/s at 4096) but still produces non-finite logits above 512
  prompt tokens. It is not the default; the fp32 path is now correct after the dequantizer fix.


