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
`cancel`, or `nonfinite`. `STOP` during a request finishes it with `cancel`. `BGEN` returns `ERR` and no
`DONE`. `GENI` is the same as `GEN` plus one `img=` key naming the pictures in prompt order (comma separated,
no spaces): `GENI 64 temperature=0 img=C:\pics\a.png,C:\pics\b.png 1,2,...`. Unknown options are ignored in
`--serve` mode, because the server's config is shared with the Qwen families.

Sampling keys on `GEN`: `temperature` (0 is greedy argmax, ties take the lowest id), `top_k`, `top_p`,
`min_p`, `penalty_last_n`, `penalty_repeat`, `penalty_freq`, `penalty_present`, `seed`.

`--minimax-allow-nonstandard` serves a reduced test GGUF; `--serve` otherwise refuses it.

## Vision (EXPERIMENTAL)

MiniMax-M3.1-Flash's vision tower (32-layer ViT, width 1280; 1280->6144 projector; 2x2 patch merge) is in a
separate `mmproj` GGUF that setup downloads next to the quants:

```
mmproj-MiniMax-M3.1-Flash-F16.gguf
```

It is the llama.cpp `clip` layout (`clip.projector_type = minimax_m3`), so llama.cpp's mtmd reads it too. Build it
from the checkpoint with the repo's converter:

```powershell
python tools/minimax/convert_mmproj.py <checkpoint-dir> -o mmproj-MiniMax-M3.1-Flash-F16.gguf
```

The engine loads the file with `--mmproj`. A `GENI` request names one or more pictures; the engine decodes
(JPEG/PNG/BMP/GIF via stb_image), resizes the short side into 28-pixel blocks with at most 672x672 pixels
(`smart_resize`, as the checkpoint's `image_processor.py`), normalizes with the checkpoint's mean/std, runs the ViT
and projectors on the CPU in fp32, and replaces the prompt's `]<]image[>[` tokens with the projected rows. One
672x672 picture is 576 tokens; 640x480 is 391.

```powershell
build\strata-minimax.exe --model first-part.gguf --mmproj mmproj-MiniMax-M3.1-Flash-F16.gguf --serve
# GENI 64 temperature=0 img=C:\pics\red_circle.png 1,2,...  (the server expands each picture's 391/576 tokens)
```

The server side (`serve/` on `main`) accepts OpenAI `image_url` (data: URL or http) and Anthropic `image` blocks
(base64) and passes the files to the engine; it needs no `strata-vision` process for MiniMax.

The ViT is CPU-only for now: about 64-80 s per 640x480 picture on 16 cores, 0.9-1.7 GB of RAM per encoder (3.5 GB
of fp32 weights). A GPU path and video (the temporal axis) are not implemented. Parity against transformers
5.20.2 fp32 on CPU is per-token cosine >= 0.99999 for the same `pixel_values`; see
`tools/minimax/check_vision_parity.py` and `tools/minimax/ref_vision.py`.

Only the vision tower and projectors are redistributed (about 0.87 B parameters, 1.73 GB); the MiniMax Community
License applies, the same terms as the quants.

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
  `--minimax-prefill-experts cpu` runs the same grouping through the ggml CPU dot products. The default
  `auto` uses the device when it can and keeps a model of the ported ik_llama formats (IQ2_KL, IQ2_KT, IQ3_K,
  IQ4_KS, IQ3_KT) on that CPU path, because the device dequant kernel has no type for them; an explicit `gpu`
  still asks for the device and fails with `iq_mmvq: type ... is not supported`.
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

### Tensor-core routed experts (P11)

`--minimax-prefill-expert-gemm` chooses how a batched chunk's routed experts are projected:

- `mmvq` (default) keeps the int8 `iq_mmvq` dots: the expert's weights are decoded and dotted against a q8_1
  activation, one group of tokens at a time.
- `tc` dequantizes each routed expert's gate/up/down weight matrix to FP16 **once per chunk** and runs every token
  routed to that expert through one cuBLAS HGEMM per projection (FP16 tensor cores, FP32 accumulation). The weight
  decode is the same amount of work the int8 kernel already does; it is now amortized over the group's rows instead
  of feeding a dot. The dequantized matrix is a tile of the existing 48 MiB weight scratch, so no VRAM is added.
- `auto` takes `tc` for a group of at least 16 rows and keeps `mmvq` below it. A 200-token prompt routes about six
  tokens per expert; dequantizing 18.9 M weights for six rows is not worth it, a 4 K chunk routes ~128.

The default is `mmvq`, so an existing command keeps the int8 math. On the 15-layer real-weight slice (4 K real
tokens, IQ2_XS gate/up + Q2_0 down) `tc` against `mmvq` measured **KL 1.34e-4 nats, top-1 32/32**, max abs 0.30 on a
logit std of ~1.7 (the router's 4th-expert flips, the same effect P5a measured), and the same run was 28.2 s against
36.3 s. `tools/minimax/test_p10_expert_gemm.py` checks the same shape on a synthetic IQ2_XS model plus the `auto`
selection (2 passed).

**The full model's G3 gate failed, so `mmvq` stays the default.** On the whole IQ2_XS-D3 file (whose experts are
IQ2_XS gate/up + **IQ3_XXS** down) `tc` against `mmvq`, same six short prompts, 679 positions, 512-token chunks,
`--minimax-prefill-experts gpu`: KL64 1.2346 -> 1.2998, full-vocab KL 1.4409 -> 1.4886, top-1 60.82 % -> 58.17 %,
and `code_lru` alone 38.6 % -> 30.7 % top-1. That is far past fp16 rounding, and the slice gate above passed with a
Q2_0 down, so the next thing to check is the iq3_xxs down projection on this path (`tc_d_ok` admits any type
`tc_expert_type` accepts, not only the 256-block types the gate covered).


The unbuffered read path was also tightened: an expert that the pinned RAM tier already holds is copied from the
arena instead of being re-read from the file, and a real miss is read straight into the three role windows of the
pinned ring slot with the three requests in flight (no 19 MiB staging copy). The bytes are identical: on the slice,
`--minimax-prefill-read direct` against `mmap` was `np.array_equal` (max abs 0.0). `STRATA_MM_TIME=1`'s
`batch-ring` line now prints the direct-read milliseconds, the MiB and the drive's GB/s, plus the per-layer group
counts (`vram= ram= disk= tc=`), so a drive-speed swing is visible next to the layer time.

Measured on the full IQ2_XS-D3 model, 2x RTX 5070, `--minimax-tiers` + `mm3-d3.bin`,
`--minimax-kv ram --minimax-vram-slots-per-layer 0 --memory-limit-mib 86000 --cpu-threads 12`, real text, one
batched chunk per prompt (32 K splits into 22 991 + 9 777). Drive rate 4.4-5.6 GB/s on every run below.

| prompt | `mmvq` prefill | `tc`/`auto` prefill | speedup |
|---|---|---|---|
| 200 | 7.85 tok/s (25.5 s) | same (`auto` keeps mmvq) | - |
| 512 | 15.85 tok/s (32.3 s) | 16.26 tok/s (31.5 s) | 1.03x |
| 4 K | 42.62 tok/s (96.1 s) | **56.67 tok/s (72.3 s)** | **1.33x** |
| 32 K | not re-run | 44.3 tok/s (739.6 s) | vs 37.8 tok/s in P9 |

Below 1 K tokens the chunk is the expert byte wall: all 128 experts of all 57 MoE layers are routed (139 GB, read
once per chunk) and the GEMM is not the cost. At 4 K the tensor-core path takes the routed MoE from ~700 ms to
~470 ms per layer while the read (320-540 ms/layer) and the MSA kernel (274 ms/layer) stay. At 32 K the chunk is
read-bound and attention-bound: the last chunk's MoE is 0.72 s/layer against 4.25 s/layer of attention
(`msa=112 s`, `idx=70 s`, `topk=6.7 s` over its 60 layers), so 32 K needs the attention kernel, not a faster expert
GEMM.

A 200-token chat prompt is 3.5 tok/s on the token path (`--minimax-prefill-batch 0`) and 7.85 tok/s batched: the
token path re-reads each expert once per token that routes to it, so the same prompt reads the expert set about six
times. `sparsa.py serve` has passed `--minimax-prefill-batch 4096` since 0.1.1 and it stays on by default
(`--prefill-batch auto`), so chat reads its prompt the fast way; a 411-token message measures 30.5 s against 124 s on
the token path.


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

## P6: the SSD tier read path (a model larger than RAM)

Everything above assumes the experts fit in the pinned RAM tier. MiniMax-M3.1-Flash at IQ2_XS-D3 does not: the
engine's own counters on the full pack (`--minimax-stats`, 31 decode tokens) average **26 VRAM / 139 pinned RAM /
63 SSD** routed experts per token, so **1,094 MB of every token comes off the NVMe** — 28% of the routed-expert
bytes and, at the rate the old path got out of the drive, most of the token.

The token path now treats the SSD like the other two resources: it is filled ahead of use and overlapped.

- `ssd_slots`: one 4 KiB-aligned staging slot per routed entry, holding the three role buffers with 4 KiB of
  headroom in front. A role's read is the aligned superset of its range and the dot kernels read the bytes at
  `base + (off & 4095)`, so the data lands **where it is used** and the 16.4 MB copy out of a shared read buffer
  (per expert, per token) is gone.
- Every read of a layer is queued up front (`DirectFile::submit` only enqueues) and waited on **once**, in batches
  of completions, after the GPU's experts have been launched on the side stream and the pinned-RAM experts have
  been computed. The drive therefore runs while the CPU and the GPU work, instead of strictly before them.
- `--minimax-ssd-chunks N` (default **1**): pieces one role's read is split into. On this box's NVMe one 5.45 MiB
  read is **2.4x faster** than eight 700 KiB pieces of the same range (46 ms against 110 ms of waiting per token
  on 261 MB of misses), so the default is 1. A drive that prefers many small requests wants 4-8 — Project Maya
  measures +15% at 8 on a Windows laptop's Intel RST RAID, and its own docs say a Samsung PM981a *loses* 15% at 8.
  Measure yours with `bench_p6_ssd.py`.
- `--minimax-expert-skip-ssd RANK` (default **0 = off**): a route at or past `RANK` (0 = best) whose expert is in
  neither the VRAM nor the pinned-RAM tier is left out, and its bytes are never read. This is Strata-DS's
  `SSD_TAIL_SKIP` (`docs/DEEPSEEK_LAPTOP.md`: 37.8 -> 34.9 ms/token alone, 44.8 -> 28.7 with the route tail, at
  KL 0.030-0.032 against a 0.012 noise floor). MiniMax's route is 4 deep, so `3` keeps the top three and drops the
  fourth when it is an SSD miss; `2` keeps two.
- `--minimax-stats` now also prints `ssd_wait_ms` (the token's time spent waiting for the drive) and `skipped`.

Measured on the 15-layer real-weight IQ2_XS slice, SSD-heavy placement (`--minimax-vram-slots-per-layer 0
--minimax-ram-experts-per-layer 4`, so 16.7 of 48 routes/token are misses = 261 MB/token), one RTX 5070, 12 CPU
threads, 512-token batched prefill then 32 greedy tokens, medians of 3 reps, interleaved rounds, box idle:

| arm | decode tok/s | ms/token | SSD MB/token | time waiting for the drive |
| --- | ---: | ---: | ---: | ---: |
| P5c (before) | 6.90 (6.86 / 6.61 / 6.46) | 144.4 | 261 | ~90 ms, interleaved and unmeasured |
| P6, `--minimax-ssd-chunks 1` (default now) | **8.66** (8.68 / 8.64) | 115.4 | 261 | **46.1 ms** |
| P6, `--minimax-ssd-chunks 8` | 5.69 (5.68 / 5.69 / 5.69) | 175.9 | 261 | 110.2 ms |

**+25.5%**. It is bit-identical: 40 of `python -m pytest tools/minimax -q` pass (the all-SSD tier vs the all-CPU
path is one of them), and teacher-forced logits on the real-weight slice with this placement are `array_equal`
between the two engines (max abs 0.0, KL 0.0, top-1 1.000).

**Skipping the SSD tail.** Same file and placement, 8 greedy tokens per arm, `--minimax-expert-skip-ssd`:

| arm | SSD experts/token | SSD MB/token | decode tok/s | SKIP |
| --- | ---: | ---: | ---: | ---: |
| 0 (off) | 30.2 | 472 | 4.51 | 0.0 |
| 3 (keep top three) | 21.9 (-28%) | 341 (-28%) | **5.46 (+21%)** | 8.4 |
| 2 (keep top two) | 14.9 (-51%) | 232 | not measured (the box was loading another model) | 17.2 |

Quality against this engine's own no-skip logits, 64 teacher-forced tokens on the slice, 8 dump rows: rank 3 gives
mean KL **0.0237**, top-1 0.75; rank 2 gives KL **0.0430**, top-1 0.875 (8 rows is a coarse top-1). The project's
IQ2_XS quant noise floor on the real checkpoint is 0.012 and Strata-DS ships its tail skip at 0.030-0.032, so rank
3 is in the same band — but this is a 15-layer slice and the default stays **off**: on the full model the skip
fires 63 times a token, and that is a decision for the operator, not for a default.

`python tools/minimax/bench_p6_ssd.py --exe A/strata-minimax.exe --exe B/strata-minimax.exe --model m.gguf \
  --prompt 512 --decode 32 --reps 3 --rounds 3 -- --gpus 1 --minimax-tiers --minimax-vram-slots-per-layer 0 \
  --minimax-ram-experts-per-layer 4 --minimax-kv vram --minimax-stats --minimax-prefill-batch 512` runs the A/B
above (`--arm-args " --minimax-expert-skip-ssd 3"` gives one arm its own flags).

**What is left** (ranked, with sources, in `_ref/research/DECODE_GAP.md`): size the pinned RAM tier from the box
rather than "available - 2 GiB" (87 experts/layer on the full pack against ~115 that fit — the SSD's 1.09 GB/token
would fall to ~0.3-0.4), read the SSD tier through the page cache when the pages are already resident (Strata's
zero-copy + `mincore`, 1.14-1.47x on its own 12 GB test), and prefetch the next layer's SSD misses (the exact next
routes already exist in `prefetch_for`).
## P7: the RAM tier sized from the box, the page-cache path, one-layer-early reads

P6 made the SSD tier's reads overlapped. P7 attacks the bytes themselves, and the two properties of the box the
engine used to ignore.

**The pinned RAM tier is sized from the machine, not from "available - 2 GiB".** That constant was wrong at both
ends: on a 32 GB box it is 6% of the machine, on a 128 GB box it is less than the desktop plus the tools that talk
to the engine, and it ignored the dense weights that must stay resident. The budget is now

```
budget = min(physical, available now) - headroom - dense_host - host_kv - host_workspace
headroom = physical / 16, clamped to 3-10 GiB     (--minimax-ram-headroom-mib overrides it)
```

* `dense_host` is every dense tensor the engine did not upload (attention projections, dense MLP, the shared
  expert, the router, the embedding, and `lm_head` when it stays on the host). They are read on every token, so
  a tier that takes their pages away buys SSD misses with page faults, which is worse.
* `host_kv` is the fp16 K/V and indexer K for the whole context (zero with `--minimax-kv vram`); the workspace is
  the token vectors, MLP scratch and the full-vocabulary logits.
* The plan is printed once at startup (`INFO minimax-ram-tier plan budget=... headroom=... dense_host=...`) so the
  arithmetic is visible on a machine that behaves differently from this one.
* If the operating system refuses to pin that much (`PinnedArena` fails), an automatic budget steps down 15% at a
  time and says so, instead of failing the run. An explicit `--memory-limit-mib` or
  `--minimax-ram-experts-per-layer` still fails loudly: the operator asked for that number.
* The slots are then split **per layer by the routing profile**, not uniformly: the global ranked list is walked
  hot to cold and each entry offers its layer one more slot for its next-hottest expert that is not already going
  to VRAM, until the byte budget is spent. A layer the profile says is routed twice as often gets twice the slots,
  and no layer is starved (each remaining slot is filled round-robin from whatever budget is left). Layers'
  regions are 2 MiB-aligned. With no profile the split is uniform, as before, and
  `--minimax-ram-experts-per-layer N` still pins exactly N per layer. The idea is Project Maya's priced layer
  split (`STRATA_GLM_*` in its `docs/SETTINGS.md`, MIT) and Strata's profile-filled cache; the arithmetic here is
  this engine's own.

**`--minimax-ssd-zerocopy on|off` (default on): an expert whose pages are already resident is not read at all.**
Before a route's three reads are queued, the three page ranges are queried (Windows `QueryWorkingSetEx` through
`K32QueryWorkingSetEx`; `mincore` on Linux -- the same question Strata 0.2.0 asks, `docs/DETAILS.md`: *"before,
every such expert was first assembled into a staging buffer"*). If every page is resident the dot kernels read the
mapping in place and nothing is queued; `ssd_mapped` in `--minimax-stats` counts the hits. This is free when it
fires and costs one query when it does not.

**`--minimax-ssd-prefetch on|off` (default off): the next layer's reads, issued a layer early.** `prefetch_for`
already applies layer *l+1*'s router to the residual to fill VRAM slots; it now also queues *l+1*'s SSD misses, so
they are in flight across that layer's attention, router and shared expert. It is scored from the layer that is
running now: its post-attention FFN input fed to the future layer's router. `--minimax-ssd-prefetch-depth N`
(1-8) chooses how many layers ahead. The staging pool has a region per layer slot, each route's slot is recorded,
and `ssd_pf_hit` / `ssd_pf_waste` in the totals line say how good the prediction was -- which is the number that
decides whether it is worth turning on. See the P11 section at the end for the three bugs that were fixed and why
the default is still off. (Strata-DS measures next-layer copying at 52.2 -> 49.6 ms/token and Project Maya
19.64 -> 20.22 tok/s, both with a much larger resident tier.)

**CPU threads: the default is the physical core count, not the logical one.** Decode is bandwidth-bound -- the CPU
expert dots stream 2.4-4 GB per token out of the RAM tier -- and on a 12-core/24-thread part the second SMT thread
per core adds contention. `--cpu-threads 0` (the default) now counts `RelationProcessorCore` entries on Windows
(`core_id`/`physical_package_id` on Linux) and prints `INFO minimax-threads=... (source=..., logical=...)`; the old
code used `hardware_concurrency()` and capped at 24.

**A crash in the batched prompt path, fixed.** With `--minimax-kv vram` and `--minimax-dense-layers-gpu 0`, batched
prefill chose its device-attention path only when the layer's dense weights were *also* on the GPU
(`dev_attn = gpu && kv_mode == 1`), took the host-attention branch otherwise, and read a null host K/V:
`--minimax-kv vram` allocates none. On the full model that was an access violation about 50 s into the prompt. The
choice now follows the KV's home (`dev_attn = kv_mode == 1`) and the layer's attention output projection runs on the
CPU when its weights are not on the device; `attend_host()` also refuses a missing host K/V by name.
`tools/minimax/test_p7_kv_prefill.py` holds the combination to `prefill=batch` + `kv_space=vram` and compares it
with the token path (max abs < 1e-2 on the synthetic fixture), plus the `--minimax-kv host` decode comparison.

## P8: long context (32 K, 128 K, 256 K)

Everything before this phase was measured at 512-4096 tokens. Two costs grow with the context and neither was
addressable by a flag: the KV cache's size, and the MSA *indexer*'s score loop, which is linear in the context and
ran as scalar dots on one thread. The analysis (bytes per token, placement, decode traffic, the prefill ceiling) is
in `_ref/research/LONG_CONTEXT.md`; this is what the engine does about it.

### What the KV cache costs

| context | f16 | q8_0 K/V + f16 indexer | q8_0 everything |
|---|---|---|---|
| 32 K | 4.22 GiB | 2.46 GiB | 2.24 GiB |
| 128 K | 16.88 GiB | 9.84 GiB | 8.96 GiB |
| 256 K | 33.75 GiB | 19.69 GiB | 17.92 GiB |
| 1 M | 135.0 GiB | 78.75 GiB | 71.75 GiB |

2304 bytes/token/layer at f16 (K 1024 + V 1024 + indexer K 256), 60 layers. The indexer keys stay fp16 in every
mode: they only *rank* blocks, and a flipped rank is a discrete change in what a query sees, not a small error.

### `--minimax-kv-quant f16|q8_0|auto` (default f16, byte-identical)

q8_0 blocks: 32 values share one fp16 scale, 34 bytes per block, laid out along `head_dim` so one 128-dim head is
four whole blocks and a head row is self-contained. K and V independently. The host and pinned-RAM tiers hold the
blocks; the device attention kernels read fp16 rows, so `--minimax-kv vram` with `q8_0` is refused by name rather
than silently ignored.

Measured on the synthetic runnable fixture (4 layers, head_dim 32 -> one block per head, the worst case for q8_0):
mean KL 6.0e-4 nats over 96 rows, top-1 agreement 94-97%, max abs logit difference 0.093 on a logit standard
deviation of 1.0. That is inside the project's accepted quantized-path KL (the fp16 prefill activation was
accepted at 1.7e-3). `tools/minimax/test_p8_longctx.py` gates on that KL.

`auto` is a capacity decision, not a default: fp16 unless the process cannot commit it, and the plan line says which
way it went and why.

### The plan line

One parseable line at startup, on every path:

```
INFO minimax-kv plan: ctx=131072 quant=f16 kv=16.875GiB placement=ram source=auto indexer=f16 rows=294912B/layer host_attn=fast
```

`placement` is `host` (fp16 in std::vector, CPU attention), `vram` (per-layer device tensors) or `ram` (pinned host
memory, indexer keys on the card). `source` says whether `--minimax-kv` decided it or the engine did. In the `vram`
tier the line also carries `per_card=[...]` so an unequal split is visible before the run.

**A placement bug, fixed.** The pinned-RAM tier used to charge the host KV against the *VRAM* budget, so
`--minimax-kv ram` refused any context whose KV was larger than the cards' free memory -- exactly the case that
tier exists for (17 GiB of f16 KV at 128 K against ~7 GiB of free VRAM). The host KV is now what decides, and the
run refuses with the byte count and the available commit when it cannot fit.

### `--minimax-host-attn fast|legacy` (default fast, byte-identical)

`attend_host` allocated 65 vectors per token per layer (five in `select_slots`, one `prob` per query head) and ran
the indexer's block scores -- `idx_heads x every key x 128` dots -- on the calling thread. It is now:

* allocation-free: the block scores, the selection scratch and one `prob` slice per pool band are sized once;
* parallel over (indexer head, block) for the scores and over query heads for the attention, using the same pool
  the GEMV rowwise splits use. Every output is independent of every other, and each row's dots and softmax run in
  the original order, so the arithmetic is unchanged.

Measured on a 6-layer synthetic fixture with the real attention geometry (hidden 1024, 16 q / 4 kv heads,
head_dim 128, 4 indexer heads, ctx 32768), 2048-token prompt, `--cpu-threads 12`, CPU-only:

| attention | wall | rate |
|---|---|---|
| `legacy` | 251.3 s | 8.15 tok/s |
| `fast` | 73.6 s | **27.82 tok/s** |

3.4x on that workload, and the speedup grows with context because the indexer's share of the time grows with it.
The logits are bit-identical (`test_p8_longctx.py::test_host_attn_fast_is_bit_identical` requires
`np.array_equal`, not a tolerance).

### `--minimax-prefill-kv auto|off` (default auto): the prompt pass on the card, whatever holds the KV

With the KV in pinned RAM, batched prefill had no device path (`dev_attn = kv_mode == 1`), so it fell back to
`attend_host` per token -- the ~1 s/token at 32 K above. It now **stages one layer's whole K/V prefix on the card**
(a single large copy per layer, `[pos0+T][n_head_kv][head_dim]` fp16, the same layout the device tier uses, so the
selected block ids are absolute positions in both) and runs the existing `msa_attn_batch`. The staging buffer costs
one layer's KV -- 256 MiB of K plus 256 MiB of V at 256 K -- not the whole cache.

That move is `60 x KV_size` bytes for a prompt, i.e. **the whole cache once**, against the per-token gather the
decode path uses, which re-reads the same block for every query that selects it (32x more at 32 K):

| context | staged bytes for the prompt | at 22 GB/s (Gen4 x16) |
|---|---|---|
| 32 K | 4.2 GB | 0.19 s |
| 128 K | 16.9 GB | 0.77 s |
| 256 K | 33.8 GB | 1.54 s |

`off` keeps the P7 behaviour. In the pinned-RAM tier the indexer keys are device-only, so `off` now refuses by name
instead of dereferencing a null host pointer (it used to take an access violation), and it says what to do.

### What is not done

* q8_0 is not wired to `--minimax-kv vram`; that needs a dequantizing MSA kernel.
* `--minimax-kv ram` + `q8_0` keeps the host attention in prefill (the staged path wants fp16 rows), so a q8
  long-context prompt is still slow. The RAM tier at long context wants fp16 for prefill and q8 for capacity: those
  two do not yet coexist.
* Prefill is still CPU-expert-bound on the full model; the expert bytes per chunk, not the attention, are the
  ceiling. See �3.3 of the analysis.
## P9: the budget has to cover the page cache, and the chunk the caller asked for

Round 1's 32 K run held **112.8 GB of working set with 0.3 GB of RAM available** while the pinned expert tier was
inside its 74 GiB `--memory-limit-mib` budget. The budget bounded the arena; nothing bounded the 146 GB model being
streamed through the OS page cache by `--minimax-prefill-read mmap` (the default), and memcpy-ing the tier out of
that mapping left the same bytes cached a second time.

### `--minimax-prefill-read auto|mmap|direct` (default auto)

`auto` means: **direct when `--memory-limit-mib` is set**, mmap otherwise. `direct` reads every streamed expert
unbuffered (`DirectFile`) into the pinned staging buffer, so it costs no page cache; `mmap` is unchanged and is
still the faster choice when the whole expert set is meant to be cached. The bytes are identical, so the logits are
identical -- `tools/minimax/test_p9_direct_read.py` asserts `np.array_equal` against the mmap path on the synthetic
runnable model (CPU only, no CUDA), and the plan line says which path ran:

```
INFO minimax-prefill-read=direct (implied by --memory-limit-mib; --minimax-prefill-read mmap keeps the page cache)
```

The three places that read the model now honour it: the profile/static RAM-tier fill (`ram_copy`), the batched CPU
MoE (`moe_batch_cpu`, which also reads a resident expert from the pinned arena instead of the mapping), and the
batched device MoE ring (unchanged `direct` path). On Windows the process working set is also trimmed
(`EmptyWorkingSet`, after the fill and every 8 layers) so the clean mapped pages the upload faulted in go back to
the standby list instead of sitting next to the pinned arena.

### The batched workspace is 137 KiB/token, and the expert slots no longer take its VRAM

Two separate space bugs kept `--minimax-prefill-batch 32768` from being honoured:

* The per-token workspace was sized at `max_dense_n` (12288) for three fp32 buffers and one fp16 buffer that
  nothing read -- 188 KiB/token. Only the dense MLP's `rows_cap` tile needs that width; the T-wide buffers need the
  attention width (8192) and the hidden size (6144). It is now 137 KiB/token, and the sizes are printed:
  `INFO minimax-prefill-workspace dev=1 free=... MiB reserve=... per_token=137.3 KiB fit=... tokens`.
* `init_tiers` sized the automatic VRAM expert slots *before* the prompt workspace existed, so the slot count spent
  all of the free VRAM and `init_prefill` then clamped the chunk to the leftovers. With `--minimax-prefill-batch`
  set (and no explicit `--minimax-vram-slots-per-layer`) the workspace is now reserved first:
  `INFO minimax-vram-slots reserving N MiB/device for the 32768-token batched prompt workspace`.
  A run that does not ask for a batch sizes exactly as before.

`--minimax-prefill-read` and the workspace layout change nothing about the numbers a chunk produces: the same
kernels read the same bytes. `--minimax-vram-slots-per-layer 0` still forces the old "no VRAM slots" behaviour.

### Measuring it

`tools/minimax/bench_longctx.py` runs a real-text prompt (tokenized with the model's `tokenizer.json`, not random
ids) through the serve protocol and reports the engine's own prefill/decode numbers plus **peak working set** and
**minimum available RAM**, sampled every 2 s, and kills the run at `--cap-s` with the per-chunk `PP` lines intact:

```powershell
python tools/minimax/bench_longctx.py --model m.gguf --text docs/DETAILS.md --length 32768 --decode 64 `
  --ctx 33280 --cap-s 1500 -- --gpus 0,1 --minimax-tiers --minimax-profile mm3-d3.bin --minimax-kv ram `
  --minimax-prefill-batch 32768 --minimax-vram-slots-per-layer 0 --memory-limit-mib 66000
```

### Measured: 32 K of real text on the full model

MiniMax-M3.1-Flash IQ2_XS-D3 (146 GB), 2x RTX 5070 12 GB, 128 GB DDR4-3200, Ryzen 5900X, prompt = 32 768 tokens of
this repository's `docs/DETAILS.md`. Flags as above plus `--cpu-threads 12`; RAM tier 64.5 GiB pinned (3 815
slots), KV f16 4.28 GiB in pinned RAM.

| | value |
|---|---|
| chunk achieved | **22 473 tokens** (binding card GPU1, 3 938 MiB free; workspace 2 879 MiB/device) |
| chunks | 2 (22 473 + 10 295) |
| prefill | **866.7 s, 37.8 tok/s** |
| decode, 64 tokens | **34.9 s, 1.83 tok/s** (546 ms/token) |
| peak working set | **83.4 GiB** |
| minimum available RAM | **39.8 GiB** |
| startup (weights, pin, fill, workspace) | 76 s |

Round 1 stopped this prompt at 20 min with 25 of 60 layers done and 0.3 GB of RAM left. The per-chunk split (an
earlier run, same flags, `--ctx 32768`) was 22 991 tokens in **554.3 s (41.5 tok/s)** and 9 777 tokens in
**310.4 s (31.5 tok/s)** — the second chunk pays more per token because the indexer and MSA grow with position while
the expert bytes per chunk do not.

The read paths agree on real weights: the 15-layer slice (`p5a-l0-14-iq2xs.gguf`), 4 096 real tokens, is
`np.array_equal` between `--minimax-prefill-read mmap` and `direct` (4 096 x 200 064 logits, max abs 0.0, top-1
4 096/4 096).

128 K fits in RAM (16.9 GiB of KV + a 55 GiB tier leaves ~35 GiB available) but not in a 25-min run: at 37.8 tok/s it
is ~57 min, and the per-token cost grows with position, so nearer 1.5-2 h. It was not run. The next lever there is
the expert GEMM (an MMVQ at 1 024 rows/expert), not the chunk size.

## P7b: the route nudge, the lane's spin window, and page-cache-in-place

**`--minimax-route-bias on` (default off): rank the router's top-k on the expert's score plus a residency
bonus, and take the weights from the unbiased scores.** This is Xolotl's lever (Indras-Mirror/Xolotl, MIT:
9.7 -> 19.1 -> 21.7 tok/s on one RTX 3090 + 128 GB answering a 148 GB model, `docs/DEEPSEEK_V41.md:60-71`,
`tools/v41/engine/ds41_dense.cpp:1176-1244`) and it is the one change that reduces *which* experts the token
needs instead of making the same bytes faster. With the flag off the selection is the model's own, bit for bit
(the gate below checks exactly that).

* `--minimax-route-bias-in A` (default 0.25) is added to an expert that is in the VRAM or the pinned-RAM tier;
  `--minimax-route-bias-out C` (default 0.15) is subtracted from one that is in neither, i.e. one the token would
  read from the drive.
* The three guards are Xolotl's, and its own quality table is why they are not optional: an unguarded flat bias
  of 1.0 costs 14 points of API top-1 and loops on some prompts, while 0.25 keeps 85.2% against 86.2% exact.
  `--minimax-route-protect P` (default 2) keeps the exact top-P experts always; `--minimax-route-pool M`
  (default 12) makes every pick stay inside the exact top-M (the max-rank window of arXiv:2412.00099);
  `--minimax-route-margin D` (default 0 = everywhere) applies the nudge only to experts whose exact score is
  within D of the k-th exact score.
* The route meter is printed by `--minimax-stats`: `route_swapped` (share of expert picks that differ from the
  exact choice) and `route_weight_kept` (the router weight of the picks against the exact picks' weight).
* The same selection runs in all four routing loops (the tiered token path, the all-CPU token path, the batched
  prompt path and the lookahead in `prefetch_for`), so the prefetch prediction agrees with the decision it is
  predicting.

**`--minimax-pool-spin-us N` (default 0): the CPU pool spins before it sleeps.** A token's layers hand the pool
several small jobs in a row, and a condition-variable wake per job is a context switch each; Project Maya's pool
spins 20 ms through a token's layers for that reason (`ds4_model.cu:5560`). This is a measured knob, not a new
default: 0 keeps the old behaviour.

**`--minimax-ram-mode pinned|inplace` (default pinned): whether the RAM tier is a copy at all.** Maya's DeepSeek
engine reads experts in place and records that its own opt-in RAM tier was *slower* than the page cache on a
V100/64 GB, and that copies "were measured no faster than the page cache � a part copied cost decode speed on
64 GB and on 32 GB of RAM" (`ds4_model.cu:1206-1211`, `5164-5166`). `inplace` builds no arena: the ranked set is
only a ranking decision, `PrefetchVirtualMemory` walks the ranked experts' pages once at startup (Maya's
Windows prefetch of only the non-resident pages measured +6-7% decode, `ds4_os.hpp:250-332`), and the CPU lane
reads the mapping. The A/B is in the table below; the default stays `pinned` until the measurement says
otherwise on this box.

Measured on the 15-layer real-weight IQ2_XS slice with the SSD-heavy placement
(`--minimax-vram-slots-per-layer 0 --minimax-ram-experts-per-layer 11`, i.e. 11 of 128 experts per layer resident and
~27 of 48 routes a token read from the drive), one RTX 5070, 128-token prompt then 24 greedy tokens, one round,
`--minimax-stats`. **The box was not idle**: another agent's full-model quality job shared the drive for the whole
round, so the tok/s column is noisy to about +-4% and the *bytes* column is the load-independent part. Raw rows in
`bench_p7b_arms.json`.

| arm | decode tok/s | ms/token | SSD experts/token | SSD MB/token | time waiting for the drive |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact routing | 4.85 | 206.0 | 27.1 | 423 | 295.5 ms |
| `--minimax-route-bias on` (0.25 / 0.15, protect 2, pool 12) | **4.90** | 204.2 | **17.4 (-36%)** | **271 (-36%)** | **213.8 ms (-28%)** |
| `--minimax-ram-mode inplace` | 3.08 | 325.1 | 43.8 | 684 | 157.9 ms |
| `inplace` + route bias | 3.31 | 302.3 | 36.5 | 570 | 122.4 ms |

What the nudge costs in quality, against this engine's own exact-routing logits on the same teacher-forced prompt
(`gate_logits.py`, real G3 text): mean **KL 0.0348**, 8-row top-1 **0.886**; the 0.5 / 0.3 setting gives KL 0.0348,
top-1 0.877. The meter says **35.3% of picks swapped, 96.6% of the router weight kept** -- a much larger swap share
than Xolotl's 26.1% at the same 0.25, because only 11 of 128 experts per layer are resident in this placement; on the
full model ~110 of 128 are, so the same bias should move far fewer picks. The default stays **off** until the
full-model numbers say what it buys and what it costs there.

**Page-cache-in-place is slower here, and the reason is measurable.** `--minimax-ram-mode inplace` removes the
pinned copy and reads the mapping, exactly what Maya's DeepSeek engine does -- and on this box it costs **-36%**
decode. The counters say why: the token's CPU lane is 2.3 GB of expert reads, and the pinned copy delivers them at
the ~20 GB/s this engine measures for the IQ2_XS dot while the page-cache reads came in at ~5 GB/s in the same slot
(the drive's own wait went *down*, 295 -> 158 ms, so the loss is entirely on the CPU-read side). Maya's own note says
its CPU lane is compute-bound at ~46 GB/s "from either" source on 2 Xeon sockets -- the copy only pays there because
the page cache is already as fast; at 20 GB/s it does not. **The default stays `pinned`.**

### Full model: MiniMax-M3.1-Flash IQ2_XS-D3 (146 GB), 2x RTX 5070 12 GB, 128 GB DDR4

Real text (a 512-token prefix of `ppl_text.bin` from the G3 dump set), 32 greedy tokens, 2 reps per arm, one
round, `--gpus 0,1 --layer-split auto --minimax-profile mm3-d3.bin`, box otherwise idle for the first two arms:

| arm | decode tok/s | ms/token | SSD experts/token | SSD MB/token | time waiting for the drive |
| --- | ---: | ---: | ---: | ---: | ---: |
| before: P5c build (auto 98/layer, uniform) | 2.64 | 379.5 | 10.4 | 180 | not measured (no P6 path) |
| **after: this branch** (auto 94.6/layer, 69..110 by profile) | **2.88** | 347.4 | **7.8** | **135** | **13.5 ms** |
| after: same build, `--minimax-ram-experts-per-layer 87` (the old uniform size) | 2.58 | 386.9 | 8.9 | 154 | 75.9 ms |

**+9.1%** against the P5c build and **+11.6%** against the same build with the old 87-experts/layer uniform tier.
Caveat, stated plainly: the arms ran one after another on the same box, so the later arms found the file's pages
in the OS cache and the first did not; part of the difference is that ordering. The load-independent numbers -
7.8 against 10.4 SSD routes a token, 135 against 180 MB, 13.5 against 75.9 ms of drive wait on the same build - are
the honest measure of what this branch changes, and a cold-start A/B is still to be run with the box to itself.
The tier size in slots barely moves (5394 against 5586): what moves is *which* experts are in it, because the
profile's hot-to-cold split gives the busiest layers 110 and the coldest 69 instead of 98 everywhere. The engine also
reported **21 SSD-tier routes per token served straight from the page cache** (`ssd_mapped`) in this run, which is
the same warming effect and should be read as "the path fires on a warm box", not as a steady-state rate.

At 347 ms/token the drive is now ~4% of a token, the CPU expert lane and the GPU are the rest - which is why the
route nudge (fewer, better-placed experts) and the lane shape are the next levers rather than more I/O work.

### Real weights on the 15-layer slice

`models/quant-d3/p5a-l0-14-iq2xs.gguf`, batched prompt pass, `--gpus 0 --minimax-dense-layers-gpu 15
--minimax-prefill-batch 4096 --cpu-threads 12`, real-text token ids, last-K logits compared:

| arm | context | prompt | wall | logits |
|---|---|---|---|---|
| `--minimax-kv host` (host attention, the P5a path) | 4096 | 4096 tok | 544 s | reference |
| `--minimax-kv ram` (**staged**, new) | 4096 | 4096 tok | **32 s** | KL 6.1e-4 nats, top-1 4/4 |
| `--minimax-kv vram` (device attention, pre-existing) | 4096 | 4096 tok | 42.7 s | KL 6.1e-4 nats, top-1 4/4 |
| `--minimax-kv ram` (**staged**) | 32768 | 8192 tok | 67.7 s | reference |
| `--minimax-kv vram` (device attention) | 32768 | 8192 tok | 64.5 s | `np.array_equal`, max abs 0.0 |

17x on the 4K slice prompt, and the staged path is **bit-identical to the existing VRAM device-attention path**.
The 6.1e-4-nat difference against host attention is the pre-existing device-kernel difference: the `vram` arm
produces the same delta to four digits. 8192 tokens at a 32 K context is 64 blocks with top-16, so the indexer
genuinely drops 48 blocks per query.

q8_0 K/V on the same slice, 2048 tokens: 0.1318 -> 0.0769 GiB (0.583x), **mean KL 3.8e-5 nats, top-1 2/2**, and
faster (136 s against 269 s) because that path is bandwidth-bound.

## P8: what a decode token is made of, and the three phases it pointed at

After P7 the drive was ~4 % of a token (7.8 SSD experts, 13.5 ms of wait) and the rest was unattributed. The first
job here was to attribute it, because "the GPUs are idle and the expert dots are slow" is an inference and not a
measurement. `--minimax-profile-time` is that measurement; the three changes below are the phases it named.

### `--minimax-profile-time`

One line per decode token on stderr, and the mean of the run in the `--minimax-stats` totals line:

```
INFO minimax-profile token=527 total=232.1 embed=0.46 attn=36.4 idx_sync=12.9 msa_sync=3.9 router=5.3 pick=0.1
gpu_begin=4.58 cpu_ram=142.8 ssd_wait=3.5 cpu_ssd=1.9 gpu_finish=1.9 gpu_stream=4.8 combine=1.3 shared=26.7
admit=2.0 prefetch=0.0 lm_head=1.8 other=3.1 cpu_exp=178 cpu_gbs=22.3 ssd_exp=2
```

Every key is one wall-clock pair around work that already existed, so the flag changes no arithmetic. Off by
default; with it off the token path tests one `bool` per phase.

| key | what it times |
| --- | --- |
| `embed` | the token's embedding row |
| `attn` | the attention blocks of every layer: the norms, the per-layer graph launch (q8_1, q/k/v MMVQ, norm+rope, fp16), the MSA attention kernel, the output projection and the residual add. The indexer readback and the MSA sync are inside this window and reported separately below, so they are a breakdown of `attn`, not extra time |
| `idx_sync` | the MSA indexer's block scores: kernel, device sync, and the readback of the scores to the host that decides the slots |
| `msa_sync` | the MSA attention kernel's device sync and the readback of its output |
| `router` | the router GEMV, the sigmoids, the top-k selection (including the P7b nudge) and the skip renormalisation |
| `pick` | classifying each route into VRAM / pinned RAM / SSD, including the page-cache probes |
| `gpu_begin` | uploading the expert pointers and launching the GPU experts on the side stream |
| `cpu_ram` | the CPU lane's pass over the pinned-RAM experts |
| `ssd_wait` | the wait for the layer's queued SSD reads |
| `cpu_ssd` | the CPU lane's pass over the experts that had to be read |
| `gpu_finish` | the side stream's join and the copy-back of the GPU experts |
| `gpu_stream` | the **device-busy** time of that side stream, measured with timing events: the pair `gpu_stream` / `gpu_finish` separates "the card was slow" from "the card was idle" |
| `combine` | the weighted sum of the routes |
| `shared` | the shared expert's MLP |
| `admit` | copying this layer's SSD experts into the RAM tier |
| `prefetch` | the next layer's lookahead (off by default) |
| `lm_head` | the final norm and the output projection |
| `other` | everything the clocks did not name (the residual adds, the counters). A large value is a missing timer, not work |
| `cpu_exp`, `ssd_exp`, `cpu_gbs` | experts on the CPU lane, how many of them came off the drive, and the lane's achieved GB/s (the lane's bytes over its two passes) |

`bench_p6_ssd.py` parses these lines and prints the phase table under the decode table when an arm runs with the
flag.

### The full model, measured

MiniMax-M3.1-Flash IQ2_XS-D3 (146 GB), 2x RTX 5070 12 GB, 128 GB DDR4-3200, Ryzen 9 5900X. A 512-token prefix of
`ppl_text.bin` (real G3 text), 16 greedy tokens, `--gpus 0,1 --layer-split auto --minimax-profile mm3-d3.bin
--memory-limit-mib 90000 --minimax-prefill-batch 512 --minimax-prefill-read direct`, box to itself. **The prompt ran
token by token in this path** (the run's profile covers 527 tokens = 512 prompt + 16 decode-1, so `prefill_batch`
did not engage; each arm therefore costs ~5 min, most of it the prompt). The RAM tier came out at 65..106 experts a
layer (mean 91.3, 5203 slots); the token's routes were **66 VRAM / 136 pinned RAM / 9.3 SSD** (161 MB from the
drive). One round, arms in this order, so the later arms found a warmer drive:

| arm | decode tok/s | ms/token | SSD experts/token | SSD MB/token |
| --- | ---: | ---: | ---: | ---: |
| pre-P8 build | 4.17 | 239.8 | not counted (no `--minimax-stats`) | — |
| **this branch** (12 threads, the default) | **4.37** | **229.0** | 9.3 | 161 |
| this branch, `--cpu-threads 24` | 4.31 | 232.1 | 9.4 | 162 |
| this branch, `--minimax-route-bias on` | **4.70** | **212.9** | **4.3** | **75** |

**+4.8 %** for this branch at the default thread count. The phase table below is the deliverable; everything else
in this section is measured against it.

| phase | ms/token | share | what it says |
| --- | ---: | ---: | --- |
| `cpu_ram` | **108.4** | 44 % | the CPU lane's pinned-RAM pass: 146 experts a token, ~2.5 GB, **~22 GB/s** (see the lane section) |
| `attn` | 41.6 | 17 % | the attention blocks of 60 layers; `idx_sync` (~14) and `msa_sync` (~6) inside it are device syncs + host readbacks |
| `shared` | 27.5 | 11 % | the shared expert's MLP: three `gemv` calls a layer, each with its own stream sync |
| `ssd_wait` | 19.6 | 8 % | the drive, for 161 MB a token |
| `pick` | 13.5 | 5.5 % | classifying 212 routes, including the page-cache probes |
| `admit` | 9.5 | 4 % | copying the 9.3 SSD experts into the RAM tier (162 MB, 17 GB/s over the pool) |
| `cpu_ssd` | 6.6 | 2.7 % | the same lane, for the experts that had to be read |
| `router` | 5.4 | 2.2 % | the router GEMV, sigmoid and top-k per layer |
| `gpu_begin` | 4.3 | 1.8 % | the expert pointer uploads and launches |
| `gpu_stream` / `gpu_finish` | 4.7 / 1.9 | 2.7 % | the card was **busy 4.7 ms** of the token and the host waited 1.9 ms for it |
| `other` | 3.4 | 1.4 % | residual adds and counters |
| `lm_head` | 1.7 | 0.7 % | the final norm and the 200 K output projection |
| `combine` | 1.1 | 0.5 % | the weighted sum of four routes |
| **total** | **244.4** | | |

`attn` splits, in the run's own mean over its 527 tokens, into `idx_sync` 13.8 ms (57 device syncs and readbacks of
the block scores, one per layer) and `msa_sync` 6.1 ms of device sync, with the rest - the norms, the per-layer
graph, the output projection, the residual add - about 22 ms. Both syncs are host round trips that a device-side
slot selection and a device-side combine would remove; that is the "de-synchronise" item, and it is worth ~20 ms
(8 %) before any graph work.

Two things fall out. **The CPU lane is the token** (108 + 6.6 ms, 47 %), and **the idle GPUs are idle by
construction**: the GPU expert stream is busy 4.7 ms, the attention kernels sit inside their own syncs, and the
cards never have more than one layer's work queued. The drive is 8 %, and `pick` + `admit` together are 9.5 % -
more than the drive.

### The CPU lane: two parallel regions instead of twelve

`mlp_ptr` ran `dot_rows_base` for gate, up and down as three separate `pool.run` calls per expert: twelve barriers
a layer for four routed experts, and with the pool's spin window at its default 0 every barrier after the first
wakes eleven sleeping threads. `mlp_ptr_multi` dispatches the whole pass instead - gate and up rows of all four
experts in one region (they share one activation image), the down rows in a second - so a layer pays two barriers
and quantizes its activation once instead of twice.

Measured on the 15-layer real-weight IQ2_XS slice with every expert in the pinned tier (`--minimax-tiers
--minimax-vram-slots-per-layer 0 --minimax-ram-experts-per-layer 128`, 48 routes a token all on the CPU, one RTX
5070, 512-token prompt then 16 greedy tokens, one round):

| arm | decode tok/s | ms/token | lane ms/token | lane GB/s |
| --- | ---: | ---: | ---: | ---: |
| pre-P8 build, 12 threads | 17.52 | 57.1 | — | — |
| **this branch, 12 threads** | **19.38** | **51.6** | 36.0 | 20.8 |
| **this branch, 24 threads (SMT)** | **24.64** | **40.6** | **26.1** | **28.7** |

**+10.6 %** from the fused regions alone, and **+27 %** more from using the second thread of each core. On this
placement the lane is compute-bound (the 24-thread lane moves 28.7 GB/s against 20.8 GB/s, and a raw stream of the
same bytes is 39.5 GB/s), so the second SMT thread is not contention - it is the difference between 17.5 and
24.6 tok/s.

**That does not transfer to the full model**, and the full-model table says why: there the same arm at 24 threads
was 4.31 against 4.37 tok/s (**-1.4 %**). The lane in the full model runs beside the SSD reader threads and the
admissions that copy 162 MB a token, and 22-23 GB/s is what the memory system gives this access pattern once the
drive and the copies are in it; the extra SMT threads only add contention. So the default stays the physical core
count, and the slice's SMT result is recorded as placement-specific rather than as a default change.

A software-prefetch probe (`STRATA_LANE_PF=N`, rows ahead) is in the lane's row loops for the same reason the
default is 0: the hardware prefetcher already sees the sequential row stride, and an unmeasured prefetch hint is
the kind of change `SPEED_RESEARCH.md` §4 measured *slower* (15-18 tok/s against 24). It is an environment probe,
not a flag, and off unless set.

### The RAM-tier admission

Admitting a routed expert into the pinned tier is a 17.4 MB copy that used to run on the calling thread, between
two layers, once per SSD miss. `ram_admit_many` picks the victims serially (still LRU) and then copies the whole
layer's admissions in ~1 MiB pieces over the pool. On the full model the phase now costs **9.5 ms for 162 MB
(17 GB/s across 12 threads)**, measured with `--minimax-profile-time` above. The before number is from the
SSD-heavy slice placement earlier in this round (`--minimax-ram-experts-per-layer 11`, 27-30 of 48 routes read
from the drive), where the serial per-expert copy showed up as **16-60 ms a token** in the same profile; that is a
different file and placement, so it is a scale, not a delta. **A clean slice A/B of this change was cut for box
time and is not in the table.**

### The route nudge: measured speed, unmeasured quality

With the whole box, the same placement and the same prompt, `--minimax-route-bias on` (the P7b defaults: in 0.25,
out 0.15, protect 2, pool 12) is **4.70 against 4.37 tok/s (+7.5 %)** and halves the drive's share of the token
(9.3 -> 4.3 SSD experts, 161 -> 75 MB). The engine's own meter says **9.1 % of expert picks are swapped and 99.5 %
of the router weight is kept** - against 35.3 % swapped on the slice placement that has only 11 of 128 experts
resident, because the full model's tier holds 91 of 128.

What is *not* measured: KL and top-1 for the nudge on the full model. The plan was
`tools/minimax/gate_logits.py` over the six short G3 prompts concatenated into one 320-position teacher-forced
pass (one model load an arm); the PC window ran out after the speed arms. The extrapolation from the meter -
Xolotl's own table has 26.1 % swapped at 85.2 % API top-1 against 86.2 % exact, so 9.1 % should cost well under a
point - is an extrapolation, not a measurement. **The default stays off** until that run exists.

## P10: the shared expert beside the lane, the indexer on the card, and a prefill-aware slot count

P8 named three phases to attack: the shared expert's three host round trips, the indexer's readback, and the
page-cache walk inside `pick`. This round does those three, plus the automatic-slot OOM on a big card.

### The shared expert runs beside the routed-expert CPU lane (bit-identical)

The shared MLP is three dense GEMVs (Q8_0 here) and the old path ran them with three host round trips between the
routed experts and the next layer. Nothing about it depends on routing: its input is the layer's MoE input and its
result is added after the combine. `shared_begin` now enqueues the gate/up pair on its own stream **after routing**
(so the router's GEMV is not delayed) and just before the CPU lane; `shared_finish` waits after the lane, runs the
host `swigluoai` and the down GEMV, and returns. The tensors, the q8_1 activation, the native MMVQ kernels and the
host swiglu are the same ones `mlp` used, so the result is bit-identical. Only the blocking cost is left in
`shared=`, which is the wait plus swiglu plus the down GEMV.

### The indexer and the attention output stay on the card (`--minimax-kv vram`)

`idx_select` is the host's `select_slots` as a device kernel: one thread per indexer head, the same ranking
(score desc, block index asc), the same forced local blocks, and the same ascending position order in the slot
list. With the KV in VRAM the block scores and the slot lists no longer cross PCIe and there is no sync between the
indexer and the MSA kernel. The attention output also stays on the card: `gemv_dev_in` feeds it to the output
projection through the legacy default stream (native MMVQ needs a non-null handle but must stay ordered with the
MSA kernel), so the D2H of `att` and the H2D back are gone. The pinned-RAM KV tier still selects on the host,
because it needs the counts to know how many rows to copy.

### `--minimax-page-probe N` (default 32; 0 = walk every page)

P7b sampled eight spread pages, then walked all 4-8 K pages of a 17 MB expert when the sample passed. A sampled
"yes" is only ever a page fault on the pages it did not see, never a wrong byte, so the default is now 32 sampled
pages and no walk. `0` restores the full walk.

### Automatic VRAM slots reserve the batched-prefill workspace

The old rule was `slots = (free - reserve) / (layers * expert)` at sizing time. The batched-prefill workspace,
the rows_cap scratch, the expert ring and the RAM tier's staged prompt K/V are allocated later (`init_prefill`), so
on a card big enough for the slots to fill it - the 1 x RTX PRO 6000 96 GB, 153 GB model report - the prefill
allocation died in `cudaMalloc` at startup. `prefill_device_bytes()` mirrors those allocations and the auto plan
subtracts them before counting slots. The rule lives in `include/strata/minimax/vram_plan.hpp`, is header-only and
is unit-tested with no GPU (`tests/minimax/vram_plan_test.cpp`, `tools/minimax/test_vram_plan.py`). An explicit
`--minimax-vram-slots-per-layer` is unchanged: the prefill batch shrinks to the room that count left, as before.
With `--minimax-prefill-batch 0` (this engine's default) the plan is byte-for-byte the old one.

### Measured

**Bit-identical gate.** Real-weight 15-layer slice `models/quant-d3/p5a-l0-14-iq2xs.gguf`, 96 real-text tokens,
`--gpus 0 --minimax-dense-layers-gpu 15 --minimax-tiers --minimax-vram-slots-per-layer 0
--minimax-ram-experts-per-layer 128 --minimax-kv vram --cpu-threads 12 --memory-limit-mib 40000`, merged base
(`1390bdb8`) against this branch: **max abs 0.0, KL 0.0, top-1 1.000, identical**.

**Full model, same run setup.** MiniMax-M3.1-Flash IQ2_XS-D3, 2 x RTX 5070 12 GB, `--gpus 0,1 --layer-split auto
--minimax-profile mm3-d3.bin --memory-limit-mib 90000 --minimax-prefill-batch 0 --minimax-profile-time`, 64-token
real-text prompt then 32 greedy tokens, one round, merged base against this branch:

| phase | base ms/token | this branch | delta |
| --- | ---: | ---: | ---: |
| `attn` | 37.6 | **27.8** | -9.8 |
| - `idx_sync` | 13.9 | **0.6** | -13.3 |
| - `msa_sync` | 4.2 | **0.3** | -3.9 |
| `shared` | 27.4 | **15.5** | -11.9 |
| `pick` | 0.8 | **0.4** | -0.4 |
| `cpu_ram` | 134.3 | 135.7 | +1.4 |
| `ssd_wait` | 26.1 | 75.7 | +49.6 (drive) |
| `admit` | 13.4 | 13.0 | -0.4 |
| `total` | 271.3 | 300.4 | +29.1 |
| **total - `ssd_wait`** | **245.2** | **224.7** | **-20.5 (-8.4 %)** |

The CPU lane, the drive and the admissions are the same work in both arms; the raw wall time moved the wrong way
because the second arm hit a slower drive window (26.1 -> 75.7 ms of wait for the same ~248 MB). **The compute-side
cost of a token fell 20.5 ms (-8.4 %).** Raw decode was 3.44 tok/s (base) and 3.30 (this branch) in that pair; an
isolated run of this branch on the same box measured **4.34 tok/s** (8 greedy tokens, 1.84 s) with the drive
slower than either, which is why the raw number is reported as drive-bound rather than as a win.

The box was not idle (the desktop agent, browser and Task Manager were running) and the NVMe fluctuated between
0.12 and 2.1 GB/s over the session, so the raw tok/s is not comparable to the round-3 4.37 measured on an idle box.
What is comparable is the per-phase table on the same pair of processes.

### The route nudge: now measured, and it stays opt-in

`tools/minimax/route_bias_quality.py` builds one teacher-forced stream from the six short G3 prompts and runs
`--minimax-route-bias off` and `on` with the P7b defaults (0.25 / 0.15, protect 2, pool 12). 156 positions, full
model, same placement as the speed run:

| arm | KL64 mean | top-1 agreement |
| --- | ---: | ---: |
| `off` (reference) | 0 | 100 % |
| `on` | **0.343** | **59.0 %** |

The engine's meter says **9.97 % of picks swapped and 98.97 % of the router weight kept**, the same first-order
numbers round 3 reported (9.1 % / 99.5 %). A two-identical-arms control is bit-identical (KL 0.0), and `on` with
`--minimax-route-bias-in 0 --minimax-route-bias-out 0` is also bit-identical, so the code path itself is neutral
and the loss comes from the nudge. **The decision rule (top-1 drop <= 1 point, KL64 increase <= 0.02) fails by a
wide margin: the default stays off.** This is a different result from the P7b slice (KL 0.0348 at 35 % swaps): with
112 of 128 experts resident, the nudge demotes the few file-resident experts that carry real weight, while the
slice's 35 % were near-tied swaps.

### The 96 GB auto plan

`vram_plan_test` covers the big-card case directly (`reserve + prefill + slots * layers * expert <= free`), the
12 GB case (the prefill reserve costs at most one slot), the no-prefill case (identical to the old rule) and the
degenerate inputs. It passes with the host compiler. The 2 x 5070 auto plan is unchanged when prefill is off
(this engine's default); when `--minimax-prefill-batch` is on, the workspace is taken out first, which is what PR
#19 already did for its per-token part and this branch extends to the rows scratch, ring and staged K/V.

**Tests.** `python -m pytest tools/minimax`: **68 passed**. The two setup errors in `test_g3_sparsa.py`
(`ref_dir`, `name`, `logits`, `ids` are used as fixtures but are not defined) reproduce on the merged base
`1390bdb8` in 0.19 s and are unrelated to this branch.

## P11: the drive's slow windows, measured with `io_probe`

The decode wait on the SSD tier swings between runs (20-76 ms/token, and a single token can spend seconds in
`ssd_wait`), so this branch adds a drive-side probe instead of guessing: `tools/minimax/io_probe.cpp` issues the
same 4 KiB-aligned unbuffered requests the tier issues, at a chosen queue depth, and reports GB/s. Build it with
`tools/minimax/build_io_probe.bat`.

Measured on the full D3 pack's shards, 5.2 MiB per request, 32-200 requests per configuration:

| configuration | first ~1 GB after an idle box | immediately after, same offsets |
| --- | ---: | ---: |
| plain `NO_BUFFERING | OVERLAPPED` | 0.13-0.15 GB/s | 4.9-7.2 GB/s |
| `+ FILE_FLAG_RANDOM_ACCESS` (what the engine sets) | 0.13-0.14 GB/s | 5.1-7.1 GB/s |
| queue depth 1 / 2 / 8 / 32 | 0.13 / 0.13 / 0.13 / 0.15 | 4.9 / 7.0 / 7.2 / 7.2 |

Three things follow. **The flag is not the cause** (the two arms are indistinguishable inside a window, and the
mature `FileExpertSource` path in `src/core/expert_source.cpp` opens plain `NO_BUFFERING | OVERLAPPED` for the
same reason); **queue depth is not the lever inside a slow window** (0.13 GB/s at depth 1 and at depth 32 means
the device is not serving requests, not that it has too few); and **the drive has multi-second degraded windows**
after sustained reads, which the token inherits as `ssd_wait`. The engine can only (a) have fewer bytes to read and
(b) have them in flight before the layer needs them.

### The one-layer-early prefetch, and the three bugs that made it wrong

`--minimax-ssd-prefetch` was reported as producing no completed generation. Three separate faults were found:

1. **It matched a prediction rank-for-rank.** A prediction that named the right experts in a different order was
   counted as a miss and the bytes were read twice. It now matches the predicted *set* (`ssd_pf_exp` is searched).
2. **It reused a staging region while a read into it was still in flight.** Regions are now per
   `layer % (depth + 1)` and the region is drained before reuse.
3. **It scored the predicted layer's router on the wrong CUDA device.** With `--gpus 0,1 --layer-split auto`,
   layer `j`'s MoE predicted layer `j+2` and ran `j+2`'s router GEMV while device `j` was current; the first
   token died with `cuda: gemv sync: an illegal memory access`. `use_layer_device(target)` fixes it. This was the
   "no completed generation": the engine reports the error on **stdout** (`ERR minimax engine: ...`) and exits 0,
   which a harness that only looks for `DONE` sees as silence.

With those fixed, `gate_logits.py` on the 15-layer real-weight slice is **max abs 0.0 / KL 0.0 / top-1 1.000**
(prefetch off against on), and the prediction is scored from the current layer's post-attention FFN input, which
is what Project Maya's device uses.

**It is still not a win on the full model.** The depth is a whole-layer drain (`ssd_wait_all` before reusing a
region), because a per-region completion tag deadlocked after the full model's prefill when a prediction's
completion was not reaped on its shard. On the slice, prefetch on reads 48 % more bytes and spends more time in
`ssd_wait` (60 -> 124 ms); the default stays **off**, and the flag remains opt-in while a per-region drain that
does not leak is written.

## P12: the prompt workspace borrows the decode slots

A batched prompt needs a device workspace (137 KiB/token/device plus the rows scratch, the expert ring and, with
the KV in RAM, one layer's staged K/V). P10 made the auto slot count subtract that workspace up front so a big
card could not fail in `cudaMalloc` at startup -- correct, but the workspace then sat in VRAM for the whole
session while decode used the rest. With `--minimax-prefill-batch 4096` that is ~1.5 GiB per card; on a 12 GB card
it was the difference between a usable slot pool and almost none.

The workspace is only needed *while a prompt is being processed*, so the two no longer coexist:

* `--minimax-prefill-vram swap`: the slots are sized as if there were no workspace. At the start of a prompt the
  engine frees the slots it needs (round-robin over a device's MoE layers, so no layer is emptied), allocates the
  workspace, and at the end of the prompt frees it and re-allocates exactly the slots it took
  (`Impl::pf_acquire` / `pf_release`, `PrefillWs` in `Engine::generate` and `teacher_forced`). The slots come back
  empty, which is the state a decode-only run starts in: the token path's compulsory admission fills them.
* `--minimax-prefill-vram reserve` (**the default**): the P10 rule, unchanged -- the workspace is subtracted at
  startup and lives for the run. It is the default because the swap measured slower on this box's full model (see
  the table below); a placement with a slot pool worth having should turn it on.

`include/strata/minimax/vram_plan.hpp` is untouched: swap mode passes `prefill_bytes = 0`, reserve passes the real
figure. An explicit `--minimax-vram-slots-per-layer N` never swaps (the operator asked for N), so its workspace
still comes out of the free VRAM as before.

**The admission path reads the pinned tier now.** `vram_fill` used to copy a compulsory-admitted expert out of the
file mapping even when the pinned RAM tier already held the same bytes, so every fill re-faulted 15 MB through the
page cache (or the drive). It now copies from the arena when the expert is resident there -- same bytes, and the
`pick` phase of a token stops paying for the drive.

### Measured

Bit-identical gates on the 15-layer real-weight slice (`models/quant-d3/p5a-l0-14-iq2xs.gguf`, 96 real-text
tokens, `--gpus 0 --minimax-dense-layers-gpu 15 --minimax-tiers --minimax-kv vram --cpu-threads 12
--memory-limit-mib 20000`), merged base `4658d3c3` against this branch:

| arm | max abs | KL | top-1 |
| --- | ---: | ---: | ---: |
| `--minimax-vram-slots-per-layer 6 --minimax-prefill-batch 32` (the reserve path, this branch's default) | 0.0 | 0.0 | 1.000 |
| `--minimax-vram-slots-per-layer -1 --minimax-prefill-batch 0` (the token path) | 0.0 | 0.0 | 1.000 |
| the same with `--minimax-prefill-vram swap` (auto slots, 32 a layer, workspace borrowed) | 0.0 | 0.0 | 1.000 |

The swap costs 21-23 ms per turn on that slice: two turns of 96 + 112 tokens, `--decode 8`, borrowed 34 then 38
slots in 9 ms, workspace released and 32 slots a layer re-allocated in 12-14 ms, both turns `DONE`, and the second
turn's decode is 69 ms/token against the first turn's 177 (the slots it filled are warm again).

**Full model, the released server's flags.** MiniMax-M3.1-Flash IQ2_XS-D3 (146 GB), 2x RTX 5070 12 GB, 128 GB
DDR4-3200, 512-token real-text prompt, 64 greedy tokens, 2 reps, all arms interleaved with a fresh process each
rep: `--gpus 0,1 --max-context 32768 --minimax-kv vram --minimax-tiers --vram-reserve-mib 700
--minimax-prefill-batch 4096 --minimax-prefill-expert-gemm mmvq --memory-limit-mib 90000 --minimax-stats
--minimax-profile-time`, auto slots (`--minimax-vram-slots-per-layer -1`):

| arm | slots/layer | decode tok/s | ms/token | ssd/token | ssd MB/token | ssd wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P10 reserve (base) | 0 | **3.33** | 300.2 | 56.7 | 981 | 30.3 ms |
| swap, after the admission fix | 2 | 3.15 | 317.1 | 57.4 | 994 | 35.1 ms |

The per-token breakdown says exactly why: the two slots move work off the CPU lane (`cpu_ram` 184.4 -> 164.9 ms)
but the compulsory admissions that fill them cost more than they save (`pick` 0.6 -> 12.7 ms, plus `gpu_begin`
4.0, `gpu_stream` 3.4 and `gpu_finish` 1.3) -- net **-5.4 %** in this window. Before the admission fix (which
copies a resident expert out of the pinned tier instead of re-faulting the file mapping) the same pair was 3.44
against 2.95 tok/s on 32 decode tokens: the fix takes the loss from -14 % to -5 %.

The reason is the placement, not the mechanism: at 32 K context with the KV in VRAM and 60 dense layers on the
two 12 GB cards, the automatic slot pool is **2 experts a layer** (0 in the reserve arm -- the 1.5 GiB workspace
ate the whole pool, which is the starvation this flag removes). Two of 128 experts cannot pay for their own
fill-in inside a 64-token reply. `--minimax-vram-slots-per-layer 0` (what `sparsa.py` passes today) is unchanged
by any of this: there is no slot pool to borrow from or reserve.

**So the default stays `reserve`.** The flag exists, is bit-identical, and gives decode the decode-only slot count
where a slot pool is worth having; on this box's full model it is a wash that costs 5 %, and the honest next step
is not a bigger workspace but a cheaper admission: the compulsory fill is three synchronous H2D copies on the
default stream, which wait behind the layer's queued GPU work, while `prefetch_for` already knows how to enqueue
the same bytes on the side stream.


## P13: the VRAM slot count is per device (and two measured negatives)

The decode gap to Project Maya, third pass: `_ref/research/MAYA_GAP_3.md` holds the full table and every number.
Two of its rows became measurements here, one of them a change.

### The slot count is per device, not one number for the split

The old rule sized the expert slots as `free / (MoE layers x expert bytes)` **and then took the minimum over the
devices**, giving every layer that many slots. On this box the two cards are not equal: upstream's "last card" rule
puts `lm_head` (Q6_K, 1.0 GB) on the card that owns the last layer, and the KV is per layer per device, so the card
that binds has ~1 GB less room for experts. Measured on the full pack with the new `INFO minimax-vram-slots` line
(`dev=`, `moe_layers=`, `free=`, `slots_here=`, `shared_min=`):

| context | device 0 (25 MoE layers) | device 1 (32 MoE layers) | old shared count | total slots before -> after |
| --- | ---: | ---: | ---: | --- |
| 88 | 10 | 6 | 6 | 342 -> 442 (+29%) |
| 8 192 | 9 | 5 | 5 | — |
| 32 768 | 6 | 2 | 2 | 114 -> 214 (+88%) |

Each layer now gets its own card's count, so a layer's cost is that card's `moe_layers x expert_bytes` and the
split's imbalance costs nothing. An explicit `--minimax-vram-slots-per-layer` stays uniform (the caller asked for
that number), which is also what makes the A/B below possible with one binary. `vram_slots_per_layer` in
`--minimax-stats` still reports the minimum across devices, so old logs stay comparable.

**Measured**, full pack, one binary, 64-token real-text prompt then 16 greedy tokens, 3 `GEN` requests in each of
two interleaved arms, `--gpus 0,1 --layer-split auto --minimax-profile mm3-d3.bin --memory-limit-mib 86000`:

| arm | decode tok/s (3 reps, median) | `--minimax-profile-time` total ms/token | vram hits/token | hit rate | ram/token | ssd/token | ssd MB/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `--minimax-vram-slots-per-layer 6` (the old rule) | 3.73 (4.16 / 3.36 / 3.73) | 233.8 | 61.3 | 26.9% | 150.9 | 13.8 | 251 |
| **auto, per device** (new default) | **4.04** (3.66 / 4.15 / 4.04) | **215.0** | **74.0** | **32.4%** | **139.9** | 12.7 | 230 |

The phase table for the same pair: `cpu_ram` 129.7 -> 118.1, `ssd_wait` 18.6 -> 13.4, `cpu_ssd` 9.8 -> 6.5,
`admit` 9.7 -> 6.5, `gpu_begin` 6.3 -> 7.7, `gpu_stream` 6.5 -> 7.9, `attn` 28.5 -> 29.5. The lane and every
SSD-side phase fall because a sixth of the routes moved from the RAM tier to the VRAM tier. Rep-to-rep spread on
this drive is +-10%, so the median and the profile total are what the claim rests on, not one rep.

The change is inert when the tier is off (`--minimax-vram-slots-per-layer 0`, the all-CPU parity gate): `q == 0`
and `L.vram.size() == 0`, so the allocation, the profile fill and the RAM split all take the same branch as before.
`q_dev` is only consulted when slots exist. **Gates re-run on this branch** (`gate_logits.py`, merge base
`9ada09d0` against this branch, real-weight 15-layer slice, 64 real-text tokens):

| arm | max abs | KL | top1 | |
| --- | ---: | ---: | ---: | --- |
| all CPU (`--minimax-vram-slots-per-layer 0 --minimax-ram-experts-per-layer 128`), 2 GPUs | **0.0** | 0.0 | 1.000 | identical |
| tiers, explicit uniform count (`--minimax-vram-slots-per-layer 4`), 1 GPU | **0.0** | 0.0 | 1.000 | identical |
| tiers, explicit uniform count 4, 2 GPUs | **0.0** | 0.0 | 1.000 | identical |
| tiers auto, 1 GPU (one card, so the placement is the same as the old rule) | 0.0 | 0.0 | 1.000 | identical |
| **tiers auto, 2 GPUs: old 6/6 against new 10/6** | 9.48e-01 | 2.57e-03 | 0.969 | DIFFERS |
| control: new auto (10/6) against new all-CPU, same binary | 9.83e-01 | 4.66e-03 | 0.938 | DIFFERS |

Two-arm controls with *different* arguments on the same path read KL 5.25e-03 / top-1 0.906, so the harness detects
a difference when there is one.

**Why the last two rows are not 0.0, and why that is the tier's existing property rather than this change.** Only
the placement moves between the last two arms - same binary, same flags but `--minimax-vram-slots-per-layer 0` -
and it still differs, so the *VRAM expert kernel is not bit-identical to the CPU one*. That is the P3 tier's
documented behaviour and it is known: the device iq kernels quantize the activation with `q8_1` while the CPU path
uses ggml's `Q8_K` (the synthetic-file gate in the P3 section records the same cause at max-abs 2.1e-2). So
**no gate can be 0.0 across a slot-count change**; the question a slot change has to pass is KL. On the slice,
moving to the larger per-device cache reads **KL 2.57e-03 and top-1 0.969**, against the P7b decision rule
(top-1 drop <= 1 point *for a routing change* and KL increase <= 0.02); the top-1 drop is bigger because a
sixth of the routes change *kernel*, not their expert. A larger VRAM cache is therefore a small numeric change
traded for speed, exactly like any other slot count: the tiers have never been bit-identical and the flag that
makes them so is `--minimax-vram-slots-per-layer 0`.

### Rejected: Project Maya's 20 ms pool spin (`--minimax-pool-spin-us`)

Maya spins 20 ms before a worker sleeps (19 workers on a 20-core part 28.3 tok/s, 18 -> 47.2). Here it loses:
12 threads with a 20 ms spin measured `cpu_ram` 182.8 ms/token against 139.5 without, interleaved, same binary,
and 24 threads with the spin 122.0 against 110.5 without. **The default stays 0.**

### Rejected: 24 threads on the 12-core/24-thread part

The lane does like the SMT sibling - on the 15-layer slice with all 12 MoE layers pinned it is 6 threads 10.7 GB/s,
12 threads 22.0, 24 threads 24.7, i.e. linear in *hardware threads* to 12 and +12% for the sibling - but the pool
shares the box with the thread that issues each layer's GPU work. Same-binary 3-rep A/B: `cpu_ram` 129.9 -> 121.5
(**-6.5%**) while `attn` 28.2 -> 31.1, `router` 5.7 -> 7.1, `gpu_begin` 5.7 -> 8.8, `shared` 15.4 -> 18.4,
`admit` 10.0 -> 17.4, so the token goes 232.2 -> 248.9 ms and 3.52 -> 3.17 tok/s. **The default stays at the
physical core count**; `--cpu-threads` still raises it.

### The lane's ceiling, measured (priority 1 of the round)

`cpu_ram` is 130 of a 232 ms token, so it was the first target. It is **compute-bound per hardware thread**, not
bandwidth-bound: 2.05 GB/s a thread, 24.7 GB/s at 12 cores / 24 threads, and the DDR4-3200 ceiling (~40 GB/s) is
never approached. `ggml_vec_dot_iq2_xs_q8_K` asserts `nrc == 1`, and Strata's AVX2 IQ row kernel is within 5% of it
at one token (P5c), so the remaining bit-identical headroom is the SMT sibling - which the A/B above says is not
worth taking. Project Maya's GLM lane does the *same* arithmetic at one token; its advantage there is a per-expert
`[gate|up|down]` blob and a ~48-piece job split, neither of which changes the inner loop.

### The context default is 16 K (launcher plan and the engine's unasked default)

The KV is f16 and it is allocated for the whole context **per layer per device**, so it comes straight out of the
binding card's expert slots. Measured with the new `INFO minimax-vram-slots` line on the full pack:

| context | KV (f16) | slots/layer device 0 | device 1 |
| ---: | ---: | ---: | ---: |
| 88 | 0.01 GiB | 10 | 6 |
| 8 192 | 1.05 GiB | 9 | 5 |
| **16 384** | 2.11 GiB | **8** | **4** |
| 32 768 | 4.22 GiB | 6 | 2 |

`strata_cli/launch/catalog.py::max_context` recommended 32 K for a card under 14 GB and the engine capped an
unasked context at 32 K; both are 16 K now, and `--max-context` (or the dashboard's context control) still raises
it. The trade is explicit: +2 slots a layer on the binding card for a conversation that does not need 32 K.

### Where the MoE layers go across two unequal cards: the link, not just the slots

The split is picked before any weight is uploaded, so it is priced on layer bytes. Measured with `--layer-split K`
(full pack, ctx 16384, 3 interleaved `GEN` reps, same binary):

| K (layers on device 0) | slots/layer dev0 / dev1 | decode tok/s (median) | profile ms/token | `cpu_ram` | ssd MB/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| auto (29/28) | 5 / 6 | 3.56 | 226.2 | 113.4 | 231 |
| 30 (27/30) | 7 / 5 | 2.43 | 867.4 | 100.9 | 212 (drive window: `ssd_wait` 672 ms, discard) |
| **24 (21/36)** | **12 / 2** | **4.08** | **196.6** | 100.0 | 246 |

`--layer-split 24` is **+14.6%** over auto even though its total slot count (324) is *below* auto's (328), and the
coverage curve of §3 cannot tell the two apart (7.04 against 7.03). The difference is the link: device 0 is on
**PCIe 3.0 x4** (Gen4 x16 on device 1), so every expert it holds is copied over a quarter of the bandwidth -
`gpu_stream` and `gpu_begin` fall when its MoE layers are fewer. That is the term Project Maya's `STRATA_GLM_SPLIT`
prices ("each layer's reads over each card's memory bandwidth") and this engine does not have.

An attempt to make `auto` search the split points is **not** in this branch: the search runs before
`place_weights()`, where a device's real free VRAM (after its own work buffers, MSA scratch and the model's fixed
allocations, ~3 GB on this box) is not yet known, and a model of it misprices the choice - the first attempt picked
56/4 by maximising the slot *sum*, which is unbounded, and the coverage-curve version picked a split 14 points
worse than the measured best. The honest fix is to move the decision after the dense upload (the refactor the
per-device slot count already needed), and until then `--layer-split 24` is the measured answer for this box.

### A1: the RAM-fit pack, per-layer mixing (assembled, not yet accepted by the engine)

`tools/minimax/mix_layers.py` builds a mixed pack by copying whole layers' expert tensors, bit for bit, from two
published packs, so nothing is re-quantized. Measured sizes for this pack set:

| | bytes/layer of experts | 57 layers |
| --- | ---: | ---: |
| D3 (gate/up IQ2_XS, down IQ3_XXS) | 2.322 GB | 132.3 GB |
| IQ1_M (all three IQ1_M) | 1.680 GB | 95.7 GB |

25 of the 57 MoE layers taken from IQ1_M gives **116.3 GB of experts** (43.9% of the layers, 40.5% of the profile's
routing mass, deepest-first), a ~130 GB file - i.e. `ssd/token` 0 with ~6 GB of it in VRAM slots instead of pinned.
The assembled file passes a structural read (`1188` tensors, per-layer types as intended) but the engine rejects it
at the last tensor: the source is a 13-shard pack whose `split.no`/`split.count`/`split.tensors.count` keys are
`u16`/`i32` and describe 13 shards, and a single-file output has to drop them (or be written as 13 shards). That is
the next step, with the G3 six-prompt gate, before the mix is a default or even a documented recipe.

### A1, continued: the mixed pack loads, runs +54%, and one path still refuses it

`mix_layers.py` now writes a single-file GGUF the engine's reader accepts (`1188` tensors, `split.*` dropped,
`kv_count` re-emitted, and the data block aligned on its **absolute** position - the key/value block ended 8 bytes
into an alignment boundary on this pack, and aligning the info block's length instead of the file's position put the
data 8 bytes past the table, which every tensor then read shifted). Audited tensor by tensor with gguf-py: `token_embd`,
`blk.0.attn_norm`, `blk.30.ffn_gate_exps` (D3 shard 7), `blk.59.ffn_gate_exps` (IQ1_M shard 13) and `blk.59.attn_norm`
(D3 shard 13) are **byte-identical** to their source packs.

**Measured, serve path, 3 interleaved reps, ctx 16384, auto slots, profile mm3-d3.bin** (D3 at the same settings
below for the phase comparison):

| | D3 | MIX | |
| --- | ---: | ---: | --- |
| decode tok/s | 3.56 | **5.50** | +54% |
| `--minimax-profile-time` total | 226.2 | **163.5** | -63 ms |
| `cpu_ram` | 113.4 | 94.9 | fewer RAM experts |
| `ssd_wait` | 18.6 | **2.2** | |
| `cpu_ssd` | 9.4 | 0.9 | |
| `admit` | 11.9 | 1.2 | |
| ssd/token | 13.4 | **2.8** (43 MB) | the goal was 0 |
| ssd MB/token | 231 | 43 | |

The RAM tier sized itself to 100.0 GiB / 6753 experts (118.5 a layer), so 543 of 7296 experts still come off the
drive; `--memory-limit-mib` above ~101 GiB leaves under 10 GiB free on this 128 GB box, so 2.8 a token is the floor
for the 25-cold-layer pack as assembled.

**Blocked on the gate:** `--minimax-eval` (the G3 harness's path) fails on this file with *"norm weights look
un-baked (mean -8.6e11)"* - `decide_norm()`'s heuristic on `output_norm.weight` + `blk.0.attn_norm.weight`, both of
which are byte-identical to the source, and the same binary loads the same file and decodes correctly in the serve
path. `--minimax-allow-nonstandard` would pass it but that also tells the engine the norms are not +1-baked, which
is wrong for this pack, so it is not a fix. **Root cause found in round 4: the gate opened the pack while
`mix_layers.py` was still writing it** - the engine read a partial file. On the finished file the eval path reports
`INFO minimax-norm mean=1.264317`, the same number the D3 pack reports and the same one the serve path reports, and
`--minimax-eval` produces logits. Round 4 adds that line permanently: it is the engine's own proof that it resolved
`output_norm.weight` and `blk.0.attn_norm.weight` to the right bytes, and comparing it between `--serve` and
`--minimax-eval` is how a loader difference between the two paths would show up.

## The shared PC: `tools/minimax/pc_launch.py`

The PC runs one memory-hungry job at a time (`pc-engine.lock`, `E:\Jon's Projects\AI\Strata Fork Minimax\`). Two
out-of-memory crashes showed that a written rule is not enough, so this launcher makes the rule mechanical. Every
engine run - a bench, a gate, a g3 pass - goes through it:

```
python tools/minimax/pc_launch.py --owner dspark --est-ram-gb 86 -- \
    build/strata-minimax.exe --model <model.gguf> --serve ...
```

1. **The lock.** Plain ASCII, no BOM, `free` when unused, else `<owner> <pid> <iso-time>`; a bare name with no PID
   (an older task's hand-written lock) counts as held. It refuses when another owner holds it with a live PID, says so
   when the PID is stale and takes over, and takes it atomically (an `O_EXCL` sidecar `.lck` is the mutex, and the
   lock itself is written to a temp file and renamed). A stale sidecar older than a minute is recovered, so a
   crashed launcher cannot wedge the PC.
2. **The RAM floor.** Available RAM comes from `GlobalMemoryStatusEx` through ctypes (no PowerShell), and the run's
   need from `--est-ram-gb` or from the engine's own flags (`estimate_ram_gb`: pinned expert tier, dense host copy,
   KV, prompt workspace, margin - the breakdown is printed). It refuses to start when `available - need < 10 GB`.
3. **The watchdog.** Every 2 s, if available RAM falls under the floor, the child's whole process tree is stopped and
   the launcher exits 4 with the reading that triggered it.
4. **The release.** `try/finally` plus SIGINT/SIGTERM/SIGBREAK handlers put the lock back to `free`: a clean exit, a
   child crash, Ctrl-C/Ctrl-Break and a watchdog kill all release.

Exit codes: the child's own, `2` another owner holds the lock, `3` not enough RAM, `4` the watchdog stopped the run,
`5` the lock mutex timed out, `130` interrupted. stdout and stderr are the child's (the launcher's lines are prefixed
`[pc_launch]` on stderr) and stdin is inherited, so an engine's `GEN` protocol works unchanged.

`tools/minimax/test_pc_launch.py` covers the protocol with no engine: take/release (`free` is exactly four bytes),
the lock line's owner/pid/time and no BOM, a live owner refused, a stale PID taken over, a bare name refused, the RAM
refusal, the estimate from engine flags, the watchdog (a scripted `--ram-source seq:20,20,3` reading, so the decision
and the kill are exercised without eating the box), stdin passthrough, the child's exit code, two launchers where one
wins, and the interrupt release.

### Round 4: the loader proof, and the A1 gate says IQ1_M is too lossy

**The eval-path refusal was not a loader bug.** On the finished pack both entry points print the same line -
`INFO minimax-norm mean=1.264317 over output_norm+blk.0.attn_norm (12288 values)` - and it is the same number the D3
pack reports, so `--serve` and `--minimax-eval` resolve those tensors to identical bytes. The round-3 gate had opened
the pack while `mix_layers.py` was still writing it. That line is now printed always, for exactly this reason.

**The byte proof is mechanical now.** `--minimax-verify-tensors` prints `name bytes fnv1a-64` for every tensor the
engine resolved; `--minimax-verify-mod M --minimax-verify-rem R` hashes every M-th one, because a whole pass faults
the pack's 130 GB through the mapping and trips the launcher's RAM floor on a 128 GB box. `tools/minimax/verify_pack.py`
compares those against the donor packs (a mixed pack holds one tensor *name* from more than one donor, so it picks the
donor whose byte length matches what the engine read). Sample of the mixed pack, both a hot layer and a cold one:
**6/6 identical to a donor** (`output_norm`, `blk.0.attn_norm`, `blk.59.attn_norm`, `blk.0.attn_q`, and
`blk.35`/`blk.59` `ffn_gate_exps` - the cold ones read 528,482,304 B, the IQ1_M size, not D3's 698,351,616 B).

**G3, six short prompts, the mixed pack against D3 in the same session and flags** (`--vram-slots 0`, all-CPU, so
placement cannot be the difference; `--memory-limit-mib 96000`; 679 positions, two runs of three prompts):

| arm | positions | top-1 | KL64 |
| --- | ---: | ---: | ---: |
| D3 (default) | 679 | **58.47%** | 1.328 |
| **MIX, 25 of 57 MoE layers at IQ1_M (116.3 GB)** | 679 | **55.39%** | 1.453 |
| - first half (chinese_local_llm, code_lru, math_bakery) | 339 | 47.49% (D3 48.9%) | 1.986 |
| - second half (multi_turn_threads, prose_moe, tool_weather) | 340 | 63.24% (D3 67.9%) | 0.978 |

**The gate fails: -3.1 points of top-1 against the default, and 4.0 below the 59.4% bar.** IQ1_M on 44% of the MoE
layers - 40.5% of the profile's routing mass - is simply too lossy for this model, whatever it buys in speed (5.50
tok/s against 3.56, ssd/token 2.8 against 13.4). The arithmetic for the next attempt: the honest middle band is
**IQ2_XXS** (1,246 MB a layer against IQ1_M's 1,680 and D3's 2,322), so 25 IQ1_M + 25 IQ2_XXS would be 104.9 GB and
lose less than half of what IQ1_M loses at the same size. That needs the published IQ2_XXS tier downloaded (~119 GB
at this link's ~26 MB/s, so a job of its own), which is what round 4's remaining time did not cover.

### The copy-all rule (Strata 0.2.6 / Maya v1.1.27 `--mmap-experts auto`), ported

`init_tiers` now takes the whole tail when it fits with room to spare, instead of the profile's hottest part:

```
can = min(available - headroom, commit_room)          (commit_room = ullAvailPageFile on Windows)
if (every expert left off the GPUs) + 8 GiB <= can:  pin all of them, at load
```

Maya's rule is all-or-none on purpose - "a PART copied cost decode speed on 32 / 64 GB of RAM" - because a partial
tier still sends the rest to the drive every token while paying for the copies (their #339: +22% on 4 x V100 with
504 GB, ~2x on a 20 GB Ada, +23% on 4 x 3090 with 126 GB). The 8 GiB spare is also what keeps this engine's own RAM
floor satisfiable. It only fires when `--memory-limit-mib` is not given, so an operator's budget still wins.

### Round 7: the loader proof's mechanism, the link log, and the first sweep block

**Loader proof (`--minimax-verify-tensors`) does no hashing and touches no pages.** It prints, per resolved tensor,
`name shard offset bytes type`, then `--minimax-verify-exit` returns 0 straight away - so a proof costs seconds and a
few MB of RAM instead of faulting 130 GB through the mapping (which is what tripped the 10 GB floor in round 4).
`tools/minimax/verify_pack.py` takes those lines from **both** entry points, streams each range in 64 MiB buffered
reads and SHA-256s it, compares with the donor tensor of the same name *and length* (a mixed pack holds one name from
two donors), and cross-checks that `--serve` and `--minimax-eval` resolved every tensor to the same shard, offset and
length. Its modulo filter had an off-by-one (`at % mod != rem + 1`) that reported `total=0`; fixed, and the report also
has to add each shard's data start, since a split GGUF stores a tensor at `data_start(shard) + offset` while the
engine reports the offset inside its shard.

**Measured on a 13-shard pack, all 1188 tensors, both entry points** (`--minimax-verify-exit`, so each run is a few
seconds and a few MB):

```
     serve: 1188/1188 tensors identical to a donor
      eval: resolution against the first log: 0 tensors only here, 0 resolved differently
      eval: 1188/1188 tensors identical to a donor
PROOF PASS - SHA-256 of every tensor, streamed in 64 MiB reads, and every log resolving every tensor the same way
```

**`INFO minimax-link dev=N h2d=X GB/s pcie=genG xW`** at startup: the driver's own generation and width (NVML loaded
at run time, so nothing links against it) beside this process's measured host-to-device rate. CUDA reports what the
*card* can do, not what the slot gives it - device 0 measured 6.4 GB/s against device 1's 21.9 on this box - so a
split or a PCIe expert lane has to be priced on both.

**The sweep driver** (`tools/minimax/sweep_blocks.py`) swaps one block of consecutive MoE layers to a cold donor,
gates it with the same all-CPU harness, records loss-per-GB, and deletes the candidate pack after its run.
`mix_layers.py --cold-layers 36,37,38,39,40` names a block, which is how a sweep candidate is built without touching
the fraction/order heuristics.

First measured block, subset gate (chinese_local_llm + code_lru, 211 positions, `--memory-limit-mib 80000`):

| block | layers | GB saved | top-1 | KL64 | D3 on the same subset |
| --- | --- | ---: | ---: | ---: | --- |
| 3-7 | 5 | 3.44 | **43.13%** | 2.344 | 41.69% / 2.686 |

The earliest block is **+1.4 points** better than the baseline on that subset - free gigabytes, and the reason a
per-block sweep is the right instrument: *where* the bits go matters more than how many. The remaining blocks were
lost to the round's I/O contention (a concurrent IQ2_XXS download raised the page-cache floor until the watchdog
stopped two gates) and to the lock going to the other task; they are the first thing the next round re-runs.

`--memory-limit-mib 80000` is now the sweep's budget (77 GB free before the run, per the launcher's own line), and
the download is paused rather than run beside a gate.

### Round 8: the zero-copy PCIe lane (written, not yet measured)

Maya #476's design, ported: `--minimax-pcie-share f` takes `f` of each layer's RAM-tier experts and hands them to the
GPU expert path with the **device address of their pinned bytes** as the base, instead of a VRAM slot.

* `cu::host_register_mapped` registers the `PinnedArena` range with `cudaHostRegisterMapped | Portable` and returns
  `cudaHostGetDevicePointer`'s device view (`cuda_ops.cu`). A 100 GB range does not always register; a refusal leaves
  the lane off with `INFO minimax-pcie-lane off: the driver refused to map the … RAM tier`.
* `gpu_experts_begin` takes an optional `alt` vector aligned with `exps`: nullptr keeps the VRAM slot, anything else
  is read in place. The blob layout is already `[gate | up | down]` in the arena, the same as a VRAM slot, so no
  repacking.
* The token path marks the chosen routes out of the CPU pass and appends them to the GPU list, so the combine and the
  ordering are unchanged; `INFO minimax-stats … pcie=N` counts them per token.

Not measured yet: the share has to be set from the two links' measured rates (the startup `INFO minimax-link` line
gives them), and the whole-token A/B that keeps the lane only when it is >3% faster is still to be written.

### Round 9: D3's baseline on the sweep's own flags, the link log's first reading, and two bugs in --expert-source

**The baseline is measured with exactly the blocks' flags** (`--vram-slots 0 --memory-limit-mib 80000
--layer-split auto --cpu-threads 12`, same six prompts, same reference logits, same build):

| arm | prompt set | positions | top-1 | KL64 |
| --- | --- | ---: | ---: | ---: |
| D3 | subset (chinese_local_llm + code_lru) | 211 | 41.23% | **2.7021** |
| D3 | all six | 679 | 58.32% | **1.3332** |

That supersedes the 58.47%/1.328 taken at `--memory-limit-mib 104000`; the deltas a block is ranked by are against
these two numbers.

**The KL anomaly is not reproduced by the flag path yet, and the flag has two bugs.** `--expert-source 3-7=<IQ1_M>`
(the round-9 answer to "don't write a 100+ GB file per block") resolves an expert tensor from an alternate pack
through the same `GgufModel` lookup, and it reaches the model - but the run then dies with *"minimax SSD read: short
read at 5897672192"*: the **SSD tier** still resolves its file by shard index against the **base** pack's file list,
so a layer whose bytes live in the alternate pack is read from the wrong file. Until the SSD path is source-aware
(the next fix), a block gate has to run with those experts resident - `--memory-limit-mib` large enough to hold them,
which is exactly what the earlier file-based block did.

That also means the earlier file-based `3-7` row (KL64 2.344 against D3's 2.686) has not been re-derived, and per the
round-9 rule it is **not** used for ranking until it is.

**`INFO minimax-link` first reading, under load:**

```
INFO minimax-link dev=0 h2d=3.29 GB/s pcie=gen3 x4
INFO minimax-link dev=1 h2d=3.28 GB/s pcie=gen4 x16
```

NVML reports device 0 at **gen3 x4** - matching the ~3.5 GB/s ceiling the round-9 note derived - and device 1 at
gen4 x16, but *both* measure ~3.3 GB/s, so the probe is not separating them: a 32 MiB pageable copy per call is
probably limited by the copy engine's staging rather than by the link. The probe needs pinned source memory and a
larger transfer before its number is used to split work between the lanes.

### Round 10: the SSD tier is source-aware; the link probe is two-directional and sampled under load

**Every tier now opens the tensor's own file.** `--expert-source` folds each alternate pack's shards into ONE flat
`ssd_files` list (the base pack's shards, then each alternate's), and `try_must` maps a tensor's shard into that list
at resolution time. The SSD read, the prefetch and the pin-time fill therefore open the file the bytes actually live
in, with no per-tier knowledge of where a tensor came from. The bug that made the first attempt die with
*"minimax SSD read: short read"* was exactly this, and a second one (**an out-of-bounds read of the shard-base vector,
which is now filled at open time rather than by the later SSD block - an access violation right after `decide_norm`**)
came with it.

`tools/minimax/test_expert_source.py` forces layers 3-7 onto the SSD tier with a tiny `--memory-limit-mib`, runs the
base arm and the `--expert-source` arm, and checks (a) both arms run, (b) every tensor in both arms hashes to a donor,
(c) the block's expert tensors resolve to the donor's file.

**The link probe is two-directional and samples NVML from a second thread during the transfer.** An idle card drops
its link generation, so a reading taken between transfers describes the idle state: `INFO minimax-link` now prints
`h2d`, `d2h`, `pcie_idle` and `pcie_load` per card, where `pcie_load` is the highest generation/width the driver
reported while the 256 MiB pinned copies were running.

**Correction, in this doc as well as the code:** the 6.4 GB/s figure the split probe reported for device 0 was a
measurement error. Device 0's real rate is **3.3 GB/s**, which matches NVML's **gen3 x4** - and the earlier probe
(pageable source, default stream) read ~3.3 GB/s on *both* cards, which is the driver's staging rate and cannot tell
the two slots apart. Only the pinned 256 MiB probe's two directions are used from now on.

### The KL anomaly is real, and it is a quant-versus-quant difference

Round 8's file-based block read `--expert-source`-free and said block 3-7 was 0.34 KL64 *better* than D3; round 9
could not re-derive it because `--expert-source` was broken. It is now fixed and the pair was re-run **back to back in
one turn, same flags, same reference logits, same build** (subset of two prompts, 211 positions, `--vram-slots 0
--memory-limit-mib 80000 --layer-split auto --cpu-threads 12`, reference SHA-256 `4f25e1a7be140715` in both runs):

| arm | top-1 | KL64 |
| --- | ---: | ---: |
| D3 | 41.23% | 2.7021 |
| D3 with layers 3-7's experts from IQ1_M | **43.13%** | **2.3443** |
| delta | **+1.90 points** | **-0.3578** |

So the effect reproduces on an independent path (a pack built by copying whole layers, and an override that reads
those layers from another pack without writing a file), which rules out the file build as the cause. The leading
explanation is that it is genuine: the two packs were quantized from the same checkpoint with **different imatrices**,
and on these two prompts IQ1_M's is the better one, so "fewer bits" is not the same as "more error" here. Two
consequences, both of which the sweep has to respect:

* a block's KL delta is a statement about **that prompt set**, not about quantisation in general - hence the rule that
  the ranking is validated on all six prompts (and, when it lands, on the held-out set);
* a negative delta (a block that *helps*) is not evidence that more blocks will keep helping. The candidate has to be
  measured as a whole mix, never by adding block deltas up.

`tools/minimax/test_expert_source.py` is the check that the block's bytes really are the donor's: it forces every
layer onto the SSD tier with a tiny `--memory-limit-mib` and hashes what the engine resolved, per tensor, against the
donors. It has not been run yet - it is the first thing the next turn does, before the sweep uses any block number.

### The imatrix explanation is rejected: IQ1_M reconstructs these layers *worse*, not better

Engine-free, CPU only, no lock. Per tensor, two experts each, dequantized with `gguf.quants.dequantize` from both
packs and compared with the source checkpoint's NVFP4 weights (`models/MiniMax-M3.1-Flash`, decoded with the E2M1
table and the global-scale divisor).  `relF` is ‖q − s‖ / ‖s‖.

| layer | matrix | D3 type | IQ1_M type | relF D3 | relF IQ1_M |
| ---: | --- | --- | --- | ---: | ---: |
| 3 | gate/up | IQ2_XS | IQ1_M | **0.324 / 0.324** | 0.501 / 0.501 |
| 3 | down | IQ3_XXS | IQ2_XXS | **0.208** | 0.377 |
| 30 | gate/up | IQ2_XS | IQ1_M | **0.307 / 0.307** | 0.465 / 0.465 |
| 30 | down | IQ3_XXS | IQ2_XXS | **0.198** | 0.364 |
| 55 | gate/up | IQ2_XS | IQ1_M | **0.308 / 0.308** | 0.468 / 0.468 |
| 55 | down | IQ3_XXS | IQ2_XXS | **0.225** | 0.411 |

D3 is closer to the source on **every** tensor sampled, in the block (layers 3-7), the middle (30) and the tail (55),
by 1.5-1.8x. So "IQ1_M's imatrix fits these prompts better" does not explain the +1.90 points / −0.358 KL64 the block
gate measured - and the two packs do not even record a `quantize.imatrix.*` key, so which imatrix each used cannot be
checked from the files at all (both carry only `general.quantization_version` and `strata.indexer_quant`). The
imatrix-weighted column of the check fell back to uniform weights for these shapes (`in_sum2`'s expert axis did not
line up with the dequantized tensor's column axis), so it is not yet an independent number; the plain Frobenius
column is.

What that leaves: the block's *logits* gain is not a better weight reconstruction, so it is either error cancellation
against the FP32 reference on a 211-position subset or a property of those two prompts. The decisive test is the one
ordered next: block 3-7 on **all six** prompts against D3 in the same session.

### Round 11 status: the SSD fix works, the hash checker still mis-matches shards

* **The `--expert-source` SSD path works.** `test_expert_source.py` ran both all-SSD arms (`--memory-limit-mib 600`,
  every expert on the drive) to completion - `base: rc 0`, `alt: rc 0` - where the same shape of run previously died
  with *"minimax SSD read: short read"*. Layer 3-7's bytes are now read from the alternate pack.
* **`verify_pack.py` still fails on a multi-shard pack**, and the failure is in the checker, not the engine: a later
  shard's data block can start at a few KiB (shard 2 of the D3 pack at **5120 B**), the engine reports an **absolute**
  offset in the resolved file (confirmed against gguf-py: `blk.4.attn_k` at 53507072 in both), and the checker's
  file/donor matching then picks the wrong side for tensors whose *size* coincides across the two donor packs. It
  needs to match on (leaf, size) and fall back to the leaf's own donor, not the first donor whose length agrees.
* The 6-prompt block gate (block 3-7 against D3, same session) is the next turn's first job, then the sweep.

### The byte checker, fixed and given a negative control

Three faults, all in `verify_pack.py`, none in the engine:

1. **The flat file list double-counted the base pack.** The engine's shard index runs over ONE list - the base pack's
   shards, then each `--expert-source` pack's - and `--donor` usually repeats the base pack (it *is* a donor), so
   index 13 landed on the base pack's second shard instead of the first alternate pack's. The base pack is now
   dropped from the donor-shard list when building it.
2. **Donors were matched by byte length alone.** Two packs can give a tensor the same length, so the check hashed the
   wrong side. It now passes when the engine's bytes equal **any** length-matching donor's copy, which is the right
   rule for a mixed pack (each name comes from exactly one donor).
3. **The reported `leaf` is not the tensor's shard file** for a base tensor - it is the file the run was pointed at.
   The shard index alone names the file; `leaf` is kept as a cross-check only.

And the checker now has a **negative control** (`--selftest`): it copies the first log with **one tensor's offset
moved by 4096 bytes** and requires the comparison to fail. On the cropped logs used for this work it reads
`base: 5/6, 1 differ` / `selftest PASS (a moved offset was caught)`, i.e. the checker can distinguish a wrong byte
from a right one.

**Full N/N, both all-SSD arms, all 1188 tensors** (regenerated with this build, `--memory-limit-mib 600` so every
expert goes through the drive, `--vram-slots 0`):

```
      base: 1188/1188 tensors identical to a donor
       alt: resolution against the first log: 0 tensors only here, 15 resolved differently
       alt: 1188/1188 tensors identical to a donor
PROOF PASS - SHA-256 of every tensor, streamed in 64 MiB reads, and every log resolving every tensor the same way
```

The 15 are exactly the block: layers 3-7 x {gate, up, down}. `verify_pack.py` now takes `--same-config` (require
identical resolution, for a `--serve` against a `--minimax-eval` log) and `--expect-moved N` (require exactly N
tensors to move), because a base arm and an `--expert-source` arm are *supposed* to differ - comparing them as if they
were the same configuration was the last false failure.
