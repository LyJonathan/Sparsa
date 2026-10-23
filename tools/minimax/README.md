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
| VRAM | per-layer slots from the STRP profile (or compulsory first use) | grouped GPU kernel over the selected experts; IQ2_XS and Q8_0 use `iq_kernels.cu`'s dot products, F32/F16/BF16 have a direct kernel; swigluoai fused into gate/up |
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

| layer / prompt | attn_out rel | block ids | router top-4 | experts (matched routing) | layer_out rel |
|---|---|---|---|---|---|
| 0 / short (from the embedding) | 1.4e-3 | — (1 block) | — | — | 1.3e-3 |
| 3 / short (seeded) | 2.2e-3 | — | 97.4% | 2.1e-3 | 7.1e-4 |
| 40 / short (seeded) | 3.5e-3 | — | 96.5% | 4.4e-3 | 2.4e-3 |
| 40 / long (4093 tokens, sparse) | 1.3e-2 | 99.51% sets, 0.9992 Jaccard | 98.8% | 7.7e-3 | 1.2e-3 |

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

