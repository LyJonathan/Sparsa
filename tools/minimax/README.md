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
all-RAM and all-SSD were byte-identical (`0.0`) to the all-CPU path. IQ2_XS on the GPU tier measured max-abs
`1.7e-2`, KL `1.1e-5`, top-1 `1.000` against the CPU IQ2_XS path (GPU iq kernels quantize the activation with q8_1,
the CPU path with ggml's Q8_K). Q8_0 matched to max-abs `3.6e-7`.

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

## Real-weight parity (when the other branches have artifacts)

`Strata-MiniMax/_local/ref_out/` currently has only `ppl_text.txt`, no per-layer activation dump, so the
layer 0 / 3 / deep comparison was not run. The quant smoke file that is present is one MoE layer, not a
3-layer or 60-layer model:

```powershell
build\strata-minimax.exe --model "..\models\quant-smoke\smoke-l3-q8.gguf" --minimax-inspect
```

That file is `minimax-m3`, 1 layer, embd 6144, vocab 200064, leading dense 0, routed experts Q8_0, output
Q6_K. A 4-token teacher-forced run on the CPU path (`--minimax-dense-layers-gpu 0 --max-context 32`) wrote
finite fp32 logits (shape `[1, 200064]`, max abs about 2.85). It is not a parity result.

Once a truncated Q8_0 GGUF and a reference dump exist:

```powershell
build\strata-minimax.exe --model trunc.gguf --minimax-allow-nonstandard `
  --minimax-eval tokens.txt --minimax-dump-logits logits.npy --minimax-last-k 4 `
  --max-context 512 --minimax-dense-layers-gpu 1 --cpu-threads 8
```

Compare `logits.npy` to the reference dump for layers 0, 3, and one deep layer. `--minimax-allow-nonstandard`
is only for a file that fails the M3.1 fingerprint (a truncated layer count). A full M3.1 GGUF does not need it.
Drop `--minimax-dense-layers-gpu` (or pass `-1`) to place as many leading dense layers as the 12 GB card allows
after the KV cache; pass `0` for the strict CPU path.
