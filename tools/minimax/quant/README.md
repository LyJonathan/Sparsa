# strata-mm-quant

Streams MiniMax-M3.1-Flash safetensors into a `minimax-m3` GGUF. One tensor is dequantized to f32 in memory and quantized with ggml. The tool does not write a full-model BF16 or F32 file.

ggml comes from the patched llama.cpp tree. That tree has the IQ quantizer and `GGML_TYPE_NVFP4`. `third_party/ggml` in this repo is only the headers the native expert kernels include.

`STRATA_BUILD_MM_QUANT` in the root `CMakeLists.txt` defaults to OFF, so a normal strata configure does not build this tool. The tested path is the standalone project in this directory.

## Windows

This tree lives under a directory name that contains an apostrophe. nvcc rejects that path. This tool does not call nvcc. `build.ps1` and `build-msvc.cmd` still map the worktree to `Q:` and the llama.cpp tree to `R:` so a later CUDA step can reuse the letters. Leave `L:` alone. The baseline llama.cpp build already uses it.

```
powershell -ExecutionPolicy Bypass -File tools\minimax\quant\build.ps1
```

`build-msvc.cmd` is the same build without PowerShell. Both write `tools/minimax/quant/build-msvc/strata-mm-quant.exe` with MSVC and Ninja.

Do not configure with a one-line `cmd /c "call vcvars && set PATH=...%PATH%"`. cmd expands `%PATH%` before `vcvars64.bat` runs, `cl` disappears, and CMake picks MSYS `cc` when it is earlier on `PATH`. The `.cmd` file expands `PATH` after `vcvars`.

A hand configure needs `-DSTRATA_LLAMA_CPP_DIR`. Point it at the M3.1 llama.cpp tree, or at `R:/` after the subst. The relative candidate `../../../_baseline/llama.cpp` does not exist from this worktree.

Verified here with MSVC 19.51 and, separately, with MSYS2 UCRT gcc 15.2. The gcc configure happened only because `cc` was on `PATH` before the `.cmd` fix. Use the `.cmd` when you want MSVC.

## Linux

```
cmake -S tools/minimax/quant -B build-mm-quant -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_LLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build-mm-quant
```

The llama.cpp tree must contain ggml with the IQ types and `GGML_TYPE_NVFP4`. The tool links `ggml-base` only. CUDA, OpenMP, and the CPU backend are forced off for this target. `ggml_quantize_chunk` is in `ggml-base`. The executable links `Threads::Threads` on Linux.

This Windows machine did not run the Linux configure.

## Run

```
strata-mm-quant --model DIR --recipe MM-IQ2_XS --imatrix real.gguf --out out.gguf
```

`--recipe` accepts a file path or a name looked up under `recipes/` next to the executable.

IQ1_S, IQ1_M, IQ2_XXS, and IQ2_XS stop if `--imatrix` is missing. ggml asserts for IQ1_S, IQ2_XXS, and IQ2_XS. This tool also requires an imatrix for IQ1_M.

`--write-test-imatrix FILE` writes a uniform importance matrix. Its dataset string is exactly `TEST-ONLY uniform importance; not a real activation imatrix`. Loading that file prints a warning. Do not publish a GGUF built from it.

`--max-layers N` sets `block_count` to N and keeps source layers 0 through N-1. Every one of those layers must already be on disk.

`--layers 3,4` remaps those source layers onto `blk.0..` and sets `block_count` to the length of the list. It cannot be combined with `--max-layers`.

`--layers A-B` keeps the source names `blk.A` .. `blk.B`. Metadata `block_count` and `leading_dense_block_count` stay the full model (60 and 3), so the file is one piece of the real model rather than a renumbered smoke model.

`--emit A-B` keeps a remapped plan and writes only those output layers. It is for the split-identity test. Do not combine it with `--layers A-B`.

`--split-part K --split-count N` writes a final llama.cpp split file. The name is `<prefix>-%05d-of-%05d.gguf` with the part number one-based (`K+1`) and `split.no` zero-based. Part 0 carries the KV metadata, including the vocabulary. Later parts carry only `split.no`, `split.count`, and `split.tensors.count`. `--split-tensors` overrides the total; the default counts a full plan so every machine computes the same total from the config. Globals are written on part 0 unless `--include-globals` or `--no-globals` says otherwise.

`--threads N` defaults to every logical processor. IQ2_XS row quant is already split across those threads.

`--placeholder-missing` stands in only for a missing `token_embd`, `output`, and `output_norm`. The embedding value is the constant 0.02. An all-zero Q8 row makes RMSNorm produce NaN. The output-norm stand-in is 1, the baked identity of a missing zero norm. The tool never invents layer weights.

`--override experts=TYPE` sets `experts`, `ffn_down_exps`, and `hot_experts` to the same type.

`--dense-type Q6_K` or `Q4_K` changes attention, indexer, dense MLP, and shared experts together. The default for those four is Q8_0.

`--nvfp4-check` reports the sampled expert matrices of the first complete MoE layer: the per-expert `weight_global_scale`, whether folding it into the E4M3 scale byte would be exact, and that every scale byte is a non-negative E4M3 value. Exit code 0 means the report finished. MM-NVFP4 does not depend on the fold: it stores the divisor as a side tensor.

`--dump-dequant layers.N.experts.E.w1 out.npy` writes one dequantized matrix as float32, C order, shape `(n_out, n_in)`.

`--bench` times dequant plus quant on the first MoE layer that is present. The default sample is 8 experts of `w1`, in the recipe's expert type.

`--mmproj OUT.gguf` runs llama.cpp `convert_hf_to_gguf.py --mmproj --outtype f16` through `mmproj.py`. Pass `--llama-cpp` or set `STRATA_LLAMA_CPP_DIR`. If `model-00013-of-00048.safetensors` or `model-00048-of-00048.safetensors` is absent, the tool prints `SKIP mmproj` and still returns 0. It does not invent a vision file. Vision weights in the checkpoint are BF16.

`--dry-run` prints `ggml_row_size` totals for the full 60-layer model and for the selected layers. It does not read shard bytes.

The process refuses to start a GGUF when free space on the output drive is below the tensor bytes plus 8 GiB. A failed write deletes the partial file.

## Recipes

| file | experts | ffn_down_exps | hot_experts | imatrix |
| --- | --- | --- | --- | --- |
| MM-IQ1_S | IQ1_S | IQ1_M | IQ2_XXS | yes |
| MM-IQ1_M | IQ1_M | IQ2_XXS | IQ2_XS | yes |
| MM-IQ2_XXS | IQ2_XXS | IQ2_XS | IQ2_S | yes |
| MM-IQ2_XS | IQ2_XS | IQ2_XS | IQ2_S | yes |
| MM-IQ2_S | IQ2_S | IQ2_S | IQ2_S | no |
| MM-IQ3_XXS | IQ3_XXS | IQ3_XXS | IQ3_XXS | no |
| MM-IQ3_S | IQ3_S | IQ3_S | IQ3_S | no |
| MM-IQ4_XS | IQ4_XS | IQ4_XS | IQ4_XS | no |
| MM-NVFP4 | NVFP4 | NVFP4 | NVFP4 | no |

Router gate and `exp_probs_b` stay F32. Norms stay F32. `token_embd` is Q8_0. `output.weight` is Q6_K. MM-NVFP4 additionally writes three f32 side tensors per MoE layer (see below); they are 512 bytes each.

IQ1_S, IQ1_M, and IQ2_XXS step the whole `ffn_down_exps` tensor up one type. A GGUF tensor has one ggml type, so mixed types inside one stacked expert tensor are not possible. MM-IQ2_XS keeps IQ2_XS on gate, up, and down.

An expert-usage file looks like `{"mode":"record","layers":{"3":[12,40]}}`. `record` stores the ids and does not change types. `promote-layer` sets that source layer's whole `ffn_gate_exps` and `ffn_up_exps` to `hot_experts`. `--hot-expert-mode` overrides the file.

The resolved types are stored in the GGUF key `strata.recipe`.

Full 60-layer tensor bytes from `ggml_row_size`, decimal GB, with the default Q8_0 dense and attention side:

| recipe | total GB | GiB |
| --- | --- | --- |
| MM-IQ1_S | 97.632 | 90.927 |
| MM-IQ1_M | 109.467 | 101.949 |
| MM-IQ2_XXS | 124.528 | 115.976 |
| MM-IQ2_XS | 133.135 | 123.992 |
| MM-IQ2_S | 146.045 | 136.015 |
| MM-IQ3_XXS | 171.865 | 160.062 |
| MM-IQ3_S | 191.230 | 178.097 |
| MM-IQ4_XS | 233.188 | 217.173 |
| MM-NVFP4 | 246.098 | 229.197 |

MM-NVFP4 is the exact copy of the released expert weights. The row above is `ggml_row_size`, so it includes the side tensors' 87 KB only in the last digit. See below.

## NVFP4

MM-NVFP4 stores the routed experts as `GGML_TYPE_NVFP4` (type id 40, `block_nvfp4`, 36 bytes per 64 values, 4.5 bpw) and is **lossless**: the dequantized matrix is the source checkpoint's dequantized matrix, bit for bit in fp32.

`block_nvfp4` has no global-scale field, and the checkpoint's `weight_global_scale` is a **divisor** (spec A.3: `W = e2m1 * e4m3(scale) / G`). Folding `e4m3(scale)/G` into a new E4M3 byte is not exact: on layer 3 expert 0 `w1`, `G = 15123.6923828125`, scale byte 113 decodes to 144, the quotient is 0.00952148438 and the nearest E4M3 value is 0.009765625 - a **+2.6% gain error on the whole matrix**. So the tool does not fold. It writes:

* `blk.N.ffn_{gate,up,down}_exps.weight` - the packed nibbles remapped from the checkpoint's consecutive pair (low nibble = even input index) to ggml's half-interleaved layout (in each 16, low nibble = element j, high nibble = element j+8), and the checkpoint's E4M3 scale bytes **copied verbatim**.
* `blk.N.ffn_{gate,up,down}_exps.scale` - F32 `[n_expert]`, one `weight_global_scale` per expert, in expert order. That is 512 bytes per tensor, 87 KB for the whole model.

The engine divides by `scale[e]` exactly once per projection (it keeps `1/G` and multiplies). `G` really is per expert: on layer 40 `w2` the 128 values range from 13901.6 to 15910.5, so a single tensor-wide value would be wrong.

`--nvfp4-check` reports the sampled expert matrices, whether the fold would be exact (it is not, and no longer matters) and whether every scale byte is a non-negative E4M3 value. Exit code 0 means the report finished.

```
strata-mm-quant --model DIR --recipe MM-NVFP4 --max-layers 4 --threads 12 --out test.gguf
python tools/minimax/quant/verify_nvfp4_gguf.py --gguf test.gguf --model DIR
```

`verify_nvfp4_gguf.py` parses the GGUF blocks itself (no ggml, no gguf-py), dequantizes them with the side divisor, dequantizes the source safetensors with spec A, and requires **max |diff| == 0** over every expert matrix of every expert tensor in the file. On the 1-layer layer-40 and the 4-layer test files it printed `max |GGUF dequant - spec A dequant| = 0.0` over 384 matrices / 7.25 G elements per file.

### Full run

```
strata-mm-quant --model /path/to/MiniMax-M3.1-Flash --recipe MM-NVFP4 \
    --threads 12 --out MiniMax-M3.1-Flash-NVFP4-4.50bpw-00001-of-00013.gguf \
    --split-part 0 --split-count 13
```

Output is 246.098 GB of tensor bytes, input ~232 GB of packed NVFP4 plus the MXFP8/BF16 side. The 1.65 TB
figure quoted for the experts elsewhere is their f32 logical size; the checkpoint is already 4.5 bpw, so the
relayout reads about as many bytes as it writes. Measured here (Ryzen 9 5900X, 24 threads, `--threads 12`,
all 48 shards on the local 4 TB NVMe, 2026-10-24): a 4-layer test file (7.648 GB, one MoE layer) took 54.0 s and
a 1-layer file (6.572 GB, 2.3 GB of which is `token_embd`/`output`) took 51.8 s. The expert tensors alone run at
0.25-0.30 GB/s of output (1.359 GB in 4.5-5.5 s), so the full run is about 15-20 minutes of one machine, bounded
by the single-threaded nibble remap and the write, not by the disk (a raw read of the shards measured 2 GB/s).

Attention, indexer, dense MLP and the shared expert are MXFP8 in the checkpoint. The engine has no MXFP8 ggml type, so they are written as **Q8_0**. `q8_dense_error.py` measures the extra error of that choice on real tensors: relative L2 5.49e-3 to 5.74e-3 (median 5.56e-3) over six tensors (layer 0 q/o/dense gate, layer 3 shared gate/down, layer 40 index q). `output.weight` is Q6_K and `token_embd` Q8_0, as in every recipe.

### Engine support

CPU: `ggml-cpu`'s `ggml_vec_dot_nvfp4_q8_0` (Q8_0 activations), then the side scale is applied to the gate/up output before swigluoai and to the down output. GPU (P3 tiers): `minimax_moe_gu` / `minimax_moe_down` take optional device `f32` scale arrays and apply them in the kernel, once.

`tools/minimax/compare_nvfp4_parity.py` compares the engine's per-layer output with the reference dumps; `strata-minimax --minimax-dump-dir DIR --minimax-dump-layers 0,3,40` writes `DIR/L<layer>.npy`, and `--minimax-seed-hidden FILE.npy` replays one layer from a reference input (so a 1-layer file can be compared against layer 40 of a 60-layer run).

## Metadata

`general.architecture` is `minimax-m3`. `attention.indexer.first_layer` is written as 0 even for a one-layer remap. The loader throws if that value is greater than `block_count`, and a missing key defaults to the dense-lead count. Indexer `q` and `k` tensors are written for every output layer.

`attention.key_length` and `attention.value_length` are 128. Without them the loader uses `embedding_length / head_count`, which is 96, and the Q tensor shape does not match. `rope.freq_base` is 10000000. `rope.dimension_count` is 64. `feed_forward_length` is the dense width 12288. `expert_feed_forward_length` is 3072. `expert_count` is 128 and `expert_used_count` is 4 even when the selected layers are all dense. `expert_weights_scale` is 2 and `expert_weights_norm` is true, matching the converter. Gating is sigmoid, enum value 2.

Norm weights get +1 baked in when `use_gemma_norm` is true, which it is for this checkpoint.

`minimax-m3.swiglu_alpha` is 1.702 and `minimax-m3.swiglu_limit` is 7.0. The patched graph hardcodes swiglu-oai and does not read those keys.

The tokenizer is gpt2 with pre-type `minimax-m2`. When `tokenizer.json` is in the checkpoint, tokens, token types, and merges are taken from it. Merge pairs that are two strings encode a space as U+0120, matching llama.cpp. Holes are type UNUSED and are spelled `[PAD<id>]`. Without `tokenizer.json`, ids below the added-token range stay the placeholder spelling `t` plus six digits, and `tokenizer.ggml.merges` is an empty array. Added-token spellings come from `added_tokens.json` either way. bos `]~b]` is id 200019 and eos `[e~[` is id 200020. Both are token type CONTROL. Other added tokens are USER_DEFINED. `add_bos_token` is false. `tokenizer.chat_template` is copied from `chat_template.jinja` when that file is present. Vocab size is 200064, matching the embedding rows. A split part other than part 0 does not repeat the vocabulary.

Indexer projections are quantized with the recipe indexer type, default Q8_0. llama.cpp's converter forces `.indexer.` to F32. The difference is recorded in `strata.indexer_quant`.

## What ran on this PC

Ryzen 9 5900X, 24 threads. Shards on disk at the time started at `model-00002` and did not include shard 1 or shard 48, so the smoke model is source layer 3 remapped to `blk.0`, with placeholder embed, output, and output norm.

`strata-mm-quant-dequant-test` passed under both gcc and MSVC. It checks signed E4M3, MXFP8, BF16, the Gemma +1 bake, the nibble remap against `dequantize_row_nvfp4` at G=1, `ggml_validate_row_data` on that block, and the refusal for G=3 and for the real scale pair above.

IQ2_XS bench on layer-3 `w1`, uniform test imatrix, logical f32:

| threads | sample | time | rate |
| --- | --- | --- | --- |
| 1 | 2 experts, 0.151 GB | 28.68 s | 0.005 GB/s |
| 12 | 2 experts, 0.151 GB | 5.61 s | 0.027 GB/s |
| 12 | 8 experts, 0.604 GB | 14.22 s | 0.042 GB/s |
| 24 | 8 experts, 0.604 GB | 11.43 s | 0.053 GB/s |

One thread to twelve is 5.1× on the short sample, and twelve to twenty-four is 1.26× on the 8-expert sample. Rows are split across threads. The 24-thread rate is the one used for the PC and for rig-2. rig-3 measured 0.050 GB/s at 16 threads (8 experts, 12.13 s). `run_tier.py --plan` has the per-machine numbers. Routed experts are 413122166784 params, 1.652 TB logical f32. At 0.053 GB/s the expert matrices alone take about 521 minutes on one 5900X. One attention `q_proj`, MXFP8 to Q8_0, was about 3-4 GB/s at 24 threads. The non-expert tensors are about a minute for the whole model.

## MM-IQ2_XS across machines

`run_tier.py --plan` prints the 13-part assignment. `run_tier.sh` and `run_tier.ps1` call it. Each machine writes its own final `llama-gguf-split` parts. `fetch_shards.py` pulls the safetensors for a layer range from `MiniMaxAI/MiniMax-M3.1-Flash` with urllib. Rigs delete a shard after the parts that need it are written. The PC keeps its local checkpoint. No full-model run starts without an imatrix and an explicit part.

The Q8_0 smoke GGUF is 10.196 GB and was written in 19.5 s. It is not in git. `llama-cli -m smoke-l3-q8.gguf -p "]~b]" -n 8 -c 128 -fa on -ngl 0 --ignore-eos -st` loaded the file, generated 8 tokens, and exited 0. The printed pieces are the placeholder spellings `t######` because the real BPE merges are not available yet. Token quality on a 1-layer model is not a quality check.

No full-size tier file was written.
