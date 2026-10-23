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

`--layers 3` remaps source layer 3 onto `blk.0` and sets `leading_dense_block_count` to 0. It cannot be combined with `--max-layers`.

`--placeholder-missing` stands in only for a missing `token_embd`, `output`, and `output_norm`. The embedding value is the constant 0.02. An all-zero Q8 row makes RMSNorm produce NaN. The output-norm stand-in is 1, the baked identity of a missing zero norm. The tool never invents layer weights.

`--override experts=TYPE` sets `experts`, `ffn_down_exps`, and `hot_experts` to the same type.

`--dense-type Q6_K` or `Q4_K` changes attention, indexer, dense MLP, and shared experts together. The default for those four is Q8_0.

`--nvfp4-check` reports whether expert scales fold into `GGML_TYPE_NVFP4`. Exit code 0 means the report finished, including when the fold is impossible.

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

Router gate and `exp_probs_b` stay F32. Norms stay F32. `token_embd` is Q8_0. `output.weight` is Q6_K.

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

MM-NVFP4 is listed so the size is known. This checkpoint cannot be written as that type. See below.

## NVFP4

`GGML_TYPE_NVFP4` exists. Type id 40. `block_nvfp4` is 36 bytes, QK 64, four unsigned E4M3 scales, then 32 bytes of nibbles. Inside each group of 16, the low nibble is element j and the high nibble is element j+8. The checkpoint packs consecutive pairs along the input, low nibble even, high nibble odd. A raw byte copy is the wrong tensor.

`block_nvfp4` has no global-scale field. A lossless copy also has to fold `e4m3(scale) / weight_global_scale` into an unsigned E4M3 byte. On layer 3 expert 0 `w1`, `weight_global_scale` is 15123.6923828125 and scale byte 113 decodes to 144. The quotient is 0.00952148438. The nearest E4M3 value is byte 5, 0.009765625. The fold is not exact, so the converter refuses the tier instead of inventing a side channel. The check runs on every selected expert before the output file is created.

`--nvfp4-check` on the shards present on 2026-10-23 printed that same layer-3 pair and stopped the lossless claim there.

## Metadata

`general.architecture` is `minimax-m3`. `attention.indexer.first_layer` is written as 0 even for a one-layer remap. The loader throws if that value is greater than `block_count`, and a missing key defaults to the dense-lead count. Indexer `q` and `k` tensors are written for every output layer.

`attention.key_length` and `attention.value_length` are 128. Without them the loader uses `embedding_length / head_count`, which is 96, and the Q tensor shape does not match. `rope.freq_base` is 10000000. `rope.dimension_count` is 64. `feed_forward_length` is the dense width 12288. `expert_feed_forward_length` is 3072. `expert_count` is 128 and `expert_used_count` is 4 even when the selected layers are all dense. `expert_weights_scale` is 2 and `expert_weights_norm` is true, matching the converter. Gating is sigmoid, enum value 2.

Norm weights get +1 baked in when `use_gemma_norm` is true, which it is for this checkpoint.

`minimax-m3.swiglu_alpha` is 1.702 and `minimax-m3.swiglu_limit` is 7.0. The patched graph hardcodes swiglu-oai and does not read those keys.

The tokenizer is gpt2 with pre-type `minimax-m2`. `tokenizer.ggml.merges` is an empty string array, which the loader requires. `tokenizer.json` was not in the checkpoint during this work, so token ids below the added-token range are the placeholder spelling `t` plus six digits. Real added-token spellings come from `added_tokens.json`. bos `]~b]` is id 200019 and eos `[e~[` is id 200020. Both are token type CONTROL. Other added tokens are USER_DEFINED. `add_bos_token` is false. `tokenizer.chat_template` is copied from `chat_template.jinja` when that file is present. Vocab size is 200064, matching the embedding rows.

Indexer projections are quantized with the recipe indexer type, default Q8_0. llama.cpp's converter forces `.indexer.` to F32. The difference is recorded in `strata.indexer_quant`.

## What ran on this PC

Ryzen 9 5900X, 24 threads. Shards on disk at the time started at `model-00002` and did not include shard 1 or shard 48, so the smoke model is source layer 3 remapped to `blk.0`, with placeholder embed, output, and output norm.

`strata-mm-quant-dequant-test` passed under both gcc and MSVC. It checks signed E4M3, MXFP8, BF16, the Gemma +1 bake, the nibble remap against `dequantize_row_nvfp4` at G=1, `ggml_validate_row_data` on that block, and the refusal for G=3 and for the real scale pair above.

IQ2_XS bench, 8 experts of layer-3 `w1`, uniform test imatrix: 0.604 GB of logical f32 in 10.52 s, 0.057 GB/s. Routed experts are 413122166784 params, 1.652 TB logical f32. At that rate the expert matrices alone take about 480 minutes. One attention `q_proj`, MXFP8 to Q8_0, ran at 2.994 GB/s logical f32. Applying the IQ2_XS rate to every tensor overestimates the full file. The non-expert tensors are about a minute.

The Q8_0 smoke GGUF is 10.196 GB and was written in 19.5 s. It is not in git. `llama-cli -m smoke-l3-q8.gguf -p "]~b]" -n 8 -c 128 -fa on -ngl 0 --ignore-eos -st` loaded the file, generated 8 tokens, and exited 0. The printed pieces are the placeholder spellings `t######` because the real BPE merges are not available yet. Token quality on a 1-layer model is not a quality check.

No full-size tier file was written.
