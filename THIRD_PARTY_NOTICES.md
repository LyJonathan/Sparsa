# Third-party notices

Sparsa is a fork of Strata and is MIT (see `LICENSE`). This file lists the code the tree carries from
other projects, with the notice each one requires.

The model files are not part of this repository. MiniMax-M3.1-Flash and its quantized GGUF copies are
under the [MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE)
(commercial use requires displaying "Built with MiniMax M3.1-Flash" and notifying api@minimax.io; the
licence's prohibited uses apply). The GGUF copies on Hugging Face repeat that licence.

## ggml and llama.cpp

`third_party/ggml/ggml-common.h`, `third_party/ggml/ggml-common-tq.h` and the ggml built from the
pinned llama.cpp checkout (`third_party/ggml/VERSION.txt`) are from ggml/llama.cpp. The MiniMax
engine's CUDA helpers (`src/kernels/cuda/native_mmvq.cu`, `src/kernels/cuda/iq_kernels.cu`) use
llama.cpp's CUDA MMVQ/MMQ kernels and dequantizers.

```
MIT License

Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023-2024 The llama.cpp authors
```

## stb_image (public domain)

`src/minimax/third_party/stb_image.h` is stb_image v2.30 by Sean Barrett and contributors, taken from
llama.cpp's `vendor/stb`. It decodes JPEG/PNG/BMP/GIF for `--minimax-vision`. Public domain, no conditions.

## MiniMax vision graph

`src/minimax/vision.cpp`, `include/strata/minimax/vision.hpp` and `tools/minimax/convert_mmproj.py` follow
llama.cpp's `conversion/minimax.py` and `tools/mtmd/models/minimax-m3.cpp` (MIT, Copyright (c) 2023-2026 the
llama.cpp authors) for the mmproj tensor layout, the q/k permutation and the vision graph; the arithmetic is
transformers' `MiniMaxM3VLVisionModel` and `MiniMaxM3VLMultiModalProjector`. The engine's rotation differs from
llama.cpp's current graph by axis: the reference is 3-D RoPE with the temporal axis at zero, so H is the
(26+i, 39+i) pair and W is (52+i, 65+i).

## ik_llama.cpp (Iwan Kawrakow)

`include/strata/kernels/iqk_types.hpp`, `include/strata/kernels/cpu/iqk_quants.hpp` and
`src/kernels/cpu/iqk_quants.cpp` port the IQ2_KT, IQ3_KT, IQ3_K, IQ4_KS and IQ2_KL formats of
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), commit `804ee29c28a5`. What is ported is the trellis
codebook and the cluster search the trellis quantizers build, the five quantization row functions (with the
AVX2 evaluation order of `find_best_scale_avx2` / `evaluate_block_iq4k`, so the bytes match ik's), the
dequantizers, and the q8_1 dot products of `ggml-cuda/template-instances/mmvq-instance-iq*` into
`src/kernels/cuda/iq_kernels.cu` (`vec_dot_iq2_kt_q8_1` and friends, plus the `dq_*` dequantizers from
`ggml-cuda/convert.cu`); the type ids are Strata's own (see `include/strata/kernels/iqk_types.hpp` for why) and
ik's `iqk_mul_mat` tile kernels are not ported. `tools/minimax/quant/src/test_ikq.cpp` compares the port against
ik's own output, byte for byte, on 24 cases (six per format).

```
MIT License

Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023-2024 The llama.cpp authors
Copyright (c) 2024-2025 The ik_llama.cpp authors
```

## agentionai llama.cpp (trellis types, TQ2_T / TQK6 / TQK7)

`third_party/ggml-tq/` holds the trellis headers of the agentionai ("Gyro") llama.cpp fork: `tq.cuh`
(derived from the fork's `ggml/src/ggml-cuda/tq.cuh`) and `tq-mmq-tiles.cuh` (its
`mmq-load-tiles.cuh` trellis loader, unchanged), which `cmake/strata_ggml_tq_overlay.cmake` adds to a
build tree. The trellis codebooks and the Hadamard rotation used by the Gyro expert types are in
`third_party/ggml/ggml-common-tq.h` and `src/kernels/cuda/tq_lut_i8.cuh`. The upstream directory had no
licence file; the files' own headers name their origin, and the licence is MIT
(`third_party/ggml-tq/README.md` carries the text).

```
MIT License

Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023-2024 The llama.cpp authors
Copyright (c) 2026 Laurent Zuijdwijk
Copyright (c) 2026 agentionai
```

## Upstream Strata

Sparsa is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata); everything outside
`src/minimax/`, `include/strata/minimax/`, `tools/minimax/`, `serve/minimax*.py|jinja` and `sparsa.py`
comes from it, including the Python server, the Terminal and the desktop app.

```
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors
```

## Ideas and measurements (no code copied)

The MiniMax engine's later work compares itself against two sibling engines and follows their ideas:
the CPU pool's spin window and the RAM tier's size-from-the-box rule from
[Project Maya](https://github.com/mw00/project-maya) (Peasant Smith and contributors, MIT), and the
selection bias toward resident experts from [Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT).
The comments in `src/minimax/engine.cpp` name the measured rows each came from. No source file from
either project is copied into this tree; they are MIT, so a copy would be compatible, but the credit
is to the idea and the number.

The DeepSeek server front end's tokenizer handling (Strata-DS, Jackwwg83, MIT) is the model for
`serve/minimax.py`'s; the dynamic quant mix of the low tiers follows Unsloth's UD quants.

## MiniMax chat template

`serve/minimax_chat_template.jinja` and `serve/chat_template.jinja` are the chat templates that ship
with the models (MiniMax-M3.1-Flash and Qwen3.8-Flash-Next); they are model data under the models' own
licences, included so the server renders prompts the way the model was trained to read them.
