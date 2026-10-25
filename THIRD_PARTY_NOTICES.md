# Third-party notices

Strata itself is MIT (see `LICENSE`). This file lists the code and data the tree carries from other projects,
with the notice each one requires. Everything below is MIT, so the notices travel with the copy.

## ggml and llama.cpp

`third_party/ggml/ggml-common.h`, `third_party/ggml/ggml-common-tq.h` and the ggml built from the pinned
llama.cpp checkout (`third_party/ggml/VERSION.txt`) are from ggml/llama.cpp.

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

## Upstream Strata

Strata is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).

```
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors
```

## Gyro / agentionai llama.cpp (trellis types, TQ2_T / TQK6 / TQK7)

The trellis expert types, their block headers and the Hadamard rotation come from the agentionai ("Gyro")
llama.cpp fork, through `feat/minimax-trellis`.

```
MIT License

Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023-2024 The llama.cpp authors
Copyright (c) 2026 agentionai
```

## Project Maya (DSpark) and MiniMax-M3.1-Flash's DSpark draft head

`src/minimax/dspark.cpp`, `src/minimax/dspark.hpp`, `tools/minimax/dspark_pack.py` and the DSpark part of
`src/minimax/cuda_ops.cu` implement MiniMax-M3.1-Flash's own DSpark draft head (the checkpoint's
`dspark/config.json` and `model-dspark-00001.safetensors`) for the MiniMax engine. The reading of the checkpoint -
what the block is (a context projection from five of the target's layers into a per-layer K/V ring, a block of
mask rows, a rank-256 Markov head, a confidence head), how the block rows are positioned and how the head's argmax
is walked left to right - follows

  * [project-maya](https://github.com/mw00/project-maya) `tools/ds4/dspark_ref.py` and `src/ds4/ds4_model.cu`
    (MIT, Copyright (c) 2026 the project-maya authors), the DSpark drafter for DeepSeek-V4.1-Flash, which in turn
    follows DeepSeek's `inference/model.py` (MIT, Copyright (c) 2023-2026 DeepSeek); and
  * Strata's own DFlash2 drafter (`src/core/dflash2.cpp`, `src/kernels/cuda/dflash2.cu`, incoai's
    Qwen3.8-Flash-Next-DFlash2, Apache-2.0), whose block/ring/selector design this one mirrors so the two families
    behave the same way.

The window gating (one draft at least, up to four while the product of the confidence scores holds at 0.9), the
two-GPU verify-window shape and the speed-calibration gate (an 8k-token document, ~1,500 tokens, keep the drafter
only above +5%) are project-maya's `docs/SETTINGS.md` ("Drafting with DFlash2"). No Maya source file was copied:
the drafter here is written against Strata's own kernels (`native_mmvq`, `head_norm_rope`, `quantize_q8_1_rows`)
and its own GGUF pack format.

The 4-bit packs follow llama.cpp's `quantize_row_q4_0_ref` / `quantize_row_q5_0_ref` block layouts
(`third_party/ggml/VERSION.txt`), verified block for block by `tools/minimax/test_dspark.py`.
