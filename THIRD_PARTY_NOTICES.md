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
