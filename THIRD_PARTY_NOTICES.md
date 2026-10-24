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
`src/kernels/cpu/iqk_quants.cpp` port the IQ2_KT and IQ3_KT formats of
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), commit `804ee29c28a5`. What is ported is the trellis
codebook, the cluster search the quantizer builds, the two quantization row functions and the dequantizers;
the type ids are Strata's own (see `include/strata/kernels/iqk_types.hpp` for why) and ik's `iqk_mul_mat` tile
kernels are not ported. `tools/minimax/quant/src/test_ikq.cpp` compares the port against ik's own output, byte
for byte.

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
