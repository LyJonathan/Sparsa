# third_party/ggml-tq

Two headers from [agentiai/llama.cpp](https://github.com/agentionai/llama.cpp) ("Gyro") that give the
pinned upstream ggml-cuda the trellis expert types TQ2_T, TQK6 and TQK7 for its MMQ prompt path:

- `tq.cuh` - derived from the fork's `ggml/src/ggml-cuda/tq.cuh`; only the parts the MMQ tile loader
  needs (no matrix-vector kernels), reading Strata's copies of the codebook tables instead of the
  fork's `ggml-common.h`.
- `tq-mmq-tiles.cuh` - the fork's `ggml/src/ggml-cuda/mmq-load-tiles.cuh` trellis tile loader,
  unchanged.

`cmake/strata_ggml_tq_overlay.cmake` adds them to a build tree at configure time; they are not
compiled on their own. The types matter for the Gyro Qwen models, not for MiniMax; the MiniMax engine
does not use them.

Both files are MIT, copyright the ggml authors and Laurent Zuijdwijk. This directory carries no
separate licence file upstream, so the notice is here and in the repository's
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

```
MIT License

Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2023-2024 The llama.cpp authors
Copyright (c) 2026 Laurent Zuijdwijk
Copyright (c) 2026 agentionai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
