// include/strata/kernels/tq_encode_cuda.hpp - the GPU backend of the trellis encoder (see tq_encode.hpp for the
// format and the search; src/kernels/cuda/tq_encode.cu for the kernel).
//
// TQ2_T (ggml type 144) only: its 16-bit step window is exactly two adjacent path bytes, so the trellis is a chain
// over one byte and the whole search fits one CUDA block per 128 weights with no global scratch.  TQK6/TQK7 need a
// two-chunk DP state whose back-pointer history does not fit in shared memory at one block per trellis block; they
// run on the CPU.  `tq_encode_cuda_supported` says which.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Whether this binary was built with the CUDA encoder.
bool tq_encode_cuda_available() noexcept;
/// Whether the GPU backend encodes this type (TQ2_T only).
bool tq_encode_cuda_supported(int ggml_type) noexcept;

/// Rows x n weights of `rows` f32 rows (`n` a multiple of 128) -> `rows * n / 128` 34-byte blocks.  `w` may be
/// null (unweighted cost); when `w_row` it is one row of `n` weights reused for every row, otherwise a full
/// rows x n matrix.  Returns false when the type or the device is unavailable.
bool tq_encode_rows_cuda(int ggml_type, const float* x, const float* w, bool w_row, int64_t rows, int64_t n,
                         int passes, uint8_t* out);

}  // namespace strata::kernels
