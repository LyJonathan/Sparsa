// src/minimax/cuda_ops.hpp - device launches for the MiniMax forward pass. Host-only types; the .cu file owns CUDA.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::minimax::cu {

struct MemInfo {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

void init();
MemInfo mem();
void* alloc(size_t bytes);
void dealloc(void* p);
void copy_h2d(void* dst, const void* src, size_t bytes);
void copy_d2h(void* dst, const void* src, size_t bytes);

/// y[rows] = W[rows, cols] * x[cols], all device pointers, row-major W. Synchronizes before return.
void gemv(const float* W, const float* x, float* y, int rows, int cols);

/// Dequantize `rows` rows of a GGUF matrix starting at `row0` into device fp32. Synchronizes.
void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out);

}  // namespace strata::minimax::cu
