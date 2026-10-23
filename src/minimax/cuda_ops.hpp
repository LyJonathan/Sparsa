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

void sync();

/// A non-blocking side stream for the adaptive VRAM swaps.
void* stream_create();
void stream_destroy(void* stream);
void copy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
void stream_sync(void* stream);

/// Grouped routed-expert MLP for F32/F16/BF16 experts, all on the device.  `n_entries` selected experts:
/// gate[e]/up[e] are `n_ff` rows of `n_embd`, down[e] is `n_embd` rows of `n_ff`, GGUF row-major bytes.
/// `gu_row`/`d_row` are their ggml row sizes.  `h` is scratch `n_entries * n_ff` floats, `out` is
/// `n_entries * n_embd` floats.  swigluoai is fused into the gate/up pass.  Synchronizes.
void moe_expert_f16(int type, const void* const* gate, const void* const* up, const void* const* down,
                    int n_entries, int n_embd, int n_ff, size_t gu_row, size_t d_row, const float* x, float* h,
                    float* out, float alpha, float limit);

}  // namespace strata::minimax::cu
