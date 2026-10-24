// src/tq_cuda_stub.cpp - the CUDA trellis encoder's entry points when the tool is built without nvcc.
#include "strata/kernels/tq_encode_cuda.hpp"

namespace strata::kernels {

bool tq_encode_cuda_available() noexcept { return false; }
bool tq_encode_cuda_supported(int) noexcept { return false; }

bool tq_encode_rows_cuda(int, const float*, const float*, bool, int64_t, int64_t, int, uint8_t*) { return false; }

}  // namespace strata::kernels
