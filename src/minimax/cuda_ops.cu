// src/minimax/cuda_ops.cu - fp32 GEMV and GGUF dequant tiles for MiniMax dense tensors.
#include "cuda_ops.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "strata/kernels/dequant_bf16.hpp"

namespace strata::minimax::cu {
namespace {

void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

__global__ void gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int cols) {
    const int row = (int)blockIdx.x;
    const float* w = W + (size_t)row * (size_t)cols;
    float sum = 0.f;
    for (int i = (int)threadIdx.x; i < cols; i += (int)blockDim.x) sum += w[i] * x[i];
    const unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(mask, sum, off);
    __shared__ float warp[32];
    const int lane = (int)threadIdx.x & 31;
    const int wr = (int)threadIdx.x >> 5;
    if (lane == 0) warp[wr] = sum;
    __syncthreads();
    if (wr == 0) {
        const int nwarps = (int)blockDim.x >> 5;
        sum = (lane < nwarps) ? warp[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(mask, sum, off);
        if (lane == 0) y[row] = sum;
    }
}

}  // namespace

void init() {
    int n = 0;
    check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    if (n < 1) throw std::runtime_error("cuda: no device");
    check(cudaSetDevice(0), "cudaSetDevice(0)");
}

MemInfo mem() {
    MemInfo m;
    check(cudaMemGetInfo(&m.free_bytes, &m.total_bytes), "cudaMemGetInfo");
    return m;
}

void* alloc(size_t bytes) {
    if (bytes == 0) bytes = 4;
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

void dealloc(void* p) {
    if (p) cudaFree(p);
}

void copy_h2d(void* dst, const void* src, size_t bytes) {
    if (bytes) check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

void copy_d2h(void* dst, const void* src, size_t bytes) {
    if (bytes) check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

void gemv(const float* W, const float* x, float* y, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    gemv_f32<<<rows, 256>>>(W, x, y, cols);
    check(cudaGetLastError(), "gemv launch");
    check(cudaDeviceSynchronize(), "gemv sync");
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out) {
    strata::kernels::dequant_f32(ggml_type, blocks, row0, rows, cols, out, nullptr);
    check(cudaGetLastError(), "dequant_f32");
    check(cudaDeviceSynchronize(), "dequant sync");
}

}  // namespace strata::minimax::cu
