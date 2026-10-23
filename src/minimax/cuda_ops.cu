// src/minimax/cuda_ops.cu - fp32 GEMV and GGUF dequant tiles for MiniMax dense tensors.
#include "cuda_ops.hpp"

#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

__device__ __forceinline__ float ld_weight(int type, const uint8_t* row, int i) {
    if (type == 0) return ((const float*)row)[i];
    if (type == 1) return __half2float(((const __half*)row)[i]);
    return __bfloat162float(((const __nv_bfloat16*)row)[i]);
}

__device__ __forceinline__ float swigluoai_dev(float gate, float up, float alpha, float limit) {
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    return (up + 1.f) * gate * (1.f / (1.f + __expf(-alpha * gate)));
}

__device__ __forceinline__ float warp_sum(float v) {
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(mask, v, o);
    return v;
}

// gate and up for one selected expert row, swigluoai fused into the result.
__global__ void moe_gu_f16(int type, const void* const* gate, const void* const* up, int n_embd, int n_ff,
                           size_t gu_row, const float* __restrict__ x, float* __restrict__ h, float alpha,
                           float limit) {
    const int e = (int)blockIdx.z;
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= n_ff) return;
    const uint8_t* gr = (const uint8_t*)gate[e] + (size_t)row * gu_row;
    const uint8_t* ur = (const uint8_t*)up[e] + (size_t)row * gu_row;
    float sg = 0.f, su = 0.f;
    for (int i = (int)threadIdx.x; i < n_embd; i += (int)blockDim.x) {
        const float xv = x[i];
        sg += ld_weight(type, gr, i) * xv;
        su += ld_weight(type, ur, i) * xv;
    }
    sg = warp_sum(sg);
    su = warp_sum(su);
    if (threadIdx.x == 0) h[(size_t)e * n_ff + row] = swigluoai_dev(sg, su, alpha, limit);
}

__global__ void moe_down_f16(int type, const void* const* down, int n_embd, int n_ff, size_t d_row,
                             const float* __restrict__ h, float* __restrict__ out) {
    const int e = (int)blockIdx.z;
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= n_embd) return;
    const uint8_t* dr = (const uint8_t*)down[e] + (size_t)row * d_row;
    const float* he = h + (size_t)e * n_ff;
    float s = 0.f;
    for (int i = (int)threadIdx.x; i < n_ff; i += (int)blockDim.x) s += ld_weight(type, dr, i) * he[i];
    s = warp_sum(s);
    if (threadIdx.x == 0) out[(size_t)e * n_embd + row] = s;
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

void sync() { check(cudaDeviceSynchronize(), "sync"); }

void* stream_create() {
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream create");
    return (void*)s;
}

void stream_destroy(void* stream) {
    if (stream) cudaStreamDestroy((cudaStream_t)stream);
}

void copy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes) check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream), "copy H2D async");
}

void stream_sync(void* stream) { check(cudaStreamSynchronize((cudaStream_t)stream), "stream sync"); }

void moe_expert_f16(int type, const void* const* gate, const void* const* up, const void* const* down,
                    int n_entries, int n_embd, int n_ff, size_t gu_row, size_t d_row, const float* x, float* h,
                    float* out, float alpha, float limit) {
    if (n_entries <= 0) return;
    if (type != 0 && type != 1 && type != 30)
        throw std::runtime_error("moe_expert_f16: only F32, F16 and BF16 experts use this path");
    const dim3 block(32, 4);
    const dim3 ggrid(1, (unsigned)((n_ff + 3) / 4), (unsigned)n_entries);
    moe_gu_f16<<<ggrid, block>>>(type, gate, up, n_embd, n_ff, gu_row, x, h, alpha, limit);
    check(cudaGetLastError(), "moe_gu_f16 launch");
    const dim3 dgrid(1, (unsigned)((n_embd + 3) / 4), (unsigned)n_entries);
    moe_down_f16<<<dgrid, block>>>(type, down, n_embd, n_ff, d_row, h, out);
    check(cudaGetLastError(), "moe_down_f16 launch");
    check(cudaDeviceSynchronize(), "moe_expert_f16 sync");
}

}  // namespace strata::minimax::cu
