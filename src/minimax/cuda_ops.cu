// src/minimax/cuda_ops.cu - fp32 GEMV and GGUF dequant tiles for MiniMax dense tensors.
#include "cuda_ops.hpp"

#include <cmath>
#include <cstdio>
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

__device__ __forceinline__ float warp_max(float v) {
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(mask, v, o));
    return v;
}

__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = ((int)blockDim.x + 31) >> 5;
    v = (lane < nw) ? red[lane] : -INFINITY;
    v = warp_max(v);
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = ((int)blockDim.x + 31) >> 5;
    v = (lane < nw) ? red[lane] : 0.f;
    v = warp_sum(v);
    return __shfl_sync(0xffffffffu, v, 0);
}

// --- MSA indexer: per (block, indexer head) max of the query/key dots over the tokens of that block.
__global__ void idx_scores_kernel(const __half* __restrict__ ik, long long ntok, const float* __restrict__ q,
                                  int dim, int blk, int nb, float* __restrict__ bs) {
    __shared__ float red[32];
    const int b = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    long long t0 = (long long)b * blk;
    long long t1 = t0 + blk;
    if (t1 > ntok) t1 = ntok;
    float m = -INFINITY;
    if (t1 > t0) {
        const float* qh = q + (size_t)h * dim;
        for (long long t = t0 + threadIdx.x; t < t1; t += blockDim.x) {
            const __half* row = ik + (size_t)t * dim;
            float s = 0.f;
            for (int d = 0; d < dim; ++d) s += __half2float(row[d]) * qh[d];
            m = fmaxf(m, s);
        }
    }
    m = block_max(m, red);
    if (threadIdx.x == 0) bs[(size_t)h * nb + b] = m;
}

// --- one KV head's selected rows, copied from the fp16 cache into a contiguous staging buffer.
__global__ void kv_gather_kernel(const __half* __restrict__ k, const __half* __restrict__ v, int hkv, int dim,
                                 const int* __restrict__ slot_pos, const int* __restrict__ counts, int maxslot,
                                 __half* __restrict__ kdst, __half* __restrict__ vdst) {
    const int h = (int)blockIdx.x;
    const int j = (int)blockIdx.y;
    if (j >= counts[h]) return;
    const long long pos = slot_pos[(size_t)h * maxslot + j];
    const __half* ks = k + ((size_t)pos * hkv + h) * dim;
    const __half* vs = v + ((size_t)pos * hkv + h) * dim;
    __half* kd = kdst + ((size_t)h * maxslot + j) * dim;
    __half* vd = vdst + ((size_t)h * maxslot + j) * dim;
    for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) {
        kd[d] = ks[d];
        vd[d] = vs[d];
    }
}

// --- decode attention over the gathered slots: one block per query head.
__global__ void msa_attn_kernel(const __half* __restrict__ k, const __half* __restrict__ v,
                                const float* __restrict__ q, float* __restrict__ out, int q_per_kv, int maxslot,
                                const int* __restrict__ counts, int dim, float scale) {
    extern __shared__ float s[];
    __shared__ float red[32];
    const int qh = (int)blockIdx.x;
    const int kv = qh / q_per_kv;
    const int n = counts[kv];
    const __half* kb = k + (size_t)kv * maxslot * dim;
    const __half* vb = v + (size_t)kv * maxslot * dim;
    const float* qh_p = q + (size_t)qh * dim;
    if (threadIdx.x == 0) red[0] = 0.f;
    __syncthreads();
    if (n <= 0) {
        for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) out[(size_t)qh * dim + d] = 0.f;
        return;
    }
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        const __half* row = kb + (size_t)i * dim;
        float acc = 0.f;
        for (int d = 0; d < dim; ++d) acc += __half2float(row[d]) * qh_p[d];
        s[i] = acc * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) m = fmaxf(m, s[i]);
    m = block_max(m, red);
    __syncthreads();
    float sum = 0.f;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        const float e = __expf(s[i] - m);
        s[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n; ++i) acc += s[i] * __half2float(vb[(size_t)i * dim + d]);
        out[(size_t)qh * dim + d] = acc * inv;
    }
}

}  // namespace

void init() {
    int n = 0;
    check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    if (n < 1) throw std::runtime_error("cuda: no device");
    check(cudaSetDevice(0), "cudaSetDevice(0)");
}

int device_count() {
    int n = 0;
    check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    return n;
}

void set_device(int dev) { check(cudaSetDevice(dev), "cudaSetDevice"); }

int current_device() {
    int d = 0;
    check(cudaGetDevice(&d), "cudaGetDevice");
    return d;
}

MemInfo mem_on(int dev) {
    const int prev = current_device();
    if (prev != dev) set_device(dev);
    MemInfo m;
    check(cudaMemGetInfo(&m.free_bytes, &m.total_bytes), "cudaMemGetInfo");
    if (prev != dev) set_device(prev);
    return m;
}

void* alloc_on(int dev, size_t bytes) {
    // Leaves `dev` current: the caller is always about to use the buffer it just asked for.
    if (current_device() != dev) set_device(dev);
    return alloc(bytes);
}

void* host_alloc(size_t bytes) {
    if (bytes == 0) bytes = 4;
    void* p = nullptr;
    check(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    return p;
}

void host_free(void* p) {
    if (p) cudaFreeHost(p);
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
    if (!bytes) return;
    const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        int dev = -1;
        cudaGetDevice(&dev);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%p", dst);
        throw std::runtime_error(std::string("cuda: cudaMemcpy H2D: ") + cudaGetErrorString(e) + " (" +
                                 std::to_string(bytes) + " bytes to " + buf + " on device " + std::to_string(dev) + ")");
    }
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

void idx_block_scores(const void* ik, int64_t ntok, const float* q, int heads, int dim, int blk, int nb, float* bs,
                      void* stream) {
    if (nb <= 0 || heads <= 0) return;
    const dim3 grid((unsigned)nb, (unsigned)heads);
    idx_scores_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>((const __half*)ik, (long long)ntok, q, dim, blk, nb,
                                                              bs);
    check(cudaGetLastError(), "idx_scores launch");
}

void kv_gather(const void* k, const void* v, int hkv, int dim, const int32_t* slot_pos, const int32_t* counts,
               int maxslot, void* kdst, void* vdst, void* stream) {
    if (hkv <= 0 || maxslot <= 0) return;
    const dim3 grid((unsigned)hkv, (unsigned)maxslot);
    kv_gather_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>((const __half*)k, (const __half*)v, hkv, dim,
                                                             slot_pos, counts, maxslot, (__half*)kdst,
                                                             (__half*)vdst);
    check(cudaGetLastError(), "kv_gather launch");
}

void msa_attn(const void* kstage, const void* vstage, const float* q, float* out, int nheads, int q_per_kv,
              int maxslot, const int32_t* counts, int dim, void* stream) {
    if (nheads <= 0) return;
    const float scale = 1.f / std::sqrt((float)dim);
    const size_t shm = (size_t)maxslot * sizeof(float);
    msa_attn_kernel<<<nheads, 256, shm, (cudaStream_t)stream>>>((const __half*)kstage, (const __half*)vstage, q, out,
                                                                q_per_kv, maxslot, counts, dim, scale);
    check(cudaGetLastError(), "msa_attn launch");
}

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
