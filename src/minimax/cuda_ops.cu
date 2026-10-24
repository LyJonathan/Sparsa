// src/minimax/cuda_ops.cu - fp32 GEMV and GGUF dequant tiles for MiniMax dense tensors.
#include "cuda_ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"

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

// ============================================================================================= P5a: batched prefill

namespace {

__global__ void f32_to_f16_kernel(const float* __restrict__ x, __half* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void gather_rows_f16_kernel(const float* __restrict__ x, const int* __restrict__ rows, long long n_rows,
                                       int cols, __half* __restrict__ y) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * cols) return;
    const long long r = i / cols, c = i % cols;
    y[i] = __float2half(x[(long long)rows[r] * cols + c]);
}

__global__ void scale_rows_kernel(float* __restrict__ y, long long n, float s) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

__global__ void gather_rows_f32_kernel(const float* __restrict__ x, const int* __restrict__ rows, long long n_rows,
                                       int cols, float* __restrict__ y) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * cols) return;
    const long long r = i / cols, c = i % cols;
    y[i] = x[(long long)rows[r] * cols + c];
}

__global__ void swiglu_rows_f32_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                       float* __restrict__ h, long long n, float alpha, float limit) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gate = g[i], up = u[i];
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    h[i] = (up + 1.f) * gate * (1.f / (1.f + __expf(-alpha * gate)));
}

__global__ void swiglu_rows_f16_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                       __half* __restrict__ h, long long n, float alpha, float limit) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gate = g[i], up = u[i];
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    h[i] = __float2half((up + 1.f) * gate * (1.f / (1.f + __expf(-alpha * gate))));
}

__global__ void moe_combine_kernel(const float* __restrict__ Y, const int* __restrict__ ent_tok,
                                   const float* __restrict__ ent_w, long long n_rows, int n_embd,
                                   float* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * n_embd) return;
    const long long r = i / n_embd, c = i % n_embd;
    const int t = ent_tok[r];
    if (t < 0) return;
    out[(long long)t * n_embd + c] += ent_w[r] * Y[i];
}

// One block per (query token, indexer head, block): the max of the query/key dots over the block's causal prefix.
__global__ void idx_scores_batch_kernel(const __half* __restrict__ ik, const __half* __restrict__ q, int dim, int blk,
                                        long long pos0, float* __restrict__ bs) {
    __shared__ float red[32];
    const int b = (int)blockIdx.x, h = (int)blockIdx.y;
    const long long t = (long long)blockIdx.z;
    const long long nb = (long long)gridDim.x;
    const long long pos = pos0 + t;
    long long t0 = (long long)b * blk;
    long long t1 = t0 + blk;
    if (t1 > pos + 1) t1 = pos + 1;
    float m = -INFINITY;
    if (t1 > t0) {
        const __half* qh = q + (t * gridDim.y + h) * (long long)dim;
        for (long long p = t0 + threadIdx.x; p < t1; p += blockDim.x) {
            const __half* row = ik + p * dim;
            float s = 0.f;
            for (int d = 0; d < dim; ++d) s += __half2float(row[d]) * __half2float(qh[d]);
            m = fmaxf(m, s);
        }
    }
    m = block_max(m, red);
    if (threadIdx.x == 0) bs[(t * gridDim.y + h) * nb + b] = m;
}

// One block per (query token, head).  Indexed layers store the selected scores/positions in shared memory (at most
// top_k blocks x block_size); the dense fallback (no indexer) walks the cache three times so no 32 K-float buffer is
// needed.
__global__ void msa_attn_batch_kernel(const __half* __restrict__ k, const __half* __restrict__ v,
                                      const __half* __restrict__ q, const int* __restrict__ blk,
                                      const int* __restrict__ nblk, int q_per_kv, int q_per_idx, int dim, int blk_size,
                                      int dense_all, long long pos0, long long max_blk, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float red[32];
    float* s = reinterpret_cast<float*>(smem);
    int* sp = reinterpret_cast<int*>(s + max_blk * blk_size);
    const long long t = (long long)blockIdx.x;
    const int qh = (int)blockIdx.y;
    const int kv = qh / q_per_kv;
    const int ih = qh / q_per_idx;
    const long long pos = pos0 + t;
    const __half* qrow = q + (t * gridDim.y + qh) * (long long)dim;
    const float att_scale = 1.f / sqrtf((float)dim);
    const __half* kb = k + (long long)kv * dim;
    const __half* vb = v + (long long)kv * dim;
    const int hkv = gridDim.y / q_per_kv;
    long long n = 0;
    if (dense_all) {
        // max over all causal positions
        float m = -INFINITY;
        const long long cnt = pos + 1;
        for (long long p = threadIdx.x; p < cnt; p += blockDim.x) {
            const __half* row = k + (p * hkv + kv) * (long long)dim;
            float acc = 0.f;
            for (int d = 0; d < dim; ++d) acc += __half2float(row[d]) * __half2float(qrow[d]);
            m = fmaxf(m, acc * att_scale);
        }
        m = block_max(m, red);
        __syncthreads();
        float sum = 0.f;
        for (long long p = threadIdx.x; p < cnt; p += blockDim.x) {
            const __half* row = k + (p * hkv + kv) * (long long)dim;
            float acc = 0.f;
            for (int d = 0; d < dim; ++d) acc += __half2float(row[d]) * __half2float(qrow[d]);
            sum += __expf(acc * att_scale - m);
        }
        sum = block_sum(sum, red);
        __syncthreads();
        const float inv = sum > 0.f ? 1.f / sum : 0.f;
        for (int d = (int)threadIdx.x; d < dim; d += blockDim.x) {
            float acc = 0.f;
            for (long long p = 0; p < cnt; ++p) {
                const __half* row = k + (p * hkv + kv) * (long long)dim;
                float sc = 0.f;
                for (int e = 0; e < dim; ++e) sc += __half2float(row[e]) * __half2float(qrow[e]);
                acc += __expf(sc * att_scale - m) * __half2float(v[(p * hkv + kv) * (long long)dim + d]);
            }
            out[(t * gridDim.y + qh) * (long long)dim + d] = acc * inv;
        }
        return;
    }
    const int nb = nblk[t * gridDim.z + ih];
    for (int j = 0; j < nb && j < (int)max_blk; ++j) {
        const int b = blk[(t * gridDim.z + ih) * max_blk + j];
        if (b < 0) continue;
        long long p0 = (long long)b * blk_size;
        long long p1 = p0 + blk_size;
        if (p1 > pos + 1) p1 = pos + 1;
        for (long long p = p0; p < p1 && p <= pos && n < max_blk * blk_size; ++p) sp[n++] = (int)p;
    }
    __syncthreads();
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const __half* row = k + ((long long)sp[i] * hkv + kv) * (long long)dim;
        float acc = 0.f;
        for (int d = 0; d < dim; ++d) acc += __half2float(row[d]) * __half2float(qrow[d]);
        s[i] = acc * att_scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[i]);
    m = block_max(m, red);
    __syncthreads();
    float sum = 0.f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = __expf(s[i] - m);
        s[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int d = (int)threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.f;
        for (long long i = 0; i < n; ++i) acc += s[i] * __half2float(v[((long long)sp[i] * hkv + kv) * dim + d]);
        out[(t * gridDim.y + qh) * (long long)dim + d] = acc * inv;
    }
}

__global__ void dequant_raw_f16_kernel(int type, const void* __restrict__ src, long long n, __half* __restrict__ dst) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (type == 0) dst[i] = __float2half(((const float*)src)[i]);
    else if (type == 1) dst[i] = ((const __half*)src)[i];
    else dst[i] = __float2half(__bfloat162float(((const __nv_bfloat16*)src)[i]));
}

__global__ void dequant_raw_f32_kernel(int type, const void* __restrict__ src, long long n, float* __restrict__ dst) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (type == 0) dst[i] = ((const float*)src)[i];
    else if (type == 1) dst[i] = __half2float(((const __half*)src)[i]);
    else dst[i] = __bfloat162float(((const __nv_bfloat16*)src)[i]);
}

cublasHandle_t blas_handle() {
    static cublasHandle_t h[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) throw std::runtime_error("cublas: device out of range");
    if (!h[dev]) {
        if (cublasCreate(&h[dev]) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublas: cublasCreate failed");
    }
    return h[dev];
}

}  // namespace


// Four query heads of one KV head per block: the K rows are dotted against all four queries and the V rows are read
// Four query heads of one KV head per block: the K rows are dotted against all four queries and the V rows are read
__global__ void msa_attn_flash_kernel(const __half* __restrict__ k, const __half* __restrict__ v,
                                      const __half* __restrict__ q, const int* __restrict__ blk,
                                      const int* __restrict__ nblk, int q_per_kv, int dim, int blk_size, int heads,
                                      long long pos0, long long max_blk, float* __restrict__ out) {
    // P5b: one block per (query token, KV head) covering all `q_per_kv` query heads of that KV head, positions in
    // chunks of 128 with an online softmax.  Keeps Q in shared memory, reads every V row once for the whole head
    // group, and needs ~15 KB of shared memory instead of the 43 KB a per-position score array costs - two blocks
    // per SM was the ceiling of the P5a kernel.
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float red8[16][32];
    const int H = q_per_kv;              // query heads in this block (<= 8)
    const int C = 128;                   // positions per chunk
    const long long ms = (long long)max_blk * (long long)blk_size;
    __half* qs = (__half*)smem;                                  // [H][dim]
    int* sp = (int*)(qs + (size_t)H * (size_t)dim);              // [ms]
    float* s_sh = (float*)(sp + ms);                             // [H][C]
    float* run_m = s_sh + (size_t)H * C;                         // [H]
    float* run_l = run_m + H;                                    // [H]
    float* sm_mn = run_l + H;                                    // [H] chunk max, set by one thread per head
    float* sm_alpha = sm_mn + H;                                 // [H] rescale factor
    const long long t = (long long)blockIdx.x;
    const int kv = (int)blockIdx.y;
    const int q0 = kv * H;
    const long long pos = pos0 + t;
    const int hkv = (int)gridDim.y;
    const float att_scale = 1.f / sqrtf((float)dim);

    const int nb = nblk[t * (long long)gridDim.y + kv];
    __shared__ int n_shared;
    if (threadIdx.x == 0) {
        long long nn = 0;
        for (int j = 0; j < nb && j < (int)max_blk; ++j) {
            const int b = blk[(t * (long long)gridDim.y + kv) * max_blk + j];
            if (b < 0) continue;
            long long p0 = (long long)b * blk_size;
            long long p1 = p0 + blk_size;
            if (p1 > pos + 1) p1 = pos + 1;
            for (long long p = p0; p < p1 && p <= pos && nn < ms; ++p) sp[nn++] = (int)p;
        }
        n_shared = (int)nn;
    }
    for (int i = (int)threadIdx.x; i < H * dim; i += (int)blockDim.x)
        qs[i] = q[((t * (long long)heads) + q0) * dim + i];
    if (threadIdx.x < H) {
        run_m[threadIdx.x] = -INFINITY;
        run_l[threadIdx.x] = 0.f;
    }
    __syncthreads();
    const long long n = n_shared;

    float acc[16];
#pragma unroll
    for (int h = 0; h < 16; ++h) acc[h] = 0.f;

    for (long long base = 0; base < n; base += C) {
        const int cn = (int)((n - base) < (long long)C ? (n - base) : (long long)C);
        // ---- scores for this chunk: one thread per position, every head of the group
        if ((int)threadIdx.x < cn) {
            const int p = sp[base + threadIdx.x];
            const __half2* k2 = (const __half2*)(k + ((long long)p * hkv + kv) * dim);
            for (int h = 0; h < H; ++h) {
                const __half2* q2 = (const __half2*)(qs + h * dim);
                // Two independent float2 chains over the head's dimensions (the parallelism is what makes the
                // chunked score pass cheap; see the note in test_p5a.py about its summation order).
                float2 a0 = make_float2(0.f, 0.f), a1 = make_float2(0.f, 0.f);
                int d2 = 0;
                for (; d2 + 1 < dim / 2; d2 += 2) {
                    const float2 ka = __half22float2(k2[d2]);
                    const float2 qa = __half22float2(q2[d2]);
                    a0.x += ka.x * qa.x;
                    a0.y += ka.y * qa.y;
                    const float2 kb = __half22float2(k2[d2 + 1]);
                    const float2 qb = __half22float2(q2[d2 + 1]);
                    a1.x += kb.x * qb.x;
                    a1.y += kb.y * qb.y;
                }
                for (; d2 < dim / 2; ++d2) {
                    const float2 ka = __half22float2(k2[d2]);
                    const float2 qa = __half22float2(q2[d2]);
                    a0.x += ka.x * qa.x;
                    a0.y += ka.y * qa.y;
                }
                s_sh[h * C + (int)threadIdx.x] = (a0.x + a0.y + a1.x + a1.y) * att_scale;
            }
        }
        __syncthreads();
        // ---- chunk max per head, then the running max and its rescale factor.  Only one thread per head may
        // touch run_m / run_l: with every thread doing the read-modify-write the running state (and therefore the
        // output) depended on the warp interleaving, which made the chunked prefill non-deterministic.
        float rv[16];
        for (int h = 0; h < H; ++h) {
            float m = -INFINITY;
            for (int i = (int)threadIdx.x; i < cn; i += (int)blockDim.x) m = fmaxf(m, s_sh[h * C + i]);
            rv[h] = m;
        }
        for (int h = H; h < 16; ++h) rv[h] = -INFINITY;
        for (int h = 0; h < 16; ++h) {
            float x = rv[h];
            for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, o));
            if ((threadIdx.x & 31) == 0) red8[h][threadIdx.x >> 5] = x;
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            float x = red8[threadIdx.x][0];
            for (int w = 1; w < (int)(blockDim.x >> 5); ++w) x = fmaxf(x, red8[threadIdx.x][w]);
            red8[threadIdx.x][0] = x;
        }
        __syncthreads();
        if ((int)threadIdx.x < H) {
            const int h = (int)threadIdx.x;
            const float mn = fmaxf(run_m[h], red8[h][0]);
            sm_mn[h] = mn;
            sm_alpha[h] = (run_m[h] == -INFINITY) ? 0.f : __expf(run_m[h] - mn);
            run_m[h] = mn;
        }
        __syncthreads();
        if ((int)threadIdx.x < cn)
            for (int h = 0; h < H; ++h)
                s_sh[h * C + (int)threadIdx.x] = __expf(s_sh[h * C + (int)threadIdx.x] - sm_mn[h]);
        __syncthreads();
        // ---- running denominator
        for (int h = 0; h < H; ++h) {
            float s = 0.f;
            for (int i = (int)threadIdx.x; i < cn; i += (int)blockDim.x) s += s_sh[h * C + i];
            rv[h] = s;
        }
        for (int h = H; h < 16; ++h) rv[h] = 0.f;
        for (int h = 0; h < 16; ++h) {
            float x = rv[h];
            for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
            if ((threadIdx.x & 31) == 0) red8[h][threadIdx.x >> 5] = x;
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            float x = red8[threadIdx.x][0];
            for (int w = 1; w < (int)(blockDim.x >> 5); ++w) x += red8[threadIdx.x][w];
            red8[threadIdx.x][0] = x;
        }
        __syncthreads();
        if ((int)threadIdx.x < H) {
            const int h = (int)threadIdx.x;
            run_l[h] = run_l[h] * sm_alpha[h] + red8[h][0];
        }
        for (int h = 0; h < H; ++h) acc[h] *= sm_alpha[h];
        // ---- P(chunk) x V(chunk), one thread per dimension; the V row is read once for the whole head group
        const int d = (int)threadIdx.x;
        if (d < dim) {
            for (int i = 0; i < cn; ++i) {
                const float vv = __half2float(v[((long long)sp[base + i] * hkv + kv) * dim + d]);
                for (int h = 0; h < H; ++h) acc[h] += s_sh[h * C + i] * vv;
            }
        }
        __syncthreads();
    }
    const int d = (int)threadIdx.x;
    if (d < dim) {
        for (int h = 0; h < H; ++h) {
            const float l = run_l[h] > 0.f ? run_l[h] : 1.f;
            out[((t * (long long)heads) + q0 + h) * dim + d] = acc[h] / l;
        }
    }
}

__global__ void msa_attn_batch4_kernel(const __half* __restrict__ k, const __half* __restrict__ v,
                                       const __half* __restrict__ q, const int* __restrict__ blk,
                                       const int* __restrict__ nblk, int q_per_kv, int q_per_idx, int dim,
                                       int blk_size, int dense_all, long long pos0, long long max_blk,
                                       float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float red[32];
    const int ms = (int)max_blk * blk_size;
    float* s = reinterpret_cast<float*>(smem);            // [4][ms]
    int* sp = reinterpret_cast<int*>(s + 4 * ms);
    const long long t = (long long)blockIdx.x;
    const int q0 = (int)blockIdx.y * 4;                   // first of the four query heads
    const int kv = q0 / q_per_kv;
    const int ih = q0 / q_per_idx;
    const long long pos = pos0 + t;
    const float att_scale = 1.f / sqrtf((float)dim);
    const __half* krow_base = k;
    const __half* vrow_base = v;
    const int hkv = gridDim.y * 4 / q_per_kv;
    long long n = 0;
    if (dense_all) {
        for (long long p = 0; p < pos + 1 && n < ms; ++p) sp[n++] = (int)p;
    } else {
        const int nb = nblk[t * gridDim.z + ih];
        for (int j = 0; j < nb && j < (int)max_blk; ++j) {
            const int b = blk[(t * gridDim.z + ih) * max_blk + j];
            if (b < 0) continue;
            long long p0 = (long long)b * blk_size;
            long long p1 = p0 + blk_size;
            if (p1 > pos + 1) p1 = pos + 1;
            for (long long p = p0; p < p1 && p <= pos && n < ms; ++p) sp[n++] = (int)p;
        }
    }
    __syncthreads();
    // scores for the four heads
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const __half* row = krow_base + ((long long)sp[i] * hkv + kv) * (long long)dim;
        const __half2* r2 = (const __half2*)row;
        for (int h = 0; h < 4; ++h) {
            const __half2* q2 = (const __half2*)(q + (t * gridDim.y * 4 + q0 + h) * (long long)dim);
            float2 acc = make_float2(0.f, 0.f);
            for (int d2 = 0; d2 < dim / 2; ++d2) {
                const float2 a = __half22float2(r2[d2]);
                const float2 b = __half22float2(q2[d2]);
                acc.x += a.x * b.x;
                acc.y += a.y * b.y;
            }
            s[(size_t)h * ms + i] = (acc.x + acc.y) * att_scale;
        }
    }
    __syncthreads();
    __shared__ float inv4[4];
    for (int h = 0; h < 4; ++h) {
        float m = -INFINITY;
        for (long long i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[(size_t)h * ms + i]);
        m = block_max(m, red);
        __syncthreads();
        float sum = 0.f;
        for (long long i = threadIdx.x; i < n; i += blockDim.x) {
            const float e = __expf(s[(size_t)h * ms + i] - m);
            s[(size_t)h * ms + i] = e;
            sum += e;
        }
        sum = block_sum(sum, red);
        __syncthreads();
        if (threadIdx.x == 0) inv4[h] = sum > 0.f ? 1.f / sum : 0.f;
    }
    __syncthreads();
    // V is read once and applied to the four heads' probabilities: this thread owns dimension `threadIdx.x`.
    if ((int)threadIdx.x < dim) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (long long i = 0; i < n; ++i) {
            const float vv = __half2float(vrow_base[((long long)sp[i] * hkv + kv) * dim + threadIdx.x]);
            acc[0] += s[(size_t)0 * ms + i] * vv;
            acc[1] += s[(size_t)1 * ms + i] * vv;
            acc[2] += s[(size_t)2 * ms + i] * vv;
            acc[3] += s[(size_t)3 * ms + i] * vv;
        }
        for (int hh = 0; hh < 4; ++hh)
            out[(t * gridDim.y * 4 + q0 + hh) * (long long)dim + threadIdx.x] = acc[hh] * inv4[hh];
    }
}


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

void copy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes) check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream),
                     "cudaMemcpyAsync D2H");
}

void copy_d2d(void* dst, const void* src, size_t bytes) {
    if (bytes) check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
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

void* event_create() {
    cudaEvent_t e = nullptr;
    check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event create");
    return (void*)e;
}

void event_record(void* event, void* stream) {
    if (event) check(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream), "event record");
}

void event_sync(void* event) {
    if (event) check(cudaEventSynchronize((cudaEvent_t)event), "event sync");
}

bool event_ready(void* event) {
    return !event || cudaEventQuery((cudaEvent_t)event) == cudaSuccess;
}

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

// ============================================================================================= P5a: batched prefill

void f32_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long blocks = (n + 255) / 256;
    f32_to_f16_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(x, (__half*)y, n);
    check(cudaGetLastError(), "f32_to_f16 launch");
}

void gather_rows_f16(const float* x, const int32_t* rows, int64_t n_rows, int64_t cols, uint16_t* y, void* stream) {
    if (n_rows <= 0 || cols <= 0) return;
    const long long n = n_rows * cols;
    const long long blocks = (n + 255) / 256;
    gather_rows_f16_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(x, rows, n_rows, (int)cols, (__half*)y);
    check(cudaGetLastError(), "gather_rows_f16 launch");
}

void scale_rows(float* y, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    const long long blocks = (n + 255) / 256;
    scale_rows_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(y, n, s);
    check(cudaGetLastError(), "scale_rows launch");
}

void gather_rows_f32(const float* x, const int32_t* rows, int64_t n_rows, int64_t cols, float* y, void* stream) {
    if (n_rows <= 0 || cols <= 0) return;
    const long long n = n_rows * cols;
    const long long blocks = (n + 255) / 256;
    gather_rows_f32_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(x, rows, n_rows, (int)cols, y);
    check(cudaGetLastError(), "gather_rows_f32 launch");
}

void dequant_to_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n <= 0) return;
    if (ggml_type == 0 || ggml_type == 1 || ggml_type == 30) {
        const long long blocks = (n + 255) / 256;
        dequant_raw_f16_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(ggml_type, src, n, (__half*)dst);
        check(cudaGetLastError(), "dequant_raw_f16 launch");
        return;
    }
    strata::kernels::dequant_f16(ggml_type, src, 0, 1, n, dst, stream);
    check(cudaGetLastError(), "dequant_to_f16 launch");
}

void gemm_f32(const float* X, const float* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
              void* stream) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    cublasHandle_t h = blas_handle();
    if (cublasSetStream(h, (cudaStream_t)stream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda: cublasSetStream failed");
    const float alpha = 1.f;
    const cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)T, (int)K, &alpha, W, (int)K, X,
                                          (int)K, &beta, Y, (int)(ldy > 0 ? ldy : N));
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda: cublasSgemm failed: " + std::to_string((int)st) + " (N=" + std::to_string((long long)N) + " T=" + std::to_string((long long)T) + " K=" + std::to_string((long long)K) + " ldy=" + std::to_string((long long)ldy) + ")");
    check(cudaGetLastError(), "gemm_f32");
}

void dequant_to_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream) {
    if (n <= 0) return;
    if (ggml_type == 0 || ggml_type == 1 || ggml_type == 30) {
        const long long blocks = (n + 255) / 256;
        dequant_raw_f32_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(ggml_type, src, n, dst);
        check(cudaGetLastError(), "dequant_raw_f32 launch");
        return;
    }
    // Everything else through the tiled dequantizer: iq_dequant_f32 only knows the i-quant formats and BF16, so a
    // Q8_0 / Q6_K weight (every "dense-lite" file, and both of this engine's default dense types) came out of it
    // as whatever the buffer held.  That was the f16 dense-prefill non-finite and the fp32 path's 9e-2 KL.
    strata::kernels::dequant_f32(ggml_type, src, 0, 1, n, dst, stream);
}

void gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              float beta, void* stream) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    cublasHandle_t h = blas_handle();
    if (cublasSetStream(h, (cudaStream_t)stream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda: cublasSetStream failed");
    const float alpha = 1.f;
    // C[T,N] = X[T,K] . W[N,K]^T, all row-major: cuBLAS sees C(N,T) = W(K,N)^T . X(K,T), i.e. op(A)=T with A=W.
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)T, (int)K, &alpha, W,
                                           CUDA_R_16F, (int)K, X, CUDA_R_16F, (int)K, &beta, Y, CUDA_R_32F,
                                           (int)(ldy > 0 ? ldy : N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda: cublasGemmEx failed: " + std::to_string((int)st));
    check(cudaGetLastError(), "gemm_f16");
}

void swiglu_rows_f16(const float* g, const float* u, uint16_t* h, int64_t rows, int64_t n_ff, float alpha,
                     float limit, void* stream) {
    const long long n = rows * n_ff;
    if (n <= 0) return;
    const long long blocks = (n + 255) / 256;
    swiglu_rows_f16_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(g, u, (__half*)h, n, alpha, limit);
    check(cudaGetLastError(), "swiglu_rows_f16 launch");
}

void swiglu_rows_f32(const float* g, const float* u, float* h, int64_t rows, int64_t n_ff, float alpha, float limit,
                     void* stream) {
    const long long n = rows * n_ff;
    if (n <= 0) return;
    const long long blocks = (n + 255) / 256;
    swiglu_rows_f32_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(g, u, h, n, alpha, limit);
    check(cudaGetLastError(), "swiglu_rows_f32 launch");
}

void moe_combine_rows(const float* Y, const int32_t* ent_tok, const float* ent_w, int64_t n_rows, int64_t n_embd,
                      float* out, int64_t n_tok, bool zero, void* stream) {
    const long long n = n_rows * n_embd;
    if (n <= 0) return;
    if (zero) check(cudaMemsetAsync(out, 0, (size_t)n_tok * (size_t)n_embd * sizeof(float), (cudaStream_t)stream),
                    "moe_combine zero");
    const long long blocks = (n + 255) / 256;
    moe_combine_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(Y, ent_tok, ent_w, n_rows, (int)n_embd,
                                                                          out);
    check(cudaGetLastError(), "moe_combine launch");
}

void idx_block_scores_batch(const void* ik, const uint16_t* q, int heads, int dim, int blk, int64_t nb, int64_t n_tok,
                            int64_t pos0, float* bs, void* stream) {
    if (nb <= 0 || heads <= 0 || n_tok <= 0) return;
    const dim3 grid((unsigned)nb, (unsigned)heads, (unsigned)n_tok);
    idx_scores_batch_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>((const __half*)ik, (const __half*)q, dim, blk, pos0,
                                                                  bs);
    check(cudaGetLastError(), "idx_scores_batch launch");
}

void msa_attn_batch(const void* k, const void* v, const uint16_t* q, const int32_t* blk, const int32_t* nblk, int heads,
                    int q_per_kv, int q_per_idx, int dim, int blk_size, int dense_all, int64_t pos0, int64_t n_tok,
                    int64_t max_blk, float* out, void* stream) {
    if (heads <= 0 || n_tok <= 0) return;
    const int idx_heads = dense_all ? 1 : (heads / (q_per_idx > 0 ? q_per_idx : 1));
    // P5b: indexed layers whose query heads per KV head equal the query heads per indexer head (M3.1: 8) take the
    // flash-style kernel: one block per (token, KV head), all 8 heads of that KV head, online softmax over the
    // selected positions in chunks of 128.  ~15 KB of shared memory instead of 43 KB.
    const int hkv_d = q_per_kv > 0 ? heads / q_per_kv : 0;
    const int idx_per_head = q_per_idx > 0 ? heads / q_per_idx : 0;
    static const bool msa_old = std::getenv("STRATA_MSA_OLD") != nullptr;
    if (!msa_old && !dense_all && hkv_d > 0 && hkv_d == idx_per_head && q_per_kv >= 1 && q_per_kv <= 16 && dim >= 2 &&
        dim <= 128 && dim % 2 == 0) {
        const size_t shm = (size_t)q_per_kv * (size_t)dim * 2 + (size_t)max_blk * (size_t)blk_size * sizeof(int) +
                           (size_t)q_per_kv * 128 * sizeof(float) + 4 * (size_t)q_per_kv * sizeof(float) + 64;
        if (shm <= 48u * 1024u) {
            const dim3 grid((unsigned)n_tok, (unsigned)hkv_d);
            msa_attn_flash_kernel<<<grid, 128, shm, (cudaStream_t)stream>>>(
                (const __half*)k, (const __half*)v, (const __half*)q, blk, nblk, q_per_kv, dim, blk_size, heads, pos0,
                max_blk, out);
            check(cudaGetLastError(), "msa_attn_flash launch");
            return;
        }
    }
    const size_t shm4 = (size_t)max_blk * (size_t)blk_size * (4 * sizeof(float) + sizeof(int));
    if (q_per_kv % 4 == 0 && q_per_idx % 4 == 0 && heads % 4 == 0 && dim % 2 == 0 && dim <= 128 &&
        (dense_all || shm4 <= 48u * 1024u)) {
        // Four query heads per block (see msa_attn_batch4_kernel).
        const dim3 grid4((unsigned)n_tok, (unsigned)(heads / 4), (unsigned)(idx_heads > 0 ? idx_heads : 1));
        msa_attn_batch4_kernel<<<grid4, 128, shm4, (cudaStream_t)stream>>>(
            (const __half*)k, (const __half*)v, (const __half*)q, blk, nblk, q_per_kv, q_per_idx, dim, blk_size,
            dense_all, pos0, max_blk, out);
        check(cudaGetLastError(), "msa_attn_batch4 launch");
        return;
    }
    const dim3 grid((unsigned)n_tok, (unsigned)heads, (unsigned)(idx_heads > 0 ? idx_heads : 1));
    const size_t shm = dense_all ? 0 : (size_t)max_blk * (size_t)blk_size * (sizeof(float) + sizeof(int));
    msa_attn_batch_kernel<<<grid, 128, shm, (cudaStream_t)stream>>>((const __half*)k, (const __half*)v,
                                                                   (const __half*)q, blk, nblk, q_per_kv, q_per_idx,
                                                                   dim, blk_size, dense_all, pos0, max_blk, out);
    check(cudaGetLastError(), "msa_attn_batch launch");
}

}  // namespace strata::minimax::cu
