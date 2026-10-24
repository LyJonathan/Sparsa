// src/kernels/cuda/tq_encode.cu - the TQ2_T encoder on the GPU.  One CUDA block (256 threads) per 128-weight
// trellis block: thread b is DP state b (the path byte of the previous step), and every thread recomputes the 256
// step costs it needs from the 2048-point codebook instead of materialising a 32 x 32768 float table.
//
// The search is the CPU one (src/kernels/cpu/tq_encode.cpp): two laps around the 32-step tail-biting circle per
// scale, traceback from the best final state, least-squares refit of d, and a second lap pair at the refitted
// scale.  The GPU evaluates the step cost as sum (x - d c)^2 while the CPU uses the algebraically rearranged
// sum x^2 - 2 d sum x c + d^2 sum c^2, so the two can pick different (equally good) paths in a tie and the written
// blocks are not required to be identical.
#include "strata/kernels/tq_encode_cuda.hpp"

#include "ggml-common-tq.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

// the codebook (and the exact fp16 conversion) live in the CPU encoder module, which is always linked with this one
namespace strata::kernels::cpu {
const float * tq_codebook();
}

namespace strata::kernels {
namespace {

constexpr int kBlock = 128;
constexpr int kSteps = 32;
constexpr uint32_t kGolden = 0x9e3779b1u;
constexpr uint32_t kStateMask = 0x7fffu;
constexpr int kStageBytes = 2 + 32;   // block_tq2_t

__device__ __forceinline__ float half_to_float_dev(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, b;
    if (e == 0x1fu) b = sign | 0x7f800000u | (m << 13);
    else if (e) b = sign | ((e + 112u) << 23) | (m << 13);
    else if (!m) b = sign;
    else {
        const float v = (float) m * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    return __uint_as_float(b);
}

__device__ __forceinline__ uint16_t float_to_half_dev(float f) {
    uint32_t b = __float_as_uint(f);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int32_t e = (int32_t) ((b >> 23) & 0xffu) - 127;
    uint32_t man = b & 0x7fffffu;
    if ((b & 0x7fffffffu) == 0) return (uint16_t) sign;
    if (e > 15) return (uint16_t) (sign | 0x7c00u);
    if (e < -24) return (uint16_t) sign;
    if (e < -14) {
        const uint32_t shift = (uint32_t) (-e - 13);
        uint32_t m = (man | 0x800000u) >> (13 + shift);
        if (((man | 0x800000u) >> (12 + shift)) & 1u) ++m;
        return (uint16_t) (sign | m);
    }
    uint32_t m = man >> 13;
    const uint32_t rest = man & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && ((man >> 12) & 1u))) {
        if (++m == 0x400u) {
            m = 0;
            if (++e > 15) return (uint16_t) (sign | 0x7c00u);
        }
    }
    return (uint16_t) (sign | ((uint32_t) (e + 15) << 10) | m);
}

// The codebook lives in shared memory (16 KiB): a trellis block needs all of it, and its access is random, so
// global (L1-resident) and shared are both acceptable; shared removes the L1 tag pressure.
__device__ __forceinline__ void load_codebook(const float* __restrict__ lut, float* sh) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) sh[i] = lut[i];
}

/// Sum of w_j (x_j - d c_j)^2 for the four weights of step t at 15-bit state s.
__device__ __forceinline__ float step_cost(const float* sh, const float* __restrict__ x,
                                           const float* __restrict__ w, float d, int t, uint32_t s) {
    const uint32_t h = s * kGolden;
    const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
    const float c0 = sh[2 * p0], c1 = sh[2 * p0 + 1], c2 = sh[2 * p1], c3 = sh[2 * p1 + 1];
    const int o = 4 * t;
    if (w) {
        const float e0 = x[o + 0] - d * c0, e1 = x[o + 1] - d * c1;
        const float e2 = x[o + 2] - d * c2, e3 = x[o + 3] - d * c3;
        return w[o + 0] * e0 * e0 + w[o + 1] * e1 * e1 + w[o + 2] * e2 * e2 + w[o + 3] * e3 * e3;
    }
    const float e0 = x[o + 0] - d * c0, e1 = x[o + 1] - d * c1;
    const float e2 = x[o + 2] - d * c2, e3 = x[o + 3] - d * c3;
    return e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
}

/// Two laps around the circle at scale d; writes the 32 path bytes into `path`.
__device__ void viterbi(const float* sh, const float* __restrict__ x, const float* __restrict__ w, float d,
                        uint8_t* path, float* dp_sh, float* dp2_sh, float* hist_sh) {
    const int b = threadIdx.x;   // 256 threads = 256 states
    dp_sh[b] = 0.0f;
    __syncthreads();
    for (int p = 0; p < 2 * kSteps; ++p) {
        const int t = p % kSteps;
        float best = 1e30f;
        for (int a = 0; a < 256; ++a) {
            const float v = dp_sh[a] + step_cost(sh, x, w, d, t, ((uint32_t) (a & 0x7f) << 8) | (uint32_t) b);
            best = v < best ? v : best;
        }
        dp2_sh[b] = best;
        __syncthreads();
        dp_sh[b] = best;
        if (p >= kSteps - 1) hist_sh[(p - (kSteps - 1)) * 256 + b] = best;
        __syncthreads();
    }
    // traceback: exact argmin of the same sum, one state at a time
    if (threadIdx.x == 0) {
        uint32_t st = 0;
        for (int i = 1; i < 256; ++i) {
            if (dp_sh[i] < dp_sh[st]) st = (uint32_t) i;
        }
        for (int p = 2 * kSteps - 1; p >= kSteps; --p) {
            const int t = p % kSteps;
            const float* prev = hist_sh + (p - 1 - (kSteps - 1)) * 256;
            float bestv = 1e30f;
            uint32_t besta = 0;
            for (uint32_t a = 0; a < 256; ++a) {
                const float v = prev[a] + step_cost(sh, x, w, d, t, ((a & 0x7fu) << 8) | st);
                if (v < bestv) {
                    bestv = v;
                    besta = a;
                }
            }
            path[t] = (uint8_t) st;
            st = besta;
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(256) tq2t_encode_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                          const float* __restrict__ lut, int64_t nblk,
                                                          int w_block_mod, int passes, uint8_t* __restrict__ out) {
    extern __shared__ float smem[];
    float* sh = smem;                  // 4096 codebook
    float* xs = sh + 4096;             // 128 weights of this block
    float* ws = xs + kBlock;           // 128 importance weights
    float* dp = ws + kBlock;           // 256 metrics
    float* dp2 = dp + 256;
    float* refit = dp2 + 256;          // {scale, score}
    float* hist = refit + 2;           // [kSteps + 1][256]
    __shared__ uint8_t path[32], best_path[32], qs[32];
    __shared__ float best_d, best_score;

    const int64_t blk = (int64_t) blockIdx.x + (int64_t) gridDim.x * blockIdx.y;
    if (blk >= nblk) return;
    load_codebook(lut, sh);
    const float* xb = x + blk * kBlock;
    const float* wb = w ? w + (w_block_mod > 0 ? (blk % w_block_mod) : blk) * kBlock : nullptr;
    for (int i = threadIdx.x; i < kBlock; i += blockDim.x) {
        xs[i] = xb[i];
        ws[i] = wb ? wb[i] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sumsq = 0.0f;
        for (int j = 0; j < kBlock; ++j) sumsq += xs[j] * xs[j];
        best_score = sumsq;
        best_d = sumsq > 0.0f ? sqrtf(sumsq / kBlock) : 0.0f;
        for (int t = 0; t < 32; ++t) best_path[t] = 0;
    }
    __syncthreads();
    if (!(best_score > 0.0f)) {   // all-zero block: d = 0, path = 0
        if (threadIdx.x == 0) {
            out[blk * kStageBytes + 0] = 0;
            out[blk * kStageBytes + 1] = 0;
            for (int i = 0; i < 32; ++i) out[blk * kStageBytes + 2 + i] = 0;
        }
        return;
    }

    const float* wcost = w ? ws : nullptr;
    float d = best_d;
    for (int pass = 0; pass < (passes < 1 ? 1 : passes); ++pass) {
        viterbi(sh, xs, wcost, d, path, dp, dp2, hist);
        // least-squares refit of d, then score at the fp16-rounded scale
        if (threadIdx.x == 0) {
            for (int i = 0; i < 32; ++i) qs[i] = path[i];
            float sxc = 0.0f, scc = 0.0f;
            for (int t = 0; t < 32; ++t) {
                const uint32_t s = (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & kStateMask;
                const uint32_t h = s * kGolden;
                const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
                const float c[4] = {sh[2 * p0], sh[2 * p0 + 1], sh[2 * p1], sh[2 * p1 + 1]};
                for (int j = 0; j < 4; ++j) {
                    const float wj = wcost ? wcost[4 * t + j] : 1.0f;
                    sxc += wj * xs[4 * t + j] * c[j];
                    scc += wj * c[j] * c[j];
                }
            }
            float dn = scc > 0.0f ? sxc / scc : d;
            float dh = half_to_float_dev(float_to_half_dev(dn));
            if (!(dh > 0.0f)) dh = dn;
            float num = 0.0f, den = 0.0f;
            for (int t = 0; t < 32; ++t) {
                const uint32_t s = (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & kStateMask;
                const uint32_t h = s * kGolden;
                const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
                const float c[4] = {sh[2 * p0], sh[2 * p0 + 1], sh[2 * p1], sh[2 * p1 + 1]};
                for (int j = 0; j < 4; ++j) {
                    const float wj = wcost ? wcost[4 * t + j] : 1.0f;
                    const float e = xs[4 * t + j] - dh * c[j];
                    num += wj * e * e;
                    den += wj * xs[4 * t + j] * xs[4 * t + j];
                }
            }
            refit[0] = dh;
            refit[1] = den > 0.0f ? num / den : num;
        }
        __syncthreads();
        const float dh = refit[0], score = refit[1];
        if (threadIdx.x == 0 && score < best_score) {
            best_score = score;
            best_d = dh;
            for (int i = 0; i < 32; ++i) best_path[i] = qs[i];
        }
        // the next lap pair runs at the refitted scale
        d = dh;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const uint16_t dh = float_to_half_dev(best_d);
        out[blk * kStageBytes + 0] = (uint8_t) (dh & 0xff);
        out[blk * kStageBytes + 1] = (uint8_t) (dh >> 8);
        for (int i = 0; i < 32; ++i) out[blk * kStageBytes + 2 + i] = best_path[i];
    }
}

void* g_lut_dev = nullptr;

}  // namespace

bool tq_encode_cuda_available() noexcept { return true; }
bool tq_encode_cuda_supported(int ggml_type) noexcept { return ggml_type == 144; }

bool tq_encode_rows_cuda(int ggml_type, const float* x, const float* w, bool w_row, int64_t rows, int64_t n,
                         int passes, uint8_t* out) {
    if (ggml_type != 144 || !x || rows <= 0 || n <= 0 || n % kBlock != 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (!g_lut_dev) {
        // The CPU module owns the codebook (same table the runtime decodes with); this copies it to the device.
        const float * cb = strata::kernels::cpu::tq_codebook();
        if (cudaMalloc(&g_lut_dev, 4096 * sizeof(float)) != cudaSuccess) return false;
        if (cudaMemcpy(g_lut_dev, cb, 4096 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    }
    const int64_t nblk = rows * (n / kBlock);
    const int64_t w_n = w ? (w_row ? n : rows * n) : 0;
    float* dx = nullptr;
    float* dw = nullptr;
    uint8_t* dout = nullptr;
    if (cudaMalloc(&dx, (size_t) rows * n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&dout, (size_t) nblk * kStageBytes) != cudaSuccess) { cudaFree(dx); return false; }
    if (w && cudaMalloc(&dw, (size_t) w_n * sizeof(float)) != cudaSuccess) {
        cudaFree(dx);
        cudaFree(dout);
        return false;
    }
    bool ok = cudaMemcpy(dx, x, (size_t) rows * n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok && w) ok = cudaMemcpy(dw, w, (size_t) w_n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        const int gx = (int) (nblk < 65535 ? nblk : 65535);
        const int gy = (int) ((nblk + gx - 1) / gx);
        const size_t shbytes = (4096 + 2 * kBlock + 2 * 256 + 2 + (kSteps + 1) * 256) * sizeof(float);
        static bool shset = false;
        if (!shset) {
            if (cudaFuncSetAttribute(tq2t_encode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) shbytes) != cudaSuccess) {
                std::fprintf(stderr, "tq_encode_rows_cuda: %s\n", cudaGetErrorString(cudaGetLastError()));
                cudaFree(dx);
                if (dw) cudaFree(dw);
                cudaFree(dout);
                return false;
            }
            shset = true;
        }
        tq2t_encode_kernel<<<dim3(gx, gy), 256, shbytes>>>(dx, dw, (const float*) g_lut_dev, nblk,
                                                          (w && w_row) ? (int) (n / kBlock) : 0, passes, dout);
        ok = cudaDeviceSynchronize() == cudaSuccess;
    }
    if (ok) ok = cudaMemcpy(out, dout, (size_t) nblk * kStageBytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) std::fprintf(stderr, "tq_encode_rows_cuda: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaFree(dx);
    if (dw) cudaFree(dw);
    cudaFree(dout);
    return ok;
}

}  // namespace strata::kernels
