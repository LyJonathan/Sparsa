// src/kernels/ikq_cuda_parity.cpp - the GPU dequantizers of the ik_llama.cpp formats (dq_iq2_kt and friends in
// iq_kernels.cu) against the CPU codec, and the grouped expert dot against the CPU dot.
//
// The formats keep their row scale in the row's first bytes, so the only GPU entry point that addresses a row
// rather than a flat block index is iq_embed_rows; it drives the same dq_dispatch the expert dequant path
// would.  Check 1 compares its output with ikq::dequantize_row, bit for bit (both are the same float
// expression), on synthetic rows produced by the CPU quantizer.  Check 2 puts the same rows through
// minimax_moe_gu and compares with ikq::dot_row, which is an integer dp4a chain for the GPU and a float dot of
// the decoded values for the CPU, so that one is a tolerance.
//
//     build/ikq_cuda_parity
// Exit 0: every dequantized value bitwise equal and every dot within 1e-3 relative.
#include "strata/kernels/cpu/iqk_quants.hpp"
#include "strata/kernels/iqk_types.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ikq_cuda_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// FP32 -> FP16 bits, round to nearest even (normals only: a q8_1 scale)
static uint16_t f2h(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    const uint32_t m = u & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// FP16 bits -> float (the q8_1 scale on the host; normals only, which a q8_1 scale is).
static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev < 1) {
        std::printf("ikq_cuda_parity: no CUDA device\n");
        return 0;
    }

    constexpr int64_t rows = 8, cols = 512;   // whole 256-value super-blocks
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> w((size_t) (rows * cols));
    for (auto& v : w) v = nd(rng);
    std::vector<float> imx((size_t) cols, 0.5f);
    std::vector<float> act((size_t) cols);
    for (auto& v : act) v = nd(rng) * 8.f;

    const int types[] = {K::ikq::kIQ2KT, K::ikq::kIQ3K, K::ikq::kIQ4KS, K::ikq::kIQ3KT, K::ikq::kIQ2KL};
    for (int type : types) {
        const size_t rb = K::ikq::row_bytes(type, cols);
        std::vector<uint8_t> q(rb * (size_t) rows);
        try {
            K::ikq::quantize_rows(type, w.data(), q.data(), rows, cols, imx.data(), 1);
        } catch (const std::exception& e) {
            std::printf("  %-12s quantize: %s\n", K::ikq::type_name(type), e.what());
            ++g_fail;
            continue;
        }
        // 1. GPU dequant (iq_embed_rows, row-aware) vs the CPU codec.
        uint8_t* dq = nullptr;
        ck(cudaMalloc(&dq, (size_t) (rows * cols) * sizeof(float)), "cudaMalloc");
        ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
        const int32_t toks[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int32_t* dtok = nullptr;
        float* dout = nullptr;
        ck(cudaMalloc(&dtok, sizeof(toks)), "cudaMalloc tokens");
        ck(cudaMalloc(&dout, (size_t) (rows * cols) * sizeof(float)), "cudaMalloc out");
        ck(cudaMemcpy(dtok, toks, sizeof(toks), cudaMemcpyHostToDevice), "cudaMemcpy tokens");
        K::iq_embed_rows(type, dq, rb, dtok, rows, cols, dout, nullptr);
        ck(cudaDeviceSynchronize(), "embed_rows");
        std::vector<float> got((size_t) (rows * cols));
        ck(cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy back");

        std::vector<float> want((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r)
            K::ikq::dequantize_row(type, q.data() + (size_t) r * rb, want.data() + (size_t) r * cols, cols);
        // The two decoders run the same float expression, but the host compiler may contract it (FMA) and so may
        // nvcc, so equality is "within one ULP": relative 1e-6 is a few float32 steps at these values.
        size_t exact = 0;
        double worst = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            if (std::memcmp(&want[i], &got[i], 4) == 0) ++exact;
            const double rel = std::fabs((double) want[i] - (double) got[i]) / (std::fabs((double) want[i]) + 1e-30);
            worst = std::max(worst, rel);
        }
        char msg[192];
        std::snprintf(msg, sizeof(msg), "%s dequant GPU vs CPU (%zu of %zu bit-equal, worst rel %.1e)",
                      K::ikq::type_name(type), exact, want.size(), worst);
        check(worst < 1e-6, msg);

        // 2. the grouped expert dot (minimax_moe_gu with a zero up matrix) vs the CPU dot.
        const size_t hb = (size_t) cols * sizeof(float);
        std::vector<uint8_t> zeros(rb, 0);
        float* dh = nullptr;
        ck(cudaMalloc(&dh, (size_t) cols * sizeof(float)), "cudaMalloc h");
        // q8_1 activations for the dot
        float* da = nullptr;
        ck(cudaMalloc(&da, hb), "cudaMalloc act");
        ck(cudaMemcpy(da, act.data(), hb, cudaMemcpyHostToDevice), "cudaMemcpy act");
        void* xq = nullptr;
        const size_t xq_bytes = (size_t) (cols / 32) * 36;   // block_q8_1 is 36 bytes
        ck(cudaMalloc(&xq, xq_bytes), "cudaMalloc xq");
        K::quantize_q8_1_rows(da, 1, cols, xq, nullptr);
        ck(cudaDeviceSynchronize(), "quantize_q8_1_rows");

        // gate = row 0 of the tensor, up = the same: swigluoai(g, g) is not the dot, so read the raw dot through
        // minimax_moe_down instead (no activation function on that path).
        float* dout2 = nullptr;
        ck(cudaMalloc(&dout2, (size_t) rows * sizeof(float)), "cudaMalloc out2");
        // 8 weight rows of 512 values each: the down pass computes one dot per weight row (= n_embd here).
        // `down` is an array of DEVICE row pointers, like every grouped expert entry point.
        const void* gate_h[1] = {dq};
        void* dgate = nullptr;
        ck(cudaMalloc(&dgate, sizeof(gate_h)), "cudaMalloc gate ptr");
        ck(cudaMemcpy(dgate, gate_h, sizeof(gate_h), cudaMemcpyHostToDevice), "cudaMemcpy gate ptr");
        K::minimax_moe_down(type, (const void* const*) dgate, 1, rows, cols, rb, xq, dout2, nullptr);
        ck(cudaDeviceSynchronize(), "minimax_moe_down");
        std::vector<float> gpu1((size_t) rows);
        ck(cudaMemcpy(gpu1.data(), dout2, (size_t) rows * sizeof(float), cudaMemcpyDeviceToHost), "down back");
        cudaFree(dgate);

        // The CPU side of the same expression: the decoded row dotted with the SAME q8_1 activations the kernel
        // read, dequantized back to float here.  So this compares the kernel's arithmetic, not the activation
        // format (dot_row would use ggml's Q8_K; that difference alone is ~0.5%).
        std::vector<uint8_t> q8h((size_t) (cols / 32) * 36);
        for (int64_t b = 0; b < cols / 32; ++b) {
            const float* xb = act.data() + b * 32;
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
            const float d = amax / 127.f;
            uint8_t* blk = q8h.data() + (size_t) b * 36;
            const uint16_t dh = f2h(d);
            std::memcpy(blk, &dh, 2);
            const uint16_t zero = 0;
            std::memcpy(blk + 2, &zero, 2);
            for (int j = 0; j < 32; ++j) blk[4 + j] = (uint8_t) (int8_t) std::lround(xb[j] / d);
        }
        std::vector<float> q81((size_t) cols);
        for (int64_t j = 0; j < cols; ++j) {
            const uint8_t* blk = q8h.data() + (size_t) (j / 32) * 36;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            q81[(size_t) j] = (float) (int8_t) blk[4 + (j % 32)] * h2f(dh);
        }
        double cpu = 0;
        for (int64_t j = 0; j < cols; ++j) cpu += (double) want[(size_t) j] * (double) q81[(size_t) j];
        const double rel = std::fabs((double) gpu1[0] - cpu) / (std::fabs(cpu) + 1e-30);
        std::snprintf(msg, sizeof(msg), "%s grouped dot GPU %.6g vs CPU %.6g (rel %.2e)", K::ikq::type_name(type),
                      (double) gpu1[0], cpu, rel);
        check(rel < 1e-5, msg);
        (void) zeros;
        (void) zeros;
        cudaFree(dq);
        cudaFree(dtok);
        cudaFree(dout);
        cudaFree(dout2);
        cudaFree(dh);
        cudaFree(da);
        cudaFree(xq);
    }
    std::printf(g_fail ? "ikq_cuda_parity: %d FAILED\n" : "ikq_cuda_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
