// src/kernels/cpu/tq_encode_parity.cpp - the acceptance test of the trellis ENCODER (tq_encode.hpp).
//
// 1. FORMAT: encode -> decode must reproduce the encoder's own reconstruction exactly, and the decode must equal a
//    verbatim port of the upstream decoder (agentionai/llama.cpp ggml/src/ggml-quants.c, MIT, the ggml authors;
//    dequantize_row_tq2_t / dequantize_row_tqk + tq2t_hyb_index + tqk_state) to the bit.
// 2. ENGINE: the same blocks through the shipped CPU decode-and-dot kernel (tq_rows) must agree with the reference
//    decode (one-hot Q8_0 activations read one weight out of the kernel at a time).
// 3. QUALITY: on Gaussian and on a per-16-scaled NVFP4-like source, the Viterbi encoder must beat the placeholder
//    encoder (a port of quantize_row_tq2_t_ref) on imatrix-weighted relative L2.
// 4. HADAMARD: with a folded weight the layer output W x must equal decode(encode(W diag(s) H^T)) applied to
//    H (s * x), i.e. the rotation convention of hadamard.hpp and tq.cuh.
#include "strata/kernels/cpu/tq_encode.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "ggml-common-tq.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::kernels::cpu;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// -------- verbatim port of the upstream decoder (agentionai/llama.cpp, MIT) --------------------------------
constexpr uint32_t kStateMask = (1u << 15) - 1u;

uint32_t tq2t_hyb_index(uint32_t e) {
    const uint32_t x = ((e >> 1) & kStateMask) * 0x9e3779b1u;
    return (e & 1) ? ((x >> 10) & 2047u) : (x >> 21);
}

uint32_t tqk_state(const uint8_t* qs, int k, int t) {
    const uint32_t nbytes = 4u * (uint32_t) k;
    const uint32_t off = (uint32_t) (31 - t) * (uint32_t) k;
    const uint32_t i0 = off >> 3;
    const uint32_t i1 = (i0 + 1) % nbytes;
    const uint32_t i2 = (i0 + 2) % nbytes;
    const uint32_t w = (uint32_t) qs[i0] | ((uint32_t) qs[i1] << 8) | ((uint32_t) qs[i2] << 16);
    return (w >> (off & 7)) & 0xffffu;
}

float f16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, b;
    if (e == 0x1fu) {
        b = sign | 0x7f800000u | (m << 13);
    } else if (e) {
        b = sign | ((e + 112u) << 23) | (m << 13);
    } else if (!m) {
        b = sign;
    } else {
        const float v = (float) m * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}

uint16_t to_half(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
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

void upstream_dequant(int type, const uint8_t* blk, float* y) {
    const int kb = type == 144 ? 8 : type == 145 ? 6 : 7;
    const float d = f16((uint16_t) (blk[0] | (blk[1] << 8)));
    const uint8_t* qs = blk + 2;
    for (int t = 0; t < 32; ++t) {
        const uint32_t s = kb == 8 ? (uint32_t) ((uint32_t) qs[(t + 31) % 32] << 8 | qs[t]) : tqk_state(qs, kb, t);
        for (int h = 0; h < 2; ++h) {
            const uint32_t idx = tq2t_hyb_index(2 * s + h);
            y[4 * t + 2 * h + 0] = d * f16(tq_lut_f16[2 * idx + 0]);
            y[4 * t + 2 * h + 1] = d * f16(tq_lut_f16[2 * idx + 1]);
        }
    }
}

// -------- verbatim port of the upstream PLACEHOLDER encoder (quantize_row_tq2_t_ref) ----------------------

float tq2t_lut_val(uint32_t i) { return f16(tq_lut_f16[i]); }

float tq2t_step_err(const float* y, int t, uint32_t a, uint32_t b) {
    const uint32_t s = (a << 8) | b;
    float err = 0.0f;
    for (int h = 0; h < 2; ++h) {
        const uint32_t idx = tq2t_hyb_index(2 * s + h);
        const float dx = y[4 * t + 2 * h + 0] - tq2t_lut_val(2 * idx + 0);
        const float dy = y[4 * t + 2 * h + 1] - tq2t_lut_val(2 * idx + 1);
        err += dx * dx + dy * dy;
    }
    return err;
}

/// Greedy coordinate descent over the 32 bytes (four sweeps), then a least-squares scale: the placeholder.
void placeholder_tq2t(const float* x, uint8_t* blk) {
    std::memset(blk, 0, 34);
    float sumsq = 0.0f;
    for (int j = 0; j < 128; ++j) sumsq += x[j] * x[j];
    if (sumsq == 0.0f) return;
    const float rms = std::sqrt(sumsq / 128.0f);
    float yn[128];
    for (int j = 0; j < 128; ++j) yn[j] = x[j] / rms;
    uint8_t* qs = blk + 2;
    for (int pass = 0; pass < 4; ++pass) {
        for (int t = 0; t < 32; ++t) {
            const uint32_t prev = qs[(t + 31) % 32];
            const uint32_t next = qs[(t + 1) % 32];
            float best = INFINITY;
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < 256; ++c) {
                const float e = tq2t_step_err(yn, t, prev, c) + tq2t_step_err(yn, (uint32_t) ((t + 1) % 32), c, next);
                if (e < best) {
                    best = e;
                    best_c = c;
                }
            }
            qs[t] = (uint8_t) best_c;
        }
    }
    float sxc = 0.0f, scc = 0.0f;
    for (int t = 0; t < 32; ++t) {
        const uint32_t s = ((uint32_t) qs[(t + 31) % 32] << 8) | qs[t];
        for (int j = 0; j < 4; ++j) {
            const float c = tq2t_lut_val(2 * tq2t_hyb_index(2 * s + (j >> 1)) + (j & 1));
            sxc += x[4 * t + j] * c;
            scc += c * c;
        }
    }
    const uint16_t dh = to_half(scc > 0.0f ? sxc / scc : rms);
    blk[0] = (uint8_t) (dh & 0xff);
    blk[1] = (uint8_t) (dh >> 8);
}

// -------- helpers ------------------------------------------------------------------------------------------

/// A per-16 NVFP4-like block: the E2M1 codebook times a per-16 scale, the scale spread over the block.
void nvfp4_like_block(std::mt19937& rng, float* x) {
    static const float e2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::uniform_int_distribution<int> q(0, 7), sg(0, 1);
    std::normal_distribution<float> g(0.f, 1.f);
    for (int g16 = 0; g16 < 8; ++g16) {
        const float scale = std::fabs(g(rng)) * 0.5f + 0.05f;
        for (int j = 0; j < 16; ++j) x[g16 * 16 + j] = e2m1[q(rng)] * scale * (sg(rng) ? -1.f : 1.f);
    }
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>* w) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double wi = w ? (*w)[i] : 1.0;
        const double e = (double) a[i] - (double) b[i];
        num += wi * e * e;
        den += wi * (double) a[i] * a[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    const int types[3] = {kTq2T, kTqk6, kTqk7};
    TqEncodeParams params;   // passes = 2, uniform weights

    if (argc > 1 && std::string(argv[1]) == "--bench") {
        // throughput of one core and of `nthreads` cores: this is what a CPU-only tier costs
        const int nthreads = argc > 2 ? std::atoi(argv[2]) : 12;
        const int nblocks = argc > 3 ? std::atoi(argv[3]) : 512;
        std::mt19937 rng(5);
        std::normal_distribution<float> g(0.f, 1.f);
        for (int ti = 0; ti < 3; ++ti) {
            const int t = types[ti];
            const int n = 128 * nblocks;
            std::vector<float> x((size_t) n);
            for (auto& v : x) v = g(rng);
            std::vector<uint8_t> q(tq_encode_row_bytes(t, n));
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            const int chunk = nblocks / nthreads;
            for (int w = 0; w < nthreads; ++w) {
                th.emplace_back([&, w]() {
                    const int b0 = w * chunk;
                    const int b1 = w + 1 == nthreads ? nblocks : b0 + chunk;
                    tq_encode_row(t, x.data() + (size_t) b0 * 128, nullptr, (int64_t) (b1 - b0) * 128, params,
                                  q.data() + (size_t) b0 * tq_encode_block_bytes(t));
                });
            }
            for (auto& c : th) c.join();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%-6s %d blocks on %d threads: %.2f s, %.3f Mweights/s, %.2f GB/s of f32 in\n",
                        tq_encode_type_name(t), nblocks, nthreads, sec, (double) n / sec / 1e6,
                        (double) n * 4 / sec / 1e9);
        }
        return 0;
    }

    // 1. format round trip + upstream decode agreement ------------------------------------------------
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> g(0.f, 1.f);
        const int n = 128 * 8;
        std::vector<float> x((size_t) n), y((size_t) n), yu((size_t) n);
        for (auto& v : x) v = g(rng);
        for (int ti = 0; ti < 3; ++ti) {
            const int t = types[ti];
            std::vector<uint8_t> q(tq_encode_row_bytes(t, n));
            tq_encode_row(t, x.data(), nullptr, n, params, q.data());
            tq_decode_row(t, q.data(), n, y.data());
            const size_t bsz = tq_encode_block_bytes(t);
            for (int b = 0; b < n / 128; ++b) {
                upstream_dequant(t, q.data() + (size_t) b * bsz, yu.data() + (size_t) b * 128);
            }
            check(std::memcmp(y.data(), yu.data(), (size_t) n * sizeof(float)) == 0,
                  std::string(tq_encode_type_name(t)) + ": reference decode == upstream decoder, bit for bit");
            const double e = rel_l2(x, y, nullptr);
            std::printf("      n=%d  rel L2 = %.5f  %zu bytes  %.4f bpw\n", n, e, q.size(),
                        8.0 * (double) q.size() / n);
            check(e > 0.02 && e < 0.55, std::string(tq_encode_type_name(t)) + ": reconstruction error in range");
        }
    }

    // 2. the engine's CPU decode-and-dot kernel agrees with the reference decode ------------------------
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> g(0.f, 1.f);
        const int n = 256;
        std::vector<float> x((size_t) n), y((size_t) n);
        for (auto& v : x) v = g(rng);
        for (int ti = 0; ti < 3; ++ti) {
            const int t = types[ti];
            std::vector<uint8_t> q(tq_encode_row_bytes(t, n));
            tq_encode_row(t, x.data(), nullptr, n, params, q.data());
            tq_decode_row(t, q.data(), n, y.data());
            double worst = 0.0;
            std::vector<float> got(1);
            const int nb32 = n / 32;
            std::vector<uint8_t> act((size_t) nb32 * 34, 0);
            const uint16_t dh = to_half(1.0f / 127.0f);
            for (int pos = 0; pos < n; ++pos) {
                // Q8_0: one block of 32 values per 34 bytes; d = 1/127 and q = 127 gives a single 1.0
                std::fill(act.begin(), act.end(), (uint8_t) 0);
                const int B = pos / 32;
                act[(size_t) B * 34 + 0] = (uint8_t) (dh & 0xff);
                act[(size_t) B * 34 + 1] = (uint8_t) (dh >> 8);
                act[(size_t) B * 34 + 2 + pos % 32] = 127;
                const void* a[1] = {act.data()};
                float* o[1] = {got.data()};
                tq_rows(TqIsa::scalar, t, q.data(), tq_encode_row_bytes(t, n), n, a, 1, o, 0, 1);
                const double diff = std::fabs((double) got[0] - (double) y[(size_t) pos]);
                const double tol = std::fabs((double) y[(size_t) pos]) * 3e-4 + 2e-5;
                if (diff > worst) worst = diff;
                if (diff > tol) {
                    std::printf("      %s state %d: kernel %.8g vs reference %.8g\n", tq_encode_type_name(t), pos,
                                got[0], y[(size_t) pos]);
                }
            }
            check(worst < 1e-3,
                  std::string(tq_encode_type_name(t)) + ": engine tq_rows kernel matches the reference decode");
        }
    }

    // 3. quality: Viterbi vs the upstream placeholder ---------------------------------------------------
    {
        std::printf("\n  type   source          placeholder   viterbi    gain\n");
        for (int src = 0; src < 2; ++src) {
            std::mt19937 rng(7 + src);
            std::normal_distribution<float> g(0.f, 1.f);
            const int nblocks = 64, n = 128 * nblocks;
            std::vector<float> x((size_t) n), w((size_t) n), y((size_t) n), yp((size_t) n);
            for (int b = 0; b < nblocks; ++b) {
                if (src == 0) {
                    for (int j = 0; j < 128; ++j) x[(size_t) b * 128 + j] = g(rng);
                } else {
                    nvfp4_like_block(rng, x.data() + (size_t) b * 128);
                }
                double s = 0;
                for (int j = 0; j < 128; ++j) s += std::fabs(x[(size_t) b * 128 + j]);
                for (int j = 0; j < 128; ++j) w[(size_t) b * 128 + j] = (float) (0.2 + std::fabs(x[(size_t) b * 128 + j]) / (s / 128));
            }
            {
                std::vector<uint8_t> q(tq_encode_row_bytes(kTq2T, n));
                tq_encode_row(kTq2T, x.data(), nullptr, n, params, q.data());
                tq_decode_row(kTq2T, q.data(), n, y.data());
                for (int b = 0; b < nblocks; ++b) {
                    std::vector<uint8_t> blk(34, 0);
                    placeholder_tq2t(x.data() + (size_t) b * 128, blk.data());
                    upstream_dequant(kTq2T, blk.data(), yp.data() + (size_t) b * 128);
                }
                const double ep = rel_l2(x, yp, &w), ev = rel_l2(x, y, &w);
                std::printf("  TQ2_T  %-14s %11.5f %10.5f %6.2fx\n", src == 0 ? "gaussian" : "nvfp4-per16", ep, ev,
                            ep / ev);
                check(ev < ep, "TQ2_T: the Viterbi encoder beats the placeholder encoder");
            }
            for (int ti = 1; ti < 3; ++ti) {
                const int t = types[ti];
                std::vector<uint8_t> q(tq_encode_row_bytes(t, n));
                tq_encode_row(t, x.data(), nullptr, n, params, q.data());
                tq_decode_row(t, q.data(), n, y.data());
                std::printf("  %-6s %-14s %11s %10.5f\n", tq_encode_type_name(t), src == 0 ? "gaussian" : "nvfp4-per16",
                            "-", rel_l2(x, y, &w));
            }
        }
    }

    // 4. Hadamard folding: the runtime's y = H (s * x) reproduces W x ------------------------------------
    {
        std::mt19937 rng(4242);
        std::normal_distribution<float> g(0.f, 1.f);
        const int rows = 16, cols = 256;
        std::vector<float> W((size_t) rows * cols), Wr((size_t) rows * cols);
        for (auto& v : W) v = g(rng);
        Wr = W;
        const std::vector<float> signs = tq_hadamard_signs(cols, 3);
        for (int r = 0; r < rows; ++r) tq_hadamard_rotate(Wr.data() + (size_t) r * cols, cols, 128, signs.data());

        std::vector<float> xin((size_t) cols);
        for (auto& v : xin) v = g(rng);
        std::vector<float> xr = xin;
        tq_hadamard_rotate(xr.data(), cols, 128, signs.data());

        std::vector<uint8_t> q(tq_encode_row_bytes(kTq2T, cols));
        std::vector<float> recon((size_t) cols);
        double num = 0, den = 0, num_plain = 0;
        for (int r = 0; r < rows; ++r) {
            tq_encode_row(kTq2T, Wr.data() + (size_t) r * cols, nullptr, cols, params, q.data());
            tq_decode_row(kTq2T, q.data(), cols, recon.data());
            double a = 0, b = 0, pf = 0;
            for (int c = 0; c < cols; ++c) {
                a += (double) W[(size_t) r * cols + c] * xin[(size_t) c];
                b += (double) recon[(size_t) c] * xr[(size_t) c];
                pf += (double) recon[(size_t) c] * xin[(size_t) c];   // the WRONG way: no activation rotation
            }
            num += (a - b) * (a - b);
            num_plain += (a - pf) * (a - pf);
            den += a * a;
        }
        const double e = std::sqrt(num / den);
        const double e_plain = std::sqrt(num_plain / den);
        std::printf("\n  folded layer output relative error = %.5f  (W x vs decode(W') . (H (s * x)))\n", e);
        std::printf("  same weights, activation NOT rotated = %.5f  (must be far worse)\n", e_plain);
        check(e < 0.55 && e < 0.8 * e_plain, "Hadamard folding convention reproduces the unrotated layer");
    }

    std::printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "OK", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
