#pragma once

// GPTQ-style error-feedback rounding, ported from Project Maya (MIT, see
// THIRD_PARTY_NOTICES.md): tools/maya_quant/gptq.py in github.com/mw00/project-maya.
//
// The quantizer is llama.cpp's own (ggml_quantize_chunk), so the bytes stay an
// ordinary GGUF row. What changes is *which* columns the quantizer sees and what
// happens to the error it leaves behind:
//
//   * a row is walked in blocks of 256 values - the block size every IQ type and
//     every K-quant is built on, so a block is exactly one superblock;
//   * inside a block the importance is the diagonal of the inverse of the block's
//     inverse-Hessian, i.e. how much an error on that column still costs once the
//     columns to its left are fixed (not the plain activation energy);
//   * the block's rounding error is then pushed into the block's not-yet-quantized
//     columns through the upper Cholesky factor of the inverse Hessian - the
//     lazy-batch form of GPTQ (Frantar et al. 2022) with the block's columns
//     removed jointly;
//   * every row is independent, so gate and up can share one Hessian and are still
//     quantized independently (each row's update only touches that row).
//
// H = sum_t w_t^2 x_t x_t^T over the tokens routed to the expert, plus `prior`
// tokens' worth of the layer-wide input Hessian so a rarely routed expert is not
// fitted to a handful of tokens (Maya's shrinkage), and the diagonal is lifted by
// `damp` * mean(diag). Dead channels first take that mean, which keeps H positive
// definite for a rank-limited calibration set.

#include <ggml.h>

#include "quantize.hpp"
#include "strata/kernels/cpu/iqk_quants.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mmq {
namespace gptq {

constexpr int64_t kBlock = 256;   // one superblock of every IQ / K-quant type

inline void parallel_for(int64_t n, int threads, const std::function<void(int64_t, int64_t)> & work) {
    if (threads < 1 || n <= 0) {
        return;
    }
    if (threads == 1 || n == 1) {
        work(0, n);
        return;
    }
    if (static_cast<int64_t>(threads) > n) {
        threads = static_cast<int>(n);
    }
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> err(static_cast<size_t>(threads));
    const int64_t chunk = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int64_t a = static_cast<int64_t>(t) * chunk;
        if (a >= n) {
            break;
        }
        const int64_t b = std::min(n, a + chunk);
        pool.emplace_back([&, t, a, b]() {
            try {
                work(a, b);
            } catch (...) {
                err[static_cast<size_t>(t)] = std::current_exception();
            }
        });
    }
    for (std::thread & th : pool) {
        th.join();
    }
    for (const std::exception_ptr & e : err) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

/// Lower Cholesky factor in place: A (n x n, row-major, symmetric) -> L with A = L L^T.
/// Right-looking and blocked: the diagonal panel is factored column by column, its rows
/// are solved for the trailing rows, then the trailing block takes one rank-panel update.
inline void chol_lower(float * a, int64_t n, int threads) {
    constexpr int64_t kB = 128;
    for (int64_t kb = 0; kb < n; kb += kB) {
        const int64_t jb = std::min(kB, n - kb);
        for (int64_t j = kb; j < kb + jb; ++j) {
            double s = a[j * n + j];
            for (int64_t k = kb; k < j; ++k) {
                const double v = a[j * n + k];
                s -= v * v;
            }
            if (!(s > 0.0)) {
                s = 1e-20;   // damping makes this unreachable; never emit a NaN
            }
            const float d = static_cast<float>(std::sqrt(s));
            a[j * n + j] = d;
            const float inv = 1.f / d;
            for (int64_t i = j + 1; i < kb + jb; ++i) {
                double t = a[i * n + j];
                for (int64_t k = kb; k < j; ++k) {
                    t -= static_cast<double>(a[i * n + k]) * a[j * n + k];
                }
                a[i * n + j] = static_cast<float>(t) * inv;
            }
        }
        const int64_t rest = n - (kb + jb);
        if (rest <= 0) {
            continue;
        }
        parallel_for(rest, threads, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const int64_t i = kb + jb + r;
                float * row = a + i * n;
                for (int64_t t = 0; t < jb; ++t) {
                    double s = row[kb + t];
                    for (int64_t u = 0; u < t; ++u) {
                        s -= static_cast<double>(row[kb + u]) * a[(kb + t) * n + kb + u];
                    }
                    row[kb + t] = static_cast<float>(s) / a[(kb + t) * n + kb + t];
                }
            }
        });
        parallel_for(rest, threads, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const int64_t i = kb + jb + r;
                float * row = a + i * n;
                for (int64_t c = 0; c <= r; ++c) {
                    const float * orow = a + (kb + jb + c) * n;
                    double s = 0.0;
                    for (int64_t t = 0; t < jb; ++t) {
                        s += static_cast<double>(row[kb + t]) * orow[kb + t];
                    }
                    row[kb + jb + c] -= static_cast<float>(s);
                }
            }
        });
    }
}

/// Lower-triangular inverse of L into `out` (lower), column by column. `l` is read only, so the
/// columns cannot race with each other (an in-place version read entries a neighbouring column had
/// already overwritten, which showed up as a 0.15 absolute error on a 256x256 check).
inline void tri_inv_lower(const float * l, int64_t n, float * out, int threads) {
    std::memset(out, 0, static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float));
    parallel_for(n, threads, [&](int64_t c0, int64_t c1) {
        for (int64_t j = c0; j < c1; ++j) {
            const double d = l[j * n + j];
            if (!(std::fabs(d) > 0.0)) {
                throw std::runtime_error("gptq: singular Cholesky factor");
            }
            out[j * n + j] = static_cast<float>(1.0 / d);
            for (int64_t i = j + 1; i < n; ++i) {
                const double di = l[i * n + i];
                double s = 0.0;
                for (int64_t k = j; k < i; ++k) {
                    s += static_cast<double>(l[i * n + k]) * out[k * n + j];
                }
                out[i * n + j] = static_cast<float>(-s / di);
            }
        }
    });
}

/// Row dequantizer for any type quantize_rows produces (ggml's, or the ported ik formats).
inline void dequant_row(ggml_type type, const void * src, float * dst, int64_t n) {
    if (is_ikq(type)) {
        strata::kernels::ikq::dequantize_row(static_cast<int>(type), src, dst, n);
        return;
    }
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
        return;
    }
    const ggml_type_traits * t = ggml_get_type_traits(type);
    if (!t || !t->to_float) {
        throw std::runtime_error("gptq: no dequantizer for the block type");
    }
    t->to_float(src, dst, n);
}

/// One block's inverse-Hessian diagonal importance, as Maya computes it:
/// diag(inv(Hinv_SS)) = diag(Uss^-1 Uss^-T), normalized to mean 1, plus W = (M_SS)^-1
/// (lower) for the error update. M = L^-1 is the lower inverse of chol(H), U = M^T.
inline void block_factors(const float * m, int64_t n, int64_t b0, int64_t nb, std::vector<double> & w,                          std::vector<float> & imp) {
    w.assign(static_cast<size_t>(nb) * static_cast<size_t>(nb), 0.0);
    for (int64_t c = 0; c < nb; ++c) {
        const double d = m[(b0 + c) * n + b0 + c];
        w[static_cast<size_t>(c * nb + c)] = 1.0 / d;
        for (int64_t r = c + 1; r < nb; ++r) {
            double s = 0.0;
            for (int64_t k = c; k < r; ++k) {
                s += static_cast<double>(m[(b0 + r) * n + b0 + k]) * w[static_cast<size_t>(k * nb + c)];
            }
            w[static_cast<size_t>(r * nb + c)] = -s / d;
        }
    }
    imp.assign(static_cast<size_t>(nb), 0.f);
    double mean = 0.0;
    for (int64_t j = 0; j < nb; ++j) {
        double s = 0.0;
        for (int64_t k = j; k < nb; ++k) {
            const double v = w[static_cast<size_t>(k * nb + j)];
            s += v * v;
        }
        imp[static_cast<size_t>(j)] = static_cast<float>(s);
        mean += s;
    }
    mean /= static_cast<double>(nb);
    if (!(mean > 0.0)) {
        mean = 1.0;
    }
    for (float & v : imp) {
        v = static_cast<float>(std::max(1e-20, static_cast<double>(v) / mean));
    }
}

/// Build M = L^-1 (the lower inverse of chol(H)) from H in place, with Maya's damping.
/// Call once per Hessian and share it between gate and up (both rows of one expert use the same H).
inline void factor(float * h, int64_t n, double damp, int threads) {
    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        mean += h[i * n + i];
    }
    mean /= static_cast<double>(n);
    if (!(mean > 0.0)) {
        mean = 1.0;
    }
    for (int64_t i = 0; i < n; ++i) {
        float & d = h[i * n + i];
        if (!(d > 0.f)) {
            d = static_cast<float>(mean);
        }
        d += static_cast<float>(damp * mean);
        if (!std::isfinite(d) || d <= 0.f) {
            throw std::runtime_error("gptq: Hessian diagonal is not positive after damping");
        }
    }
    chol_lower(h, n, threads);
    std::vector<float> m(static_cast<size_t>(n) * static_cast<size_t>(n), 0.f);
    tri_inv_lower(h, n, m.data(), threads);
    std::memcpy(h, m.data(), m.size() * sizeof(float));
}

/// Round `rows` rows of length n (row-major f32) with error feedback, given a prepared `m` from
/// factor(). `dst` gets row_size(type, n) bytes a row.
inline size_t round_rows_prepared(ggml_type type, float * w, int64_t rows, int64_t n, const float * m,
                                  void * dst, int threads, double * out_seconds = nullptr) {
    if (n % kBlock != 0) {
        throw std::runtime_error("gptq: row length " + std::to_string(n) + " is not a multiple of 256");
    }
    const auto t_start = std::chrono::steady_clock::now();
    const float * h = m;
    const size_t rq = row_size(type, kBlock);
    const size_t row_bytes = row_size(type, n);
    std::vector<double> wss;
    std::vector<float> imp;
    std::vector<float> wb(static_cast<size_t>(rows) * static_cast<size_t>(kBlock));
    std::vector<float> qb(static_cast<size_t>(rows) * static_cast<size_t>(kBlock));
    std::vector<float> t1(static_cast<size_t>(rows) * static_cast<size_t>(kBlock));
    std::vector<uint8_t> q(static_cast<size_t>(rows) * rq);
    for (int64_t b0 = 0; b0 < n; b0 += kBlock) {
        const int64_t nb = std::min(kBlock, n - b0);
        const int64_t b1 = b0 + nb;
        block_factors(h, n, b0, nb, wss, imp);
        for (int64_t r = 0; r < rows; ++r) {
            std::memcpy(wb.data() + r * nb, w + r * n + b0, sizeof(float) * static_cast<size_t>(nb));
        }
        quantize_rows(type, wb.data(), q.data(), rows, nb, imp.data(), threads);
        uint8_t * out = static_cast<uint8_t *>(dst);
        for (int64_t r = 0; r < rows; ++r) {
            dequant_row(type, q.data() + static_cast<size_t>(r) * rq, qb.data() + r * nb, nb);
            std::memcpy(out + static_cast<size_t>(r) * row_bytes + static_cast<size_t>(b0 / kBlock) * rq,
                        q.data() + static_cast<size_t>(r) * rq, rq);
        }
        if (b1 >= n) {
            break;
        }
        // T1 = E @ Uss^-1 = E @ W^T: T1[r][k] = sum_{j<=k} E[r][j] * W[k][j]
        for (int64_t r = 0; r < rows; ++r) {
            const float * wr = wb.data() + r * nb;
            const float * qr = qb.data() + r * nb;
            float * t = t1.data() + r * nb;
            for (int64_t k = 0; k < nb; ++k) {
                double s = 0.0;
                for (int64_t j = 0; j <= k; ++j) {
                    s += (static_cast<double>(wr[j]) - qr[j]) * wss[static_cast<size_t>(k * nb + j)];
                }
                t[k] = static_cast<float>(s);
            }
        }
        // W[:, b1:] -= T1 @ U[b0:b1, b1:],  U[k][c] = M[b1+c][b0+k]
        const int64_t rest = n - b1;
        parallel_for(rows, threads, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const float * t = t1.data() + r * nb;
                float * row = w + r * n;
                for (int64_t c = 0; c < rest; ++c) {
                    const float * mc = h + (b1 + c) * n + b0;
                    double s = 0.0;
                    for (int64_t k = 0; k < nb; ++k) {
                        s += static_cast<double>(t[k]) * mc[k];
                    }
                    row[b1 + c] -= static_cast<float>(s);
                }
            }
        });
    }
    if (out_seconds) {
        *out_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    }
    return static_cast<size_t>(rows) * row_bytes;
}

/// Convenience: factor H and round in one call (gate and up each pay their own factorisation).
inline size_t round_rows(ggml_type type, float * w, int64_t rows, int64_t n, float * h, void * dst,
                         double damp, int threads, double * out_seconds = nullptr) {
    factor(h, n, damp, threads);
    return round_rows_prepared(type, w, rows, n, h, dst, threads, out_seconds);
}

}  // namespace gptq
}  // namespace mmq
