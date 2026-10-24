// src/kernels/cpu/tq_encode.cpp - see include/strata/kernels/cpu/tq_encode.hpp.  Plain C++ (baseline x86-64), the
// reference encoder: the CUDA backend (src/kernels/cuda/tq_encode.cu) writes the same blocks with the same search.
#include "strata/kernels/cpu/tq_encode.hpp"

#include "ggml-common-tq.h"

#include <cmath>
#include <cstring>
#include <limits>
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace strata::kernels::cpu {
namespace {

// ----------------------------------------------------------------------------- format geometry

constexpr int kBlock = 128;      // weights per trellis block
constexpr int kSteps = 32;       // trellis steps per block
constexpr int kPoints = 2048;    // codebook points (each point is two weights)
constexpr uint32_t kGolden = 0x9e3779b1u;
constexpr uint32_t kStateMask = 0x7fffu;

int bits_of(int t) {
    return t == kTq2T ? 8 : t == kTqk6 ? 6 : t == kTqk7 ? 7 : 0;
}
/// Chunks a 16-bit step window spans: 1 for TQ2_T (two bytes are the window), 2 for TQK (three chunks).
int window_chunks(int bits) { return bits == 8 ? 1 : 2; }
inline uint32_t alphabet(int bits) { return 1u << bits; }

// ----------------------------------------------------------------------------- fp16

inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, b;
    if (exp == 0x1fu) {
        b = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        b = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        b = sign;
    } else {
        const float m = (float) man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline uint16_t float_to_half(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int32_t e = (int32_t) ((b >> 23) & 0xffu) - 127;
    uint32_t man = b & 0x7fffffu;
    if ((b & 0x7fffffffu) == 0) return (uint16_t) sign;
    if (e > 15) return (uint16_t) (sign | 0x7c00u);                       // overflow -> inf
    if (e < -24) return (uint16_t) sign;                                  // underflow -> 0
    if (e < -14) {                                                        // subnormal
        const uint32_t shift = (uint32_t) (-e - 14 + 1);
        uint32_t m = (man | 0x800000u) >> (13 + shift);
        const uint32_t half = (man | 0x800000u) >> (12 + shift);
        if (half & 1u) ++m;                                               // round to nearest even
        return (uint16_t) (sign | m);
    }
    uint32_t m = man >> 13;
    const uint32_t rest = man & 0x1fffu;
    const uint32_t half = (man >> 12) & 1u;
    if (rest > 0x1000u || (rest == 0x1000u && half)) {
        ++m;
        if (m == 0x400u) {
            m = 0;
            if (++e > 15) return (uint16_t) (sign | 0x7c00u);
        }
    }
    return (uint16_t) (sign | ((uint32_t) (e + 15) << 10) | m);
}

// ----------------------------------------------------------------------------- codebook, common tables

struct Tables {
    float lut[4096];          // the 2048-point codebook as floats, point p = (lut[2p], lut[2p+1])
    float l2[32768];          // sum of the four codebook values squared, per 15-bit state
    Tables() {
        for (int i = 0; i < 4096; ++i) lut[i] = half_to_float(tq_lut_f16[i]);
        for (uint32_t s = 0; s <= kStateMask; ++s) {
            const uint32_t x = s * kGolden;
            const float a0 = lut[2 * (x >> 21)], a1 = lut[2 * (x >> 21) + 1];
            const float b0 = lut[2 * ((x >> 10) & 2047u)], b1 = lut[2 * ((x >> 10) & 2047u) + 1];
            l2[s] = a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1;
        }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

// ----------------------------------------------------------------------------- bit stream (TQK)

inline void set_chunk(uint8_t* qs, int kb, int t, uint32_t v) {
    const int off = (31 - t) * kb;
    for (int b = 0; b < kb; ++b) {
        const int i = off + b;
        if ((v >> b) & 1u) {
            qs[i >> 3] |= (uint8_t) (1u << (i & 7));
        } else {
            qs[i >> 3] &= (uint8_t) ~(1u << (i & 7));
        }
    }
}

inline uint32_t get_chunk(const uint8_t* qs, int kb, int t) {
    const int off = (31 - t) * kb;
    uint32_t v = 0;
    for (int b = 0; b < kb; ++b) {
        const int i = off + b;
        v |= (uint32_t) ((qs[i >> 3] >> (i & 7)) & 1u) << b;
    }
    return v;
}

/// Step t's 15-bit window: the 16 stream bits at bit (31 - t) * K, LSB first, circular modulo 32*K.
inline uint32_t window_of(const uint8_t* qs, int kb, int t) {
    const int nbits = 32 * kb;
    const int off = (31 - t) * kb;
    uint32_t v = 0;
    for (int j = 0; j < 16; ++j) {
        const int i = (off + j) % nbits;
        v |= (uint32_t) ((qs[i >> 3] >> (i & 7)) & 1u) << j;
    }
    return v & kStateMask;
}

/// TQ2_T: step t's state is qs[(t+31) % 32] << 8 | qs[t].
inline uint32_t window_tq2t(const uint8_t* qs, int t) {
    return (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & kStateMask;
}

inline uint32_t step_window(int type, const uint8_t* qs, int t) {
    return type == kTq2T ? window_tq2t(qs, t) : window_of(qs, bits_of(type), t);
}

// ----------------------------------------------------------------------------- cost tables

/// cost[t][s] = sum_j w_j (x_{4t+j} - d c_j(s))^2: the four codebook values of a 15-bit window state s against
/// the four weights of every step t.  The state loop is outer so one hash and two codebook loads serve all 32
/// steps; that is ~4x the throughput of a per-step table.  `w` is nullptr for uniform weights.
void cost_all(const Tables& tb, const float* x, const float* w, float d, float cost[kSteps][32768]) {
    const float* __restrict lut = tb.lut;
    const float d2 = d * d;
    const float two_d = 2.0f * d;
    float xT[4][kSteps], wT[4][kSteps], sx2[kSteps];
    for (int t = 0; t < kSteps; ++t) {
        sx2[t] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            xT[j][t] = x[4 * t + j];
            wT[j][t] = w ? w[4 * t + j] : 1.0f;
            sx2[t] += wT[j][t] * xT[j][t] * xT[j][t];
        }
    }
    for (uint32_t s0 = 0; s0 <= kStateMask; s0 += 1024) {
        // The 32 output tables are 128 KiB apart, so walk the states in tiles: 32 x 4 KiB of write streams stays
        // in L2 instead of touching a fresh cache line pair per state.
        const uint32_t s1 = (s0 + 1024 <= kStateMask + 1) ? s0 + 1024 : kStateMask + 1;
        for (uint32_t s = s0; s < s1; ++s) {
            const uint32_t h = s * kGolden;
            const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
            const float c[4] = {lut[2 * p0], lut[2 * p0 + 1], lut[2 * p1], lut[2 * p1 + 1]};
            const float base = d2 * tb.l2[s];
            if (!w) {
                for (int t = 0; t < kSteps; ++t) {
                    const float dot = xT[0][t] * c[0] + xT[1][t] * c[1] + xT[2][t] * c[2] + xT[3][t] * c[3];
                    cost[t][s] = sx2[t] + base - two_d * dot;
                }
            } else {
                for (int t = 0; t < kSteps; ++t) {
                    const float cc = wT[0][t] * c[0] * c[0] + wT[1][t] * c[1] * c[1] + wT[2][t] * c[2] * c[2] +
                                     wT[3][t] * c[3] * c[3];
                    const float dot = wT[0][t] * xT[0][t] * c[0] + wT[1][t] * xT[1][t] * c[1] +
                                      wT[2][t] * xT[2][t] * c[2] + wT[3][t] * xT[3][t] * c[3];
                    cost[t][s] = sx2[t] + d2 * cc - two_d * dot;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------- the search

struct Scratch {
    alignas(64) float cost[kSteps][32768];   // 32 * 128 KiB
    float dp[256], dp2[256];
    float dp_pair[16384], dp_pair2[16384];
    float hist[2 * kSteps][256];             // m == 1: metrics of every position, for the argmin traceback
    float hist_pair[2 * kSteps][16384];      // m == 2
};

/// dst[i] = min(dst[i], base + row[i]) over `n` values: the inner loop of the Viterbi.  Straight-line so the
/// compiler vectorises it; the back-pointers are recovered in the traceback instead of being written here.
__forceinline void dp_min(float* __restrict dst, const float* __restrict row, float base, int n) {
#if defined(_M_X64) || defined(__SSE2__)
    // baseline x86-64: SSE2 is guaranteed, so this needs no run-time dispatch
    const __m128 b = _mm_set1_ps(base);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 r = _mm_add_ps(b, _mm_loadu_ps(row + i));
        _mm_storeu_ps(dst + i, _mm_min_ps(_mm_loadu_ps(dst + i), r));
    }
    for (; i < n; ++i) {
        const float v = base + row[i];
        dst[i] = v < dst[i] ? v : dst[i];
    }
#else
    for (int i = 0; i < n; ++i) {
        const float v = base + row[i];
        dst[i] = v < dst[i] ? v : dst[i];
    }
#endif
}

/// One tail-biting Viterbi lap pair over the 32-step circle at scale `d`; writes the chunk choice of each step
/// into `chunk[0..31]`.
void viterbi(int type, const float* x, const float* w, float d, uint8_t* chunk, Scratch& sc) {
    const Tables& tb = tables();
    const int kb = bits_of(type);
    const int m = window_chunks(kb);
    const uint32_t A = alphabet(kb);

    (void) 0;
    cost_all(tb, x, w, d, sc.cost);

    if (m == 1) {
        // state = the byte written by the previous step; step t's window is ((prev << 8) | cur) & 0x7fff.
        for (uint32_t a = 0; a < A; ++a) sc.dp[a] = 0.0f;
        for (int p = 0; p < 2 * kSteps; ++p) {
            const int t = p % kSteps;
            const float* c = sc.cost[t];
            for (uint32_t b = 0; b < A; ++b) sc.dp2[b] = std::numeric_limits<float>::infinity();
            for (uint32_t a = 0; a < A; ++a) dp_min(sc.dp2, c + ((a & 0x7fu) << 8), sc.dp[a], (int) A);
            std::memcpy(sc.dp, sc.dp2, A * sizeof(float));
            std::memcpy(sc.hist[p], sc.dp, A * sizeof(float));
        }
        uint32_t best = 0;
        for (uint32_t b = 1; b < A; ++b) {
            if (sc.dp[b] < sc.dp[best]) best = b;
        }
        for (int p = 2 * kSteps - 1; p >= kSteps; --p) {
            const int t = p % kSteps;
            const float* c = sc.cost[t];
            const float* prev = sc.hist[p - 1];
            float bestv = std::numeric_limits<float>::infinity();
            uint32_t besta = 0;
            for (uint32_t a = 0; a < A; ++a) {   // argmin of the same sum dp_min minimised
                const float v = prev[a] + c[((a & 0x7fu) << 8) | best];
                if (v < bestv) {
                    bestv = v;
                    besta = a;
                }
            }
            chunk[t] = (uint8_t) best;
            best = besta;
        }
        return;
    }

    // m == 2: state = (c_{t-2}, c_{t-1}); step t's window is c_t | c_{t-1} << K | (c_{t-2} & mask) << 2K.
    const uint32_t mask = (16 - 2 * kb) >= 32 ? 0u : ((1u << (16 - 2 * kb)) - 1u);
    const uint32_t S = A * A;
    for (uint32_t s = 0; s < S; ++s) sc.dp_pair[s] = 0.0f;
    for (int p = 0; p < 2 * kSteps; ++p) {
        const int t = p % kSteps;
        const float* c = sc.cost[t];
        for (uint32_t s = 0; s < S; ++s) sc.dp_pair2[s] = std::numeric_limits<float>::infinity();
        for (uint32_t c0 = 0; c0 < A; ++c0) {           // c_{t-2}
            for (uint32_t c1 = 0; c1 < A; ++c1) {       // c_{t-1}
                const float* row = c + (((c1 << kb) | ((c0 & mask) << (2 * kb))) & kStateMask);
                dp_min(sc.dp_pair2 + c1 * A, row, sc.dp_pair[c0 * A + c1], (int) A);
            }
        }
        std::memcpy(sc.dp_pair, sc.dp_pair2, S * sizeof(float));
        std::memcpy(sc.hist_pair[p], sc.dp_pair, S * sizeof(float));
    }
    uint32_t best = 0;
    for (uint32_t s = 1; s < S; ++s) {
        if (sc.dp_pair[s] < sc.dp_pair[best]) best = s;
    }
    for (int p = 2 * kSteps - 1; p >= kSteps; --p) {
        const int t = p % kSteps;
        const float* c = sc.cost[t];
        const float* prev = sc.hist_pair[p - 1];
        const uint32_t c1 = best / A, c2 = best % A;
        float bestv = std::numeric_limits<float>::infinity();
        uint32_t bestc0 = 0;
        for (uint32_t c0 = 0; c0 < A; ++c0) {
            const float v = prev[c0 * A + c1] + c[(((c1 << kb) | ((c0 & mask) << (2 * kb))) & kStateMask) + c2];
            if (v < bestv) {
                bestv = v;
                bestc0 = c0;
            }
        }
        chunk[t] = (uint8_t) c2;
        best = bestc0 * A + c1;
    }
}

/// Packs the 32 chunk choices of a block into `out` (after the fp16 scale) and refits d.
void pack(int type, const uint8_t* chunk, uint8_t* qs) {
    const int kb = bits_of(type);
    std::memset(qs, 0, (size_t) (4 * kb));
    for (int t = 0; t < kSteps; ++t) {
        if (kb == 8) {
            qs[t] = chunk[t];
        } else {
            set_chunk(qs, kb, t, chunk[t]);
        }
    }
}

/// Weighted least-squares scale of a path (w nullptr = uniform), in the units of `x`.
float refit_scale(int type, const float* x, const float* w, const uint8_t* chunk) {
    const Tables& tb = tables();
    uint8_t qs[32];
    std::memset(qs, 0, sizeof(qs));
    pack(type, chunk, qs);
    float sxc = 0.0f, scc = 0.0f;
    for (int t = 0; t < kSteps; ++t) {
        // Decode through the packed bytes so the search and the decoder cannot disagree.
        const uint32_t s = step_window(type, qs, t);
        const uint32_t h = s * kGolden;
        const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
        const float c[4] = {tb.lut[2 * p0], tb.lut[2 * p0 + 1], tb.lut[2 * p1], tb.lut[2 * p1 + 1]};
        for (int j = 0; j < 4; ++j) {
            const float wj = w ? w[4 * t + j] : 1.0f;
            sxc += wj * x[4 * t + j] * c[j];
            scc += wj * c[j] * c[j];
        }
    }
    return scc > 0.0f ? sxc / scc : 0.0f;
}

}  // namespace

// ----------------------------------------------------------------------------- public API

bool tq_encode_supported(int t) noexcept { return t == kTq2T || t == kTqk6 || t == kTqk7; }

const char* tq_encode_type_name(int t) noexcept {
    return t == kTq2T ? "TQ2_T" : t == kTqk6 ? "TQK6" : t == kTqk7 ? "TQK7" : nullptr;
}

int tq_encode_bits(int t) noexcept { return bits_of(t); }

const float* tq_codebook() { return tables().lut; }

size_t tq_encode_block_bytes(int t) noexcept {
    const int kb = bits_of(t);
    return kb == 0 ? 0 : (size_t) (2 + 4 * kb);
}

size_t tq_encode_row_bytes(int t, int64_t n) {
    if (!tq_encode_supported(t) || n <= 0 || n % kBlock != 0) return 0;
    return (size_t) (n / kBlock) * tq_encode_block_bytes(t);
}

void tq_decode_row(int t, const uint8_t* in, int64_t n, float* y) {
    const Tables& tb = tables();
    const size_t bsz = tq_encode_block_bytes(t);
    for (int64_t i = 0; i < n / kBlock; ++i) {
        const uint8_t* blk = in + (size_t) i * bsz;
        const float d = half_to_float((uint16_t) (blk[0] | (blk[1] << 8)));
        const uint8_t* qs = blk + 2;
        for (int s = 0; s < kSteps; ++s) {
            const uint32_t st = step_window(t, qs, s);
            const uint32_t h = st * kGolden;
            const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
            y[4 * s + 0] = d * tb.lut[2 * p0 + 0];
            y[4 * s + 1] = d * tb.lut[2 * p0 + 1];
            y[4 * s + 2] = d * tb.lut[2 * p1 + 0];
            y[4 * s + 3] = d * tb.lut[2 * p1 + 1];
        }
        y += kBlock;
    }
}

void tq_encode_block(int t, const float* x, const float* w, const TqEncodeParams& params, uint8_t* out) {
    const size_t bsz = tq_encode_block_bytes(t);
    float sumsq = 0.0f;
    for (int j = 0; j < kBlock; ++j) sumsq += x[j] * x[j];
    std::memset(out, 0, bsz);
    if (!(sumsq > 0.0f)) {
        return;   // d = 0: the decoder's zero vector
    }
    const float rms = std::sqrt(sumsq / kBlock);

    Scratch& sc = *[]() -> Scratch* {
        thread_local Scratch s;
        return &s;
    }();

    const float* ww = params.weighted ? w : nullptr;
    uint8_t chunk[32], best_chunk[32];
    float best_score = std::numeric_limits<float>::infinity();
    float best_d = rms;

    const int cand = params.candidates < 1 ? 1 : params.candidates;
    static const float kTargets[5] = {0.80f, 0.90f, 1.00f, 1.15f, 1.30f};
    bool scored = false;
    for (int ci = 0; ci < cand; ++ci) {
        const float tgt = cand == 1 ? 1.0f : kTargets[(ci * 5) / cand];
        float d = rms * tgt;
        for (int pass = 0; pass < (params.passes < 1 ? 1 : params.passes); ++pass) {
            viterbi(t, x, ww, d, chunk, sc);
            const float df = refit_scale(t, x, ww, chunk);
            if (!(df > 0.0f) || !std::isfinite(df)) break;
            d = df;
            float dh = half_to_float(float_to_half(d));
            if (!(dh > 0.0f)) dh = d;
            // Score at the scale the file will hold.
            float num = 0.0f, den = 0.0f;
            {
                const Tables& tb = tables();
                uint8_t qs[32];
                std::memset(qs, 0, sizeof(qs));
                pack(t, chunk, qs);
                for (int s = 0; s < kSteps; ++s) {
                    const uint32_t st = step_window(t, qs, s);
                    const uint32_t h = st * kGolden;
                    const uint32_t p0 = h >> 21, p1 = (h >> 10) & 2047u;
                    const float c[4] = {tb.lut[2 * p0], tb.lut[2 * p0 + 1], tb.lut[2 * p1], tb.lut[2 * p1 + 1]};
                    for (int j = 0; j < 4; ++j) {
                        const float wj = ww ? ww[4 * s + j] : 1.0f;
                        const float e = x[4 * s + j] - dh * c[j];
                        num += wj * e * e;
                        den += wj * x[4 * s + j] * x[4 * s + j];
                    }
                }
                scored = true;
            }
            const float score = den > 0.0f ? num / den : num;
            if (score < best_score) {
                best_score = score;
                best_d = dh;
                std::memcpy(best_chunk, chunk, sizeof(chunk));
            }
        }
    }
    if (!scored) {
        best_d = rms;
        std::memcpy(best_chunk, chunk, sizeof(chunk));
    }

    uint8_t qs[32];
    std::memset(qs, 0, sizeof(qs));
    pack(t, best_chunk, qs);
    const uint16_t dh = float_to_half(best_d);
    out[0] = (uint8_t) (dh & 0xffu);
    out[1] = (uint8_t) (dh >> 8);
    std::memcpy(out + 2, qs, (size_t) (4 * bits_of(t)));
}

void tq_encode_row(int t, const float* x, const float* w, int64_t n, const TqEncodeParams& params, uint8_t* out) {
    const size_t bsz = tq_encode_block_bytes(t);
    for (int64_t i = 0; i < n / kBlock; ++i) {
        tq_encode_block(t, x + i * kBlock, w ? w + i * kBlock : nullptr, params, out + (size_t) i * bsz);
    }
}

void tq_block_importance(const float* w, int64_t n, float* out) {
    for (int64_t b = 0; b < n / kBlock; ++b) {
        double s = 0.0;
        for (int j = 0; j < kBlock; ++j) s += w[b * kBlock + j];
        const float mean = (float) (s / kBlock);
        for (int j = 0; j < kBlock; ++j) out[b * kBlock + j] = mean;
    }
}

void tq_hadamard_rotate(float* x, int64_t n, int block, const float* signs) {
    if (block <= 0) return;
    for (int64_t b0 = 0; b0 + block <= n; b0 += block) {
        float* v = x + b0;
        if (signs) {
            for (int i = 0; i < block; ++i) v[i] *= signs[b0 + i];
        }
        for (int len = 1; len < block; len <<= 1) {
            for (int i = 0; i < block; i += len << 1) {
                for (int j = i; j < i + len; ++j) {
                    const float a = v[j], c = v[j + len];
                    v[j] = a + c;
                    v[j + len] = a - c;
                }
            }
        }
        const float scale = 1.0f / std::sqrt((float) block);
        for (int i = 0; i < block; ++i) v[i] *= scale;
    }
}

std::vector<float> tq_hadamard_signs(int width, uint32_t seed) {
    std::vector<float> s((size_t) width, 1.0f);
    uint32_t x = seed * 2654435761u + 1u;
    for (int i = 0; i < width; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s[(size_t) i] = (x & 1u) ? 1.0f : -1.0f;
    }
    return s;
}

}  // namespace strata::kernels::cpu
