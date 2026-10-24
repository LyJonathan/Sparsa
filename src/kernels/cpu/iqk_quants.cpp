// src/kernels/cpu/iqk_quants.cpp - codec for the quant formats ported from ik_llama.cpp.
//
// Ported from ik_llama.cpp, commit 804ee29 (MIT).
//   Copyright (c) 2023-2024 The ggml authors
//   Copyright (c) 2023-2024 The llama.cpp authors
//   Copyright (c) 2024-2025 The ik_llama.cpp authors
// See THIRD_PARTY_NOTICES.md.
//
// What is ported and what is not:
//   * the trellis codebook (`QuantizerIQKT::set_values`), byte for byte;
//   * the cluster search ik builds once per process (`cluster_points`, `finalize_clusters`), including the
//     AVX2 scan and its comparison order, so a tensor quantizes to the same bytes ik produces;
//   * `quantize_row_iq2_kt_impl` / `quantize_row_iq3_kt_impl`, with their constants unchanged;
//   * the dequantizers, as ik's `to_float` produces them.
// Not ported: ik's `iqk_mul_mat` tile kernels.  `dot_row` here decodes a block with the same codebook and
// dots it in AVX2 float; it trades some of ik's speed for being short enough to audit, and the engine's hot
// path is the grouped CUDA kernel anyway.
//
// This file needs AVX2 for the quantizer, exactly like ik (it refuses to quantize these formats without
// AVX2).  The engine builds it with /arch:AVX2 next to the other IQ kernels.

#include "strata/kernels/cpu/iqk_quants.hpp"

#include "strata/kernels/iqk_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace strata::kernels::ikq {
namespace {

constexpr int QK_K_ = 256;
constexpr int kBlockSize = 32;   // one trellis block: 32 values, 4 groups of 8
constexpr int kGroupSize = 8;
constexpr int kNg = kBlockSize / kGroupSize;      // 4
constexpr int kNblock = QK_K_ / kBlockSize;       // 8
constexpr int kNumVal = 1 << 16;                  // 16-bit index
constexpr float kTrellisScale = 1.0f;             // QuantizerIQKT::kScale with is_int = true
constexpr int kIndexOffset = 4096;

// The fudge factors are 1.0 for both formats in ik's table (`ggml_get_quantize_fudge_factor`).
constexpr float kFudgeIQ2KT = 1.0f;
constexpr float kFudgeIQ3KT = 1.0f;

// ---- tables (ik ggml-common.h / ggml-quants.c) -----------------------------------------------------------

constexpr int8_t kIq4kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

constexpr int8_t kIq4nlIndex[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};

int best_index_iq4nl(const int8_t* values, float x) {
    int ix = (int) x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = kIq4nlIndex[ix];
    return ix < 16 ? ix : x - values[ix - 16] < values[ix - 15] - x ? ix - 16 : ix - 15;
}

int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// ---- fp16 ------------------------------------------------------------------------------------------------

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // subnormal: normalize
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = (int32_t) ((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t man = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0u));
    }
    if (exp >= 0x1f) return (uint16_t) (sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        const uint32_t m = (man | 0x800000u) >> (uint32_t) (1 - exp);
        // round to nearest even
        const uint32_t round = m & 0x1fffu;
        uint32_t r = m >> 13;
        if (round > 0x1000u || (round == 0x1000u && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    uint32_t r = (uint32_t) (man >> 13) | ((uint32_t) exp << 10);
    const uint32_t round = man & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}

inline uint16_t read_u16(const void* p) noexcept {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// ---- the trellis codebook --------------------------------------------------------------------------------

/// ik's `QuantizerIQKT::set_values` for `is_int = true`: eight values from a 16-bit index, by a hash.
inline void trellis_values(uint32_t i, float* out, float scale, bool is_abs) {
    constexpr uint32_t ka = 0xCBAC1FEDu;
    uint32_t x = i + (uint32_t) kIndexOffset;
    for (int k = 0; k < kGroupSize; ++k) {
        x = ka * x;
        const uint32_t s = x & 0x3f3f3f3fu;
        int8_t sb[4];
        std::memcpy(sb, &s, sizeof(sb));
        const float v = (float) sb[0] + (float) sb[1] + (float) sb[2] + (float) sb[3] - 126.f;
        out[k] = scale * (is_abs ? std::fabs(v) : v);
    }
}

// ---- the cluster search ik builds once per process -------------------------------------------------------

struct Cluster {
    // ik's `m_c_values[i]`: the codebook points of cluster i, each kGroupSize floats.
    std::vector<std::vector<float>> c_values;
    // ik's `m_in_cluster[i]`: the codebook indices belonging to cluster i.
    std::vector<std::vector<int>> in_cluster;
};

/// Port of `QuantizerIQKT<32, 8, 16, is_abs, true>` with group_size 8 only (the KT formats Strata ports
/// use 8; ik's 4-value variant belongs to IQ4_KT, which is not ported).
class QuantizerKT {
public:
    explicit QuantizerKT(bool is_abs, int num_clusters = 256, int num_neighbours = 8) : is_abs_(is_abs) {
        values_.resize((size_t) kNumVal * kGroupSize);
        float* data = values_.data();
        for (int i = 0; i < kNumVal; ++i) {
            trellis_values((uint32_t) i, data, kTrellisScale, is_abs_);
            data += kGroupSize;
        }
        std::memset(mid_, 0, sizeof(mid_));
        if (num_clusters == 0) return;
        clusters_ = cluster_points(values_, num_clusters);
        finalize_clusters(num_neighbours);
    }

    const float* values() const { return values_.data(); }
    const std::vector<std::vector<int>>& in_cluster() const { return in_cluster_; }

    /// Port of `QuantizerIQKT::set_weights`.
    static void set_weights(float sigma2_scale, int nblock, const float* x, const float* imatrix,
                            float* row_weights, int n_per_row) {
        constexpr float kEps2 = 1e-14f;
        constexpr float kWeight = 1e-4f;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float* xbl = x + (int64_t) ibl * QK_K_;
            float* wbl = row_weights + (int64_t) ibl * QK_K_;
            const int n = (int) std::min<int64_t>(QK_K_, (int64_t) n_per_row - (int64_t) ibl * QK_K_);
            for (int j = n; j < QK_K_; ++j) wbl[j] = 0;
            float sumx2 = 0;
            for (int j = 0; j < n; ++j) sumx2 += xbl[j] * xbl[j];
            if (sumx2 < kEps2 * n) {
                for (int j = 0; j < n; ++j) wbl[j] = kWeight;
                continue;
            }
            const float sigma2 = sigma2_scale * sumx2 / n;
            if (imatrix) {
                for (int ib = 0; ib < n / kBlockSize; ++ib) {
                    const float* qw = imatrix + (int64_t) ibl * QK_K_ + (int64_t) ib * kBlockSize;
                    const float* xb = xbl + ib * kBlockSize;
                    float* wb = wbl + ib * kBlockSize;
                    float sumwx = 0, sumw2 = 0, sx2 = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        wb[j] = qw[j] * std::sqrt(sigma2 + xb[j] * xb[j]);
                        sumwx += wb[j] * std::fabs(xb[j]);
                        sumw2 += wb[j] * wb[j];
                        sx2 += xb[j] * xb[j];
                    }
                    if (sx2 < kEps2 || sumw2 < kEps2 || sumwx < kEps2) {
                        for (int j = 0; j < kBlockSize; ++j) wb[j] = kWeight;
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) wbl[j] = 0.25f * sigma2 + xbl[j] * xbl[j];
            }
        }
    }

    /// Port of `QuantizerIQKT::find_best_scale` for group_size 8.
    std::pair<float, float> find_best_scale(const float* xb, const float* weight, const int* best_idx) const {
        float sumqx = 0, sumq2 = 0;
#if defined(__AVX2__)
        auto vqx = _mm256_setzero_ps();
        auto vq2 = _mm256_setzero_ps();
        for (int l = 0; l < kBlockSize; l += 8) {
            auto vx = _mm256_loadu_ps(xb + l);
            auto vw = _mm256_loadu_ps(weight + l);
            auto vq = _mm256_loadu_ps(values_.data() + kGroupSize * best_idx[l / kGroupSize]);
            auto vqw = _mm256_mul_ps(vq, vw);
            vqx = _mm256_fmadd_ps(vqw, vx, vqx);
            vq2 = _mm256_fmadd_ps(vqw, vq, vq2);
        }
        sumqx = hsum8(vqx);
        sumq2 = hsum8(vq2);
#else
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + kGroupSize * l;
            auto wl = weight + kGroupSize * l;
            auto ql = values_.data() + kGroupSize * best_idx[l];
            for (int k = 0; k < kGroupSize; ++k) {
                sumqx += wl[k] * ql[k] * xl[k];
                sumq2 += wl[k] * ql[k] * ql[k];
            }
        }
#endif
        return sumq2 > 0 ? std::make_pair(sumqx / sumq2, sumqx * sumqx / sumq2) : std::make_pair(0.f, 0.f);
    }

    /// Port of `QuantizerIQKT::find_best_match` for group_size 8 and ncluster 256.
    void find_best_match(float d, const float* xb, const float* weight, int* best_idx) const {
        if (!d) {
            std::memset(best_idx, 0, kNg * sizeof(int));
            return;
        }
#if defined(__AVX2__)
        const int ncluster = (int) (clusters_.size() / kGroupSize);
        float id = 1 / d;
        __m256 sqx[8];
        const __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        float sx[8];
        int index[8];
        auto vid = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto vx = _mm256_mul_ps(vid, _mm256_loadu_ps(xb + 8 * l));
            auto vw = _mm256_loadu_ps(weight + 8 * l);
            int jbest = -1;
            if (ncluster == 256) {
                _mm256_store_ps(sx, vx);
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j)
                    if (sx[j] > mid_[j]) u |= (uint16_t) (1 << j);
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 8; ++i) {
                        auto vq = _mm256_loadu_ps(clusters_.data() + kGroupSize * (j + i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                    }
                    auto score = hsum8x8(sqx);
                    auto mask = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                                 _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i*) index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) {
                        best = sx[i];
                        jbest = index[i];
                    }
                }
            }
            const auto& points = in_cluster_[(size_t) jbest];
            const auto& values = points.empty() ? values_ : c_values_[(size_t) jbest];
            const int npoint = (int) (values.size() / kGroupSize);
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            float best = INFINITY;
            jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < npoint; j += 8) {
                for (int i = 0; i < 8; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize * (j + i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum8x8(sqx);
                auto mask = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                             _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i*) index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) {
                    best = sx[i];
                    jbest = index[i];
                }
            }
            if (jbest < 0) {
                throw std::runtime_error("ikq: trellis search found no codebook point");
            }
            best_idx[l] = points.empty() ? jbest : points[(size_t) jbest];
        }
#else
        (void) xb;
        (void) weight;
        std::memset(best_idx, 0, kNg * sizeof(int));
#endif
    }

    /// Values of codebook index `idx` (8 floats), scaled.
    void values_of(int idx, float scale, float* out) const {
        trellis_values((uint32_t) idx, out, scale, is_abs_);
    }

private:
#if defined(__AVX2__)
    static float hsum8(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
    static __m256 hsum8x8(__m256* accm) {
        for (int i = 0; i < 4; ++i) {
            accm[i] = _mm256_add_ps(_mm256_permute2f128_ps(accm[i], accm[i + 4], 0x20),
                                    _mm256_permute2f128_ps(accm[i], accm[i + 4], 0x31));
        }
        for (int i = 0; i < 2; ++i) {
            accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i + 2]), _mm256_unpackhi_ps(accm[i], accm[i + 2]));
        }
        return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
    }
#endif

    /// Port of `QuantizerIQKT::cluster_points` for group_size 8 (the ndim == 8, ncluster == 256 shape:
    /// one pass of midpoint binning, no iteration).
    std::vector<float> cluster_points(const std::vector<float>& points, int ncluster) {
        constexpr int ndim = kGroupSize;
        const int npoint = (int) (points.size() / ndim);
        std::vector<std::pair<float, float>> range((size_t) ndim, std::make_pair(INFINITY, -INFINITY));
        for (int i = 0; i < npoint; ++i) {
            auto v = points.data() + (int64_t) i * ndim;
            for (int k = 0; k < ndim; ++k) {
                range[(size_t) k].first = std::min(range[(size_t) k].first, v[k]);
                range[(size_t) k].second = std::max(range[(size_t) k].second, v[k]);
            }
        }
        if (is_abs_) {
            std::vector<float> P((size_t) npoint);
            for (int idim = 0; idim < ndim; ++idim) {
                for (int ip = 0; ip < npoint; ++ip) P[(size_t) ip] = points[(size_t) ip * ndim + idim];
                std::sort(P.begin(), P.end());
                mid_[idim] = npoint % 2 == 0 ? 0.5f * (P[(size_t) (npoint / 2)] + P[(size_t) (npoint / 2 - 1)])
                                             : P[(size_t) (npoint / 2)];
            }
        } else {
            for (int k = 0; k < ndim; ++k) mid_[k] = 0.5f * (range[(size_t) k].first + range[(size_t) k].second);
        }
        std::vector<float> sump((size_t) ncluster * ndim, 0.f);
        std::vector<int> counts((size_t) ncluster, 0);
        std::vector<float> result((size_t) ncluster * ndim);
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + (int64_t) ip * ndim;
            uint16_t u = 0;
            for (int k = 0; k < ndim; ++k)
                if (vp[k] > mid_[k]) u |= (uint16_t) (1 << k);
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[(int64_t) ndim * u + k] += vp[k];
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[(size_t) ic]) {
                throw std::runtime_error("ikq: empty trellis cluster");
            }
            for (int k = 0; k < ndim; ++k) result[(int64_t) ic * ndim + k] = sump[(int64_t) ic * ndim + k] / counts[(size_t) ic];
        }
        return result;
    }

    /// Port of `QuantizerIQKT::finalize_clusters`.
    void finalize_clusters(int num_neighbours) {
        const int ncluster = (int) (clusters_.size() / kGroupSize);
        in_cluster_.assign((size_t) ncluster, {});
        c_values_.assign((size_t) ncluster, {});
        std::vector<int> which_cluster((size_t) num_neighbours * kNumVal);
        std::vector<int> ibest((size_t) num_neighbours);
        std::vector<float> best((size_t) num_neighbours);
        for (int ip = 0; ip < kNumVal; ++ip) {
            auto vp = values_.data() + (int64_t) ip * kGroupSize;
            for (int j = 0; j < num_neighbours; ++j) {
                best[(size_t) j] = INFINITY;
                ibest[(size_t) j] = -1;
            }
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = clusters_.data() + (int64_t) ic * kGroupSize;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float dd = vp[k] - vc[k];
                    dist2 += dd * dd;
                }
                for (int j = 0; j < num_neighbours; ++j) {
                    if (dist2 < best[(size_t) j]) {
                        for (int k = num_neighbours - 1; k > j; --k) {
                            best[(size_t) k] = best[(size_t) (k - 1)];
                            ibest[(size_t) k] = ibest[(size_t) (k - 1)];
                        }
                        best[(size_t) j] = dist2;
                        ibest[(size_t) j] = ic;
                        break;
                    }
                }
            }
            for (int j = 0; j < num_neighbours; ++j) {
                if (ibest[(size_t) j] < 0) throw std::runtime_error("ikq: trellis cluster under-filled");
                in_cluster_[(size_t) ibest[(size_t) j]].push_back(ip);
            }
            std::memcpy(which_cluster.data() + (int64_t) num_neighbours * ip, ibest.data(),
                        (size_t) num_neighbours * sizeof(int));
        }
        std::vector<std::pair<float, int>> extra;
        extra.reserve(kNumVal);
        for (int ic = 0; ic < ncluster; ++ic) {
            auto& points = in_cluster_[(size_t) ic];
            if (!points.empty() && points.size() % 8 == 0) continue;
            extra.clear();
            auto vc = clusters_.data() + (int64_t) ic * kGroupSize;
            for (int ip = 0; ip < kNumVal; ++ip) {
                bool can_add = true;
                for (int j = 0; j < num_neighbours; ++j) {
                    if (which_cluster[(int64_t) num_neighbours * ip + j] == ic) {
                        can_add = false;
                        break;
                    }
                }
                if (!can_add) continue;
                auto vp = values_.data() + (int64_t) ip * kGroupSize;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float dd = vp[k] - vc[k];
                    dist2 += dd * dd;
                }
                extra.push_back(std::make_pair(dist2, ip));
            }
            std::sort(extra.begin(), extra.end());
            const int nadd = 8 * (int) ((points.size() + 7) / 8) - (int) points.size();
            for (int i = 0; i < nadd; ++i) points.push_back(extra[(size_t) i].second);
        }
        for (int i = 0; i < ncluster; ++i) {
            auto& points = in_cluster_[(size_t) i];
            c_values_[(size_t) i].resize(points.size() * kGroupSize);
            auto ptr = c_values_[(size_t) i].data();
            for (auto j : points) {
                std::memcpy(ptr, values_.data() + (int64_t) j * kGroupSize, kGroupSize * sizeof(float));
                ptr += kGroupSize;
            }
        }
    }

    bool is_abs_;
    std::vector<float> values_;
    std::vector<float> clusters_;
    std::vector<std::vector<float>> c_values_;
    std::vector<std::vector<int>> in_cluster_;
    float mid_[4 * kGroupSize];
};

const QuantizerKT& quantizer(bool is_abs) {
    static std::once_flag once_false, once_true;
    static std::unique_ptr<QuantizerKT> plain, absq;
    if (is_abs) {
        std::call_once(once_true, [] { absq = std::make_unique<QuantizerKT>(true, 256, 8); });
        return *absq;
    }
    std::call_once(once_false, [] { plain = std::make_unique<QuantizerKT>(false, 256, 8); });
    return *plain;
}

/// Decode one whole 256-value block (68 bytes for IQ2_KT, 100 for IQ3_KT) at row scale `d`.  Both the
/// dequantizer and the dot product go through this, so they cannot drift apart.
void decode_block(int type, const uint8_t* blk, float d, const QuantizerKT& q, float* w, float post = 1.f) {
    const uint16_t* qll = (const uint16_t*) (blk + 4);
    float* yl = w;
    float* yh = w + QK_K_ / 2;
    if (type == kIQ2KT) {
        const uint16_t* qlh = qll + kNg * kNblock / 2;
        for (int ib = 0; ib < kNblock / 2; ++ib) {
            const float sl = d * kIq4kValues[blk[ib] & 0xf];
            const float sh = d * kIq4kValues[blk[ib] >> 4];
            for (int ig = 0; ig < kNg; ++ig) {
                q.values_of(qll[ig], sl, yl);
                q.values_of(qlh[ig], sh, yh);
                yl += kGroupSize;
                yh += kGroupSize;
            }
            qll += kNg;
            qlh += kNg;
        }
        if (post != 1.f) {
            for (int j = 0; j < QK_K_; ++j) w[j] *= post;
        }
        return;
    }
    const uint16_t* qlh = qll + kNg * kNblock / 2;
    const uint8_t* qh = blk + 4 + 64;
    int jj = 0;
    for (int ib = 0; ib < kNblock / 2; ++ib) {
        const float sl = d * (float) (blk[ib] & 0xf);
        const float sh = d * (float) (blk[ib] >> 4);
        const uint8_t l_mask = (uint8_t) (1 << ib);
        const uint8_t h_mask = (uint8_t) (l_mask << (kNblock / 2));
        for (int ig = 0; ig < kNg; ++ig) {
            q.values_of(qll[jj], sl, yl);
            q.values_of(qlh[jj], sh, yh);
            for (int j = 0; j < kGroupSize; ++j) {
                if (qh[ig * kGroupSize + j] & l_mask) yl[j] = -yl[j];
                if (qh[ig * kGroupSize + j] & h_mask) yh[j] = -yh[j];
            }
            yl += kGroupSize;
            yh += kGroupSize;
            ++jj;
        }
    }
    if (post != 1.f) {
        for (int j = 0; j < QK_K_; ++j) w[j] *= post;
    }
}

// ---- quantization ----------------------------------------------------------------------------------------

/// Port of `quantize_row_iq2_kt_impl`.
void quantize_row_iq2_kt_impl(const float* x, void* vy, int n_per_row, const float* quant_weights,
                              float* all_scales, float* all_weights, int* all_idx) {
    constexpr float kSigmaScale = 2.0f;
    float* dptr = (float*) vy;
    uint8_t* y = (uint8_t*) (dptr + 1);          // block_iq2_kt: scales[4] + ql[64], 68 bytes
    int best_idx[2 * kNg];
    const auto& q = quantizer(false);
    const int nblock = n_per_row / QK_K_;

    QuantizerKT::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::fabs(x[j]));
    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {
        std::memset(y + (int64_t) ibl * 68, 0, 68);
        const float* xbl = x + (int64_t) ibl * QK_K_;
        float* scales = all_scales + (int64_t) ibl * kNblock;
        for (int ib = 0; ib < kNblock; ++ib) {
            const float* xb = xbl + kBlockSize * ib;
            const float* weight = all_weights + (int64_t) ibl * QK_K_ + ib * kBlockSize;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) amax = std::max(amax, std::fabs(xb[j]));
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < kNg; ++ig) all_idx[(ibl * QK_K_ + ib * kBlockSize) / kGroupSize + ig] = 0;
                continue;
            }
            const float scale_0 = std::max(90.f, 124.f * amax / amax_row);
            q.find_best_match(amax / scale_0, xb, weight, best_idx);
            auto [dp, score_p] = q.find_best_scale(xb, weight, best_idx);
            q.find_best_match(-amax / scale_0, xb, weight, best_idx + kNg);
            auto [dm, score_m] = q.find_best_scale(xb, weight, best_idx + kNg);
            auto idx = best_idx;
            if (score_p > score_m) {
                scales[ib] = dp;
            } else {
                scales[ib] = dm;
                idx += kNg;
            }
            for (int ig = 0; ig < kNg; ++ig) all_idx[(ibl * QK_K_ + ib * kBlockSize) / kGroupSize + ig] = idx[ig];
            const float abs_scale = std::fabs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }
    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale / kIq4kValues[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        const float id = (itry + kIq4kValues[0]) / max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float* xb = x + (int64_t) ibl * QK_K_;
            const float* wb = all_weights + (int64_t) ibl * QK_K_;
            const float* scales = all_scales + (int64_t) ibl * kNblock;
            for (int ib = 0; ib < kNblock; ++ib) {
                const int ls = best_index_iq4nl(kIq4kValues, id * scales[ib]);
                const float dl = kIq4kValues[ls];
                for (int ig = 0; ig < kNg; ++ig) {
                    const float* qb = q.values() + kGroupSize * all_idx[(ibl * QK_K_ + ib * kBlockSize) / kGroupSize + ig];
                    for (int j = 0; j < kGroupSize; ++j) {
                        const int jj = ig * kGroupSize + j;
                        const float qq = dl * qb[j];
                        sumqx += wb[jj] * xb[jj] * qq;
                        sumq2 += wb[jj] * qq * qq;
                    }
                }
                xb += kBlockSize;
                wb += kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }

    const float id = d ? 1 / d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float* scales = all_scales + (int64_t) ibl * kNblock;
        for (int ib = 0; ib < kNblock / 2; ++ib) {
            const int ls1 = best_index_iq4nl(kIq4kValues, id * scales[ib]);
            const int ls2 = best_index_iq4nl(kIq4kValues, id * scales[ib + kNblock / 2]);
            y[(int64_t) ibl * 68 + ib] = (uint8_t) (ls1 | (ls2 << 4));
        }
    }

    *dptr = d * kFudgeIQ2KT;
    if (!d) return;

    {
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            uint16_t* qs = (uint16_t*) (y + (int64_t) ibl * 68 + 4);
            const float* xbl = x + (int64_t) ibl * QK_K_;
            for (int ib = 0; ib < kNblock; ++ib) {
                const float* xb = xbl + kBlockSize * ib;
                const float* weight = all_weights + (int64_t) ibl * QK_K_ + ib * kBlockSize;
                const int lsb = y[(int64_t) ibl * 68 + (ib % (kNblock / 2))];
                const int ls = kIq4kValues[(lsb >> (4 * (ib / (kNblock / 2)))) & 0xf];
                const float dl = d * ls;
                q.find_best_match(dl, xb, weight, best_idx);
                uint16_t* prev_idx = (uint16_t*) nullptr;
                int prev[8];
                {
                    const int base = (ibl * QK_K_ + ib * kBlockSize) / kGroupSize;
                    for (int ig = 0; ig < kNg; ++ig) prev[ig] = all_idx[base + ig];
                }
                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < kNg; ++ig) {
                    const float* q1 = q.values() + kGroupSize * prev[ig];
                    const float* q2 = q.values() + kGroupSize * best_idx[ig];
                    for (int j = 0; j < kGroupSize; ++j) {
                        const int jj = ig * kGroupSize + j;
                        const float d1 = xb[jj] - dl * q1[j];
                        const float d2 = xb[jj] - dl * q2[j];
                        mse1 += weight[jj] * d1 * d1;
                        mse2 += weight[jj] * d2 * d2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < kNg; ++ig) best_idx[ig] = prev[ig];
                } else {
                    const int base = (ibl * QK_K_ + ib * kBlockSize) / kGroupSize;
                    for (int ig = 0; ig < kNg; ++ig) all_idx[base + ig] = best_idx[ig];
                }
                (void) prev_idx;
                for (int j = 0; j < kNg; ++j) {
                    qs[j] = (uint16_t) best_idx[j];
                    const float* xl = xb + kGroupSize * j;
                    const float* wl = weight + kGroupSize * j;
                    const float* ql = q.values() + best_idx[j] * kGroupSize;
                    for (int k = 0; k < kGroupSize; ++k) {
                        const float qq = ql[k] * ls;
                        sumqx += wl[k] * xl[k] * qq;
                        sumq2 += wl[k] * qq * qq;
                    }
                }
                qs += kNg;
            }
        }
        if (sumq2 > 0) {
            d = sumqx / sumq2;
            *dptr = d * kFudgeIQ2KT;
            if (!d) return;
        }
    }
}

/// Port of `quantize_row_iq3_kt_impl`.
void quantize_row_iq3_kt_impl(const float* x, void* vy, int n_per_row, const float* quant_weights,
                              float* all_scales, float* all_weights, float* qtmp) {
    constexpr float kSigmaScale = 2.0f;
    constexpr float kStep = 8.0f;
    constexpr int kNumGroups = QK_K_ / kGroupSize;

    float* dptr = (float*) vy;
    uint8_t* y = (uint8_t*) (dptr + 1);   // block_iq3_kt: scales[4] + ql[64] + qh[32], 100 bytes
    int best_idx[2 * kNg];
    const auto& q = quantizer(true);
    const int nblock = (n_per_row + QK_K_ - 1) / QK_K_;

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::fabs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y, 0, (size_t) nblock * 100);
        return;
    }

    QuantizerKT::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_scale = 0, max_scale = 0;
    float xaux[kBlockSize];

    for (int ibl = 0; ibl < nblock; ++ibl) {
        std::memset(y + (int64_t) ibl * 100, 0, 100);
        float* scales = all_scales + (int64_t) ibl * kNblock;
        const float* xbl = x + (int64_t) ibl * QK_K_;
        for (int ib = 0; ib < kNblock; ++ib) {
            const float* xb = xbl + kBlockSize * ib;
            const float* weight = all_weights + (int64_t) ibl * QK_K_ + ib * kBlockSize;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                const float ax = std::fabs(xb[j]);
                xaux[j] = ax;
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                continue;
            }
            const float scale_0 = std::max(84.f, 123.f * amax / amax_row);
            float best = 0;
            bool found_solution = false;
            for (int itry = -3; itry <= 3; ++itry) {
                q.find_best_match(amax / (scale_0 + kStep * itry), xaux, weight, best_idx);
                auto [dd, score] = q.find_best_scale(xaux, weight, best_idx);
                if (score > best) {
                    best = score;
                    found_solution = true;
                    scales[ib] = dd;
                    std::memcpy(best_idx + kNg, best_idx, kNg * sizeof(int));
                }
            }
            if (!found_solution) throw std::runtime_error("ikq: no trellis solution for a block");
            float* xt = qtmp + (int64_t) ibl * QK_K_ + ib * kBlockSize;
            for (int ig = 0; ig < kNg; ++ig) {
                const float* qq = q.values() + kGroupSize * best_idx[kNg + ig];
                for (int j = 0; j < kGroupSize; ++j) *xt++ = qq[j];
            }
            const float abs_scale = std::fabs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }
    }

    float d = max_scale / 15;
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        const float id = (itry * 0.2f + 15) / max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float* xb = x + (int64_t) ibl * QK_K_;
            const float* qb = qtmp + (int64_t) ibl * QK_K_;
            const float* wb = all_weights + (int64_t) ibl * QK_K_;
            const float* scales = all_scales + (int64_t) ibl * kNblock;
            for (int ib = 0; ib < kNblock; ++ib) {
                int ls = nearest_int(id * scales[ib]);
                ls = std::max(0, std::min(15, ls));
                const float dl = (float) ls;
                for (int j = 0; j < kBlockSize; ++j) {
                    const float qq = dl * qb[j];
                    sumqx += wb[j] * std::fabs(xb[j]) * qq;
                    sumq2 += wb[j] * qq * qq;
                }
                xb += kBlockSize;
                wb += kBlockSize;
                qb += kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }

    const float id = d ? 1 / d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float* scales = all_scales + (int64_t) ibl * kNblock;
        for (int ib = 0; ib < kNblock / 2; ++ib) {
            int ls1 = std::max(0, std::min(15, nearest_int(id * scales[ib])));
            int ls2 = std::max(0, std::min(15, nearest_int(id * scales[ib + kNblock / 2])));
            y[(int64_t) ibl * 100 + ib] = (uint8_t) (ls1 | (ls2 << 4));
        }
    }

    *dptr = d * kFudgeIQ3KT;
    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        uint8_t* blk = y + (int64_t) ibl * 100;
        uint16_t* ql = (uint16_t*) (blk + 4);
        uint8_t* qh = blk + 4 + 64;
        std::memset(qh, 0, kNumGroups / 2);
        const float* xbl = x + (int64_t) ibl * QK_K_;
        for (int ib = 0; ib < kNblock; ++ib) {
            const float* xb = xbl + kBlockSize * ib;
            const float* weight = all_weights + (int64_t) ibl * QK_K_ + ib * kBlockSize;
            for (int j = 0; j < kBlockSize; ++j) {
                xaux[j] = std::fabs(xb[j]);
                if (xb[j] < 0) qh[j] |= (uint8_t) (1 << ib);
            }
            const int ls = (blk[ib % (kNblock / 2)] >> (4 * (ib / (kNblock / 2)))) & 0xf;
            q.find_best_match(d * ls, xaux, weight, best_idx);
            for (int j = 0; j < kNg; ++j) {
                ql[ib * kNg + j] = (uint16_t) best_idx[j];
                const float* xl = xaux + kGroupSize * j;
                const float* wl = weight + kGroupSize * j;
                const float* qv = q.values() + best_idx[j] * kGroupSize;
                for (int k = 0; k < kGroupSize; ++k) {
                    const float qq = qv[k] * ls;
                    sumqx += wl[k] * xl[k] * qq;
                    sumq2 += wl[k] * qq * qq;
                }
            }
        }
    }
    if (sumq2 > 0) {
        d = sumqx / sumq2;
        *dptr = d * kFudgeIQ3KT;
    }
}

// ik's `iqk_kt_pad_row`: repeat the tail values up to the next whole super-block.
void pad_row(const float* src, float* dst, int64_t n_per_row) {
    const int64_t padded = (n_per_row + QK_K_ - 1) / QK_K_ * QK_K_;
    const int64_t n_extra = padded - n_per_row;
    std::memcpy(dst, src, (size_t) n_per_row * sizeof(float));
    if (n_extra == 0) return;
    if (n_per_row >= n_extra) {
        std::memcpy(dst + n_per_row, src + n_per_row - n_extra, (size_t) n_extra * sizeof(float));
    } else {
        for (int64_t ib = 0; ib < n_extra / 32; ++ib)
            std::memcpy(dst + n_per_row + 32 * ib, src + n_per_row - 32, 32 * sizeof(float));
    }
}

// ik's `iqk_kt_finish_row` for the two ported formats.
void finish_row(int type, const void* ypad, void* qrow, int64_t n_per_row) {
    const int nb = (int) (n_per_row / QK_K_);
    const int nt = (int) ((n_per_row % QK_K_) / 32);
    const size_t head = row_bytes(type, (int64_t) nb * QK_K_);
    const size_t total = row_bytes(type, n_per_row);
    std::memset(qrow, 0, total);
    std::memcpy(qrow, ypad, head);
    if (nt == 0) return;
    const uint8_t* blocks = (const uint8_t*) ypad + 4;
    uint8_t* tail = (uint8_t*) qrow + head;
    if (type == kIQ2KT) {
        const uint8_t* x = blocks + (int64_t) nb * 68;
        std::memcpy(tail + 0, x, 4);
        std::memcpy(tail + 4, x + 4, (size_t) 8 * nt);
    } else {
        const uint8_t* x = blocks + (int64_t) nb * 100;
        std::memcpy(tail, x + 4, (size_t) 8 * nt);
        uint8_t* qh = tail + 8 * nt;
        std::memset(qh, 0, (size_t) 4 * nt);
        for (int ib = 0; ib < nt; ++ib) {
            for (int j = 0; j < 32; ++j) {
                if (x[4 + 64 + j] & (1 << ib)) qh[4 * ib + j / 8] |= (uint8_t) (1 << (j & 7));
            }
        }
        uint8_t* st = tail + 12 * nt;
        std::memset(st, 0, (size_t) (nt + 1) / 2);
        for (int ib = 0; ib < nt; ++ib) {
            st[ib / 2] |= (uint8_t) (((x[ib % 4] >> (4 * (ib / 4))) & 0xf) << (4 * (ib & 1)));
        }
    }
}

}  // namespace

float dequant_factor(int type) {
    switch (type) {
    case kIQ2KT: return 1.05f;
    default: return 1.0f;
    }
}

/// The scale every code path multiplies the f32 row scale by, so dequant and dot cannot disagree.
inline void row_scale(int type, float d0, float& d, float& post) {
    post = dequant_factor(type);
    d = type == kIQ3KT ? d0 * 1.01f : d0;
}

void dequantize_row(int type, const void* src, float* dst, int64_t n) {
    if (n <= 0 || (n % 32) != 0) throw std::runtime_error("ikq: dequantize width");
    const bool is_abs = type == kIQ3KT;
    if (type != kIQ2KT && type != kIQ3KT) throw std::runtime_error("ikq: dequantize type");
    const auto& q = quantizer(is_abs);
    const uint8_t* base = (const uint8_t*) src;
    float d = 0, post = 1;
    row_scale(type, *reinterpret_cast<const float*>(base) * kTrellisScale, d, post);
    const uint8_t* blocks = base + 4;
    const int nb = (int) (n / QK_K_);
    const int stride = type == kIQ2KT ? 68 : 100;
    for (int ibl = 0; ibl < nb; ++ibl) {
        decode_block(type, blocks + (int64_t) ibl * stride, d, q, dst + (int64_t) ibl * QK_K_, post);
    }
    const int nt = (int) ((n % QK_K_) / 32);
    if (!nt) return;
    const uint8_t* tail = blocks + (int64_t) nb * stride;
    float* y = dst + (int64_t) nb * QK_K_;
    const float dtail = d;
    if (type == kIQ2KT) {
        const uint16_t* ql = (const uint16_t*) (tail + 4);
        for (int ib = 0; ib < nt; ++ib) {
            const float s = dtail * kIq4kValues[(tail[ib % 4] >> (4 * (ib / 4))) & 0xf];
            for (int ig = 0; ig < kNg; ++ig) {
                q.values_of(ql[kNg * ib + ig], s, y);
                if (post != 1.f) {
                    for (int j = 0; j < kGroupSize; ++j) y[j] *= post;
                }
                y += kGroupSize;
            }
        }
    } else {
        const uint8_t* ql = tail;
        const uint8_t* qh = tail + 8 * nt;
        const uint8_t* st = tail + 12 * nt;
        for (int ib = 0; ib < nt; ++ib) {
            const float s = dtail * (float) ((st[ib / 2] >> (4 * (ib & 1))) & 0xf);
            for (int ig = 0; ig < kNg; ++ig) {
                q.values_of((uint16_t) ql[kNg * ib + ig], s, y);
                for (int j = 0; j < kGroupSize; ++j) {
                    const int jr = ig * kGroupSize + j;
                    if (qh[4 * ib + jr / 8] & (1 << (jr & 7))) y[j] = -y[j];
                    if (post != 1.f) y[j] *= post;
                }
                y += kGroupSize;
            }
        }
    }
}

float dot_row(int type, const void* src, const void* q8k, int n) {
    if (n <= 0 || (n % QK_K_) != 0) throw std::runtime_error("ikq: dot width");
    if (type != kIQ2KT && type != kIQ3KT) throw std::runtime_error("ikq: dot type");
    const auto& q = quantizer(type == kIQ3KT);
    const uint8_t* row = (const uint8_t*) src;
    float d = 0, post = 1;
    row_scale(type, *reinterpret_cast<const float*>(row) * kTrellisScale, d, post);
    const int stride = type == kIQ2KT ? 68 : 100;
    const uint8_t* q8 = (const uint8_t*) q8k;
    constexpr int kQ8Bytes = 4 + QK_K_ + QK_K_ / 8;   // d, qs[256], bsums[16]
    alignas(32) float w[QK_K_];
    float total = 0.f;
    for (int ib = 0; ib < n / QK_K_; ++ib) {
        decode_block(type, row + 4 + (int64_t) ib * stride, d, q, w, post);
        const uint8_t* qb = q8 + (int64_t) ib * kQ8Bytes;
        const float dq = *reinterpret_cast<const float*>(qb);
        const int8_t* qs = (const int8_t*) (qb + 4);
        float acc = 0.f;
#if defined(__AVX2__)
        __m256 accv = _mm256_setzero_ps();
        for (int j = 0; j < QK_K_; j += 8) {
            const __m256 vw = _mm256_load_ps(w + j);
            const __m128i b8 = _mm_loadl_epi64((const __m128i*) (qs + j));
            const __m256 vq = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8));
            accv = _mm256_fmadd_ps(vw, vq, accv);
        }
        const __m128 lo = _mm256_castps256_ps128(accv);
        const __m128 hi = _mm256_extractf128_ps(accv, 1);
        __m128 sm = _mm_add_ps(lo, hi);
        sm = _mm_add_ps(sm, _mm_movehl_ps(sm, sm));
        sm = _mm_add_ss(sm, _mm_shuffle_ps(sm, sm, 1));
        acc = _mm_cvtss_f32(sm);
#else
        for (int j = 0; j < QK_K_; ++j) acc += w[j] * (float) qs[j];
#endif
        total += acc * dq;
    }
    return total;
}

size_t quantize_rows(int type, const float* src, void* dst, int64_t nrows, int64_t n_per_row,
                     const float* imatrix, int nthreads) {
    if (type != kIQ2KT && type != kIQ3KT) throw std::runtime_error("ikq: quantize type");
    const size_t row_size = row_bytes(type, n_per_row);
    if (!row_size) throw std::runtime_error("ikq: quantize width");
    if (type == kIQ3KT && (n_per_row % QK_K_) != 0) {
        // ik_llama.cpp's IQ3_KT tail path zero-fills the importance pad, so the blocks past the real end
        // have no weights and its own quantizer aborts on them.  MiniMax's expert widths (3072, 1536) are
        // whole super-blocks, so refuse the narrow case instead of producing something unverifiable.
        throw std::runtime_error("ikq: IQ3_KT needs a whole number of 256-value super-blocks per row");
    }
    (void) quantizer(type == kIQ3KT);   // build the codebook once, before the workers start
    const int64_t padded = (n_per_row + QK_K_ - 1) / QK_K_ * QK_K_;
    const bool whole = (n_per_row % QK_K_) == 0;

    auto one = [&](const float* row_src, void* row_dst, const float* row_imx) {
        std::vector<float> scales((size_t) (padded / kBlockSize));
        std::vector<float> weights((size_t) padded);
        if (whole) {
            if (type == kIQ2KT) {
                std::vector<int> iidx((size_t) (padded / kGroupSize));
                quantize_row_iq2_kt_impl(row_src, row_dst, (int) n_per_row, row_imx, scales.data(),
                                         weights.data(), iidx.data());
            } else {
                std::vector<float> qtmp((size_t) padded);
                quantize_row_iq3_kt_impl(row_src, row_dst, (int) n_per_row, row_imx, scales.data(),
                                         weights.data(), qtmp.data());
            }
            return;
        }
        std::vector<float> xpad((size_t) padded);
        // ik's tail paths differ between the two formats: IQ2_KT repeats the importance row into the pad
        // and quantizes the padded width; IQ3_KT zero-fills the importance pad and quantizes the real width.
        std::vector<float> wpad;
        if (row_imx) {
            wpad.resize((size_t) padded);
            if (type == kIQ2KT) {
                pad_row(row_imx, wpad.data(), n_per_row);       // ik repeats the tail
            } else {
                std::memcpy(wpad.data(), row_imx, (size_t) n_per_row * sizeof(float));   // ik zero-fills
            }
        }
        std::vector<char> ypad(row_bytes(type, padded));
        pad_row(row_src, xpad.data(), n_per_row);
        const int width = type == kIQ2KT ? (int) padded : (int) n_per_row;
        if (type == kIQ2KT) {
            std::vector<int> iidx((size_t) (padded / kGroupSize));
            quantize_row_iq2_kt_impl(xpad.data(), ypad.data(), width, row_imx ? wpad.data() : nullptr,
                                     scales.data(), weights.data(), iidx.data());
        } else {
            std::vector<float> qtmp((size_t) padded);
            quantize_row_iq3_kt_impl(xpad.data(), ypad.data(), width, row_imx ? wpad.data() : nullptr,
                                     scales.data(), weights.data(), qtmp.data());
        }
        finish_row(type, ypad.data(), row_dst, n_per_row);
    };

    if (nthreads < 1) nthreads = 1;
    if ((int64_t) nthreads > nrows) nthreads = (int) nrows;
    if (nthreads == 1) {
        for (int64_t r = 0; r < nrows; ++r)
            one(src + r * n_per_row, (uint8_t*) dst + (size_t) r * row_size,
                imatrix ? imatrix : nullptr);
        return (size_t) nrows * row_size;
    }
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors((size_t) nthreads);
    const int64_t chunk = (nrows + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t a = (int64_t) t * chunk;
        if (a >= nrows) break;
        const int64_t b = std::min(nrows, a + chunk);
        workers.emplace_back([&, t, a, b] {
            try {
                for (int64_t r = a; r < b; ++r)
                    one(src + r * n_per_row, (uint8_t*) dst + (size_t) r * row_size,
                        imatrix ? imatrix : nullptr);
            } catch (...) {
                errors[(size_t) t] = std::current_exception();
            }
        });
    }
    for (auto& w : workers) w.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
    return (size_t) nrows * row_size;
}

}  // namespace strata::kernels::ikq
