// src/kernels/cpu/iq_mt.cpp - see include/strata/kernels/cpu/iq_mt.hpp.
//
// This is ggml's `ggml_vec_dot_iq2_xs_q8_K` (AVX2 branch, ggml/src/ggml-cpu/arch/x86/quants.c at the pinned commit)
// with the row's decode hoisted out of the column loop: the 2-bit grid lookups, the sign masks and the scale bytes
// are produced once per block and then reused by every token of the window.  The per-column instruction sequence -
// `maddubs` on (grid, signed q8), `madd` with the sub-block scales, sumi1/sumi2, the per-block `fma` into an 8-lane
// float accumulator, and the single horizontal sum at the end - is transcribed unchanged, so a column's result is
// bit-identical to ggml's one-pair dot.  `src/kernels/cpu/iq_mt_test.cpp` asserts exactly that (0 ULP) against the
// pinned ggml, and the engine's own gate (`gate_logits`) compares a window against the token path.
#include "strata/kernels/cpu/iq_mt.hpp"

#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml.h"

namespace strata::kernels::mt {
namespace {

constexpr int QK_K_ = 256;

#if defined(__AVX2__)

/// ggml's horizontal sum of 8 floats (the order of the adds is part of the result).
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/// ggml's scale-byte shuffle for sub-block `i` (four copies of the i-th scale byte per 16-bit lane pair).
inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

/// The row's decode for one 32-value sub-block group: the unsigned 2-bit grid values and the q8 sign multiplier.
struct Group {
    __m256i q2;    ///< 32 grid bytes (maddubs' unsigned side)
    __m256i sgn;   ///< ±1 per value: ggml's `or(signs, mone)`, the multiplier `sign_epi8` takes
    __m256i sc;    ///< the sub-block's scale, int16 per pair
};

#endif  // __AVX2__

}  // namespace

bool supported(int ggml_type) {
#if defined(__AVX2__)
    return ggml_type == GGML_TYPE_IQ2_XS;
#else
    (void)ggml_type;
    return false;
#endif
}

void dot_rows(int ggml_type, const void* row, const uint8_t* const* aq, int ncols, float* out, int64_t ldy, int n) {
#if defined(__AVX2__)
    if (ggml_type != GGML_TYPE_IQ2_XS) return;
    if (ncols < 1 || ncols > kMaxCols || (n % QK_K_) != 0) return;
    const int nb = n / QK_K_;
    const block_iq2_xs* x = (const block_iq2_xs*)row;

    const __m256i mone = _mm256_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*)bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_2);
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*)k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;
    __m256i aux_gindex;
    const uint16_t* gindex = (const uint16_t*)&aux_gindex;

    // one accumulator per column, exactly where ggml keeps its single one
    __m256 accumf[kMaxCols];
    for (int c = 0; c < ncols; ++c) accumf[c] = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float dr = ggml_fp16_to_fp32(x[i].d);
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8s[kMaxCols];
        float dtok[kMaxCols];
        for (int c = 0; c < ncols; ++c) {
            const block_q8_K* y = (const block_q8_K*)aq[c];
            q8s[c] = y[i].qs;
            dtok[c] = dr * y[i].d;
        }

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1[kMaxCols], sumi2[kMaxCols];
        for (int c = 0; c < ncols; ++c) {
            sumi1[c] = _mm256_setzero_si256();
            sumi2[c] = _mm256_setzero_si256();
        }

        for (int ib32 = 0; ib32 < QK_K_ / 32; ib32 += 4) {
            // ---- the row's decode, once for all columns
            const __m256i q2_data = _mm256_loadu_si256((const __m256i*)q2);
            q2 += 16;
            aux_gindex = _mm256_and_si256(q2_data, m511);
            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting =
                _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);
            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = _mm256_set_m128i(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = _mm256_set_m128i(full_signs_h, full_signs_h);

            Group g[4];
            g[0].q2 = _mm256_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]], iq2xs_grid[gindex[1]],
                                        iq2xs_grid[gindex[0]]);
            g[1].q2 = _mm256_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]], iq2xs_grid[gindex[5]],
                                        iq2xs_grid[gindex[4]]);
            g[2].q2 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]], iq2xs_grid[gindex[9]],
                                        iq2xs_grid[gindex[8]]);
            g[3].q2 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]], iq2xs_grid[gindex[13]],
                                        iq2xs_grid[gindex[12]]);
            __m256i signs;
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            g[0].sgn = _mm256_or_si256(signs, mone);
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            g[1].sgn = _mm256_or_si256(signs, mone);
            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            g[2].sgn = _mm256_or_si256(signs, mone);
            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            g[3].sgn = _mm256_or_si256(signs, mone);
            g[0].sc = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 0)));
            g[1].sc = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 1)));
            g[2].sc = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 2)));
            g[3].sc = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 3)));

            // ---- one pass per column: the arithmetic is ggml's, unchanged
            for (int c = 0; c < ncols; ++c) {
                const int8_t* q8 = q8s[c];
                for (int k = 0; k < 4; ++k) {
                    const __m256i q8k = _mm256_loadu_si256((const __m256i*)(q8 + 32 * k));
                    const __m256i q8sk = _mm256_sign_epi8(q8k, g[k].sgn);
                    const __m256i dot = _mm256_maddubs_epi16(g[k].q2, q8sk);
                    if ((k & 1) == 0)
                        sumi1[c] = _mm256_add_epi32(sumi1[c], _mm256_madd_epi16(dot, g[k].sc));
                    else
                        sumi2[c] = _mm256_add_epi32(sumi2[c], _mm256_madd_epi16(dot, g[k].sc));
                }
                q8s[c] += 128;
            }
        }

        for (int c = 0; c < ncols; ++c) {
            const __m256i sumi = _mm256_add_epi32(sumi1[c], sumi2[c]);
            accumf[c] = _mm256_fmadd_ps(_mm256_set1_ps(dtok[c]), _mm256_cvtepi32_ps(sumi), accumf[c]);
        }
    }

    for (int c = 0; c < ncols; ++c) out[(int64_t)c * ldy] = 0.125f * hsum_float_8(accumf[c]);
#else
    (void)ggml_type;
    (void)row;
    (void)aq;
    (void)ncols;
    (void)out;
    (void)ldy;
    (void)n;
#endif
}

}  // namespace strata::kernels::mt
