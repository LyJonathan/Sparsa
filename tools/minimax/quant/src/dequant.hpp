#pragma once

// MiniMax-M3.1 weight dequant and the ggml NVFP4 remap.
//
// Spec A (NVFP4): packed U8 [O, I/2], low nibble = even input index, high nibble = odd.
// E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign bit 0x8. Scale is F8_E4M3
// [O, I/16] (group 16). weight_global_scale is a divisor:
//   W = e2m1 * e4m3(scale) / G
// input_global_scale is activation-only and is ignored.
//
// Spec B (MXFP8): weight F8_E4M3 [O, I], weight_scale_inv UE8M0 uint8 [O, I/32]:
//   W = e4m3(weight) * 2^(scale - 127)
// Scale byte 255 is the UE8M0 NaN and is not expected in these checkpoints.
//
// ggml GGML_TYPE_NVFP4 (type id 40) is a different packing of the same codes:
// QK 64, block = 4 UE4M3 scales + 32 bytes. Inside each 16-wide subgroup the
// low nibble is element j and the high nibble is element j+8 (not a consecutive
// pair). ggml's kvalues are doubled E2M1 and ggml_ue4m3_to_fp32 returns e4m3*0.5,
// so a raw byte copy is not the spec tensor. There is no field for G. Folding
// e4m3(scale)/G into a new E4M3 byte is lossless only when that quotient is on
// the E4M3 grid. Otherwise the tier is refused. No side channel is invented.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mmq {

// Signed FP8 E4M3 (bias 7). 0x7F and 0xFF are NaN and decode as 0.
// Positive codes match ggml's unsigned UE4M3 grid, which is what block_nvfp4 stores.
inline float e4m3_to_f32(uint8_t x) {
    const uint8_t mag = static_cast<uint8_t>(x & 0x7F);
    if (mag == 0 || mag == 0x7F) {
        return 0.0f;
    }
    const int exp = (mag >> 3) & 0xF;
    const int man = mag & 0x7;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -9);
    } else {
        v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    }
    return (x & 0x80) ? -v : v;
}

inline float e2m1_to_f32(uint8_t nib) {
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float mag = kMag[nib & 7];
    return (nib & 8) ? -mag : mag;
}

inline float ue8m0_to_f32(uint8_t b) {
    if (b == 255) {
        return std::nanf("");
    }
    return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct E4Fit {
    bool exact = false;
    int byte = 0;
    float decoded = 0.f;
    float target = 0.f;
};

// Exact ggml UE4M3 membership (bytes 0..126, NaN 0x7F skipped).
// block_nvfp4 stores an unsigned scale; a signed E4M3 code would not round-trip.
inline E4Fit fit_e4m3(float target) {
    E4Fit best;
    best.target = target;
    float best_abs = 1e30f;
    for (int b = 0; b < 128; ++b) {
        if (b == 0x7F) {
            continue;
        }
        const float v = e4m3_to_f32(static_cast<uint8_t>(b));
        if (v == target) {
            best.exact = true;
            best.byte = b;
            best.decoded = v;
            return best;
        }
        const float err = std::fabs(v - target);
        if (err < best_abs) {
            best_abs = err;
            best.byte = b;
            best.decoded = v;
        }
    }
    return best;
}

inline void dequant_nvfp4_row(const uint8_t * packed, const uint8_t * scales, float global_scale,
                              int64_t n_in, float * dst) {
    const int64_t npack = n_in / 2;
    const int64_t nscale = n_in / 16;
    (void)npack;
    (void)nscale;
    for (int64_t c = 0; c < n_in; ++c) {
        const uint8_t byte = packed[c / 2];
        const uint8_t nib = (c & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
        const float s = e4m3_to_f32(scales[c / 16]);
        dst[c] = e2m1_to_f32(nib) * s / global_scale;
    }
}

inline void dequant_mxfp8_row(const uint8_t * weight, const uint8_t * scale_inv, int64_t n_in, float * dst) {
    for (int64_t c = 0; c < n_in; ++c) {
        dst[c] = e4m3_to_f32(weight[c]) * ue8m0_to_f32(scale_inv[c / 32]);
    }
}

inline void dequant_bf16_row(const uint16_t * src, int64_t n, float * dst, float add) {
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = bf16_to_f32(src[i]) + add;
    }
}

// One ggml NVFP4 row. n_in % 64 == 0. dst holds (n_in/64)*36 bytes.
// Returns false when any group scale cannot be stored as E4M3 after / G.
inline bool pack_nvfp4_row(const uint8_t * packed, const uint8_t * scales, float global_scale,
                           int64_t n_in, uint8_t * dst) {
    const int64_t nblk = n_in / 64;
    for (int64_t i = 0; i < nblk; ++i) {
        uint8_t * blk = dst + i * 36;
        for (int s = 0; s < 4; ++s) {
            const uint8_t sb = scales[i * 4 + s];
            const float target = e4m3_to_f32(sb) / global_scale;
            const E4Fit fit = fit_e4m3(target);
            if (!fit.exact) {
                return false;
            }
            blk[s] = static_cast<uint8_t>(fit.byte);
            const uint8_t * src = packed + (i * 64 + s * 16) / 2;
            uint8_t nib[16];
            for (int b = 0; b < 8; ++b) {
                nib[2 * b] = static_cast<uint8_t>(src[b] & 0x0F);
                nib[2 * b + 1] = static_cast<uint8_t>(src[b] >> 4);
            }
            for (int j = 0; j < 8; ++j) {
                blk[4 + s * 8 + j] = static_cast<uint8_t>(nib[j] | (nib[j + 8] << 4));
            }
        }
    }
    return true;
}

inline bool nvfp4_row_exact(const uint8_t * scales, float global_scale, int64_t n_in) {
    const int64_t nscale = n_in / 16;
    for (int64_t i = 0; i < nscale; ++i) {
        const float target = e4m3_to_f32(scales[i]) / global_scale;
        if (!fit_e4m3(target).exact) {
            return false;
        }
    }
    return true;
}

}  // namespace mmq
