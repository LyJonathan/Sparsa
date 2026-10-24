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
// pair). A raw byte copy is not the spec tensor; the nibbles must be remapped.
// ggml's dequantize_row_nvfp4 multiplies doubled E2M1 codes by e4m3(scale)*0.5,
// so it returns e2m1 * e4m3(scale) exactly, with no G.
//
// This tool therefore stores G as a side tensor, one f32 per expert
// (`blk.N.ffn_*_exps.scale`), and the scale bytes are copied verbatim. Folding
// e4m3(scale)/G into a new E4M3 byte is NOT lossless here (layer 3 expert 0 w1:
// G=15123.6923828125, scale byte 113 = 144, quotient 0.00952148438, nearest
// E4M3 = 0.009765625, a +2.6% gain error), so it is not done.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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
// Legacy folding form. MM-NVFP4 does not use it: see pack_nvfp4_row_exact.
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

// The lossless MM-NVFP4 form: nibble remap and a verbatim copy of each E4M3 scale
// byte. The per-tensor global scale (a divisor) is written separately as f32.
// n_in % 64 == 0; dst holds (n_in/64)*36 bytes. Cannot fail.
//
// For one 16-value subgroup the checkpoint's 8 source bytes hold the elements as
// consecutive pairs; ggml wants out[j] = elem j | elem j+8 << 4. out[2b] and
// out[2b+1] depend only on src[b] and src[b+4], so one 64 KiB table maps those
// two bytes to the two output bytes (4 lookups per 16 values).
inline const uint16_t * nvfp4_nibble_lut() {
    static const std::vector<uint16_t> lut = [] {
        std::vector<uint16_t> t(65536, 0);
        for (int a = 0; a < 256; ++a) {
            for (int b = 0; b < 256; ++b) {
                const int x = a | (b << 8);
                const uint8_t out0 = static_cast<uint8_t>((x & 0x0F) | (((x >> 8) & 0x0F) << 4));
                const uint8_t out1 = static_cast<uint8_t>(((x >> 4) & 0x0F) | ((x >> 8) & 0xF0));
                t[static_cast<size_t>(x)] = static_cast<uint16_t>(out0 | (out1 << 8));
            }
        }
        return t;
    }();
    return lut.data();
}

inline void pack_nvfp4_row_exact(const uint8_t * packed, const uint8_t * scales, int64_t n_in, uint8_t * dst) {
    const uint16_t * lut = nvfp4_nibble_lut();
    const int64_t nblk = n_in / 64;
    for (int64_t i = 0; i < nblk; ++i) {
        uint8_t * blk = dst + i * 36;
        for (int s = 0; s < 4; ++s) {
            blk[s] = scales[i * 4 + s];            // E4M3 byte, unchanged
            const uint8_t * src = packed + (i * 64 + s * 16) / 2;
            for (int b = 0; b < 4; ++b) {
                const uint16_t v = lut[static_cast<size_t>(src[b]) | (static_cast<size_t>(src[b + 4]) << 8)];
                blk[4 + s * 8 + 2 * b] = static_cast<uint8_t>(v & 0xFF);
                blk[4 + s * 8 + 2 * b + 1] = static_cast<uint8_t>(v >> 8);
            }
        }
    }
}

// Dequantize one ggml NVFP4 row (the exact form above) with the side global scale.
// `global_scale` is the divisor from blk.N.ffn_*_exps.scale[expert]. This is the
// same expression as dequant_nvfp4_row: (e2m1 * e4m3(scale)) / G.
inline void dequant_ggml_nvfp4_row(const uint8_t * blocks, float global_scale, int64_t n_in, float * dst) {
    const int64_t nblk = n_in / 64;
    for (int64_t i = 0; i < nblk; ++i) {
        const uint8_t * blk = blocks + i * 36;
        for (int s = 0; s < 4; ++s) {
            const float sc = e4m3_to_f32(blk[s]);
            for (int j = 0; j < 8; ++j) {
                const uint8_t b = blk[4 + s * 8 + j];
                dst[i * 64 + s * 16 + j] = e2m1_to_f32(static_cast<uint8_t>(b & 0x0F)) * sc / global_scale;
                dst[i * 64 + s * 16 + j + 8] =
                    e2m1_to_f32(static_cast<uint8_t>(b >> 4)) * sc / global_scale;
            }
        }
    }
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
