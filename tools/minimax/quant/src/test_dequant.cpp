#include "dequant.hpp"

#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace mmq;

static int g_fail = 0;

static void expect(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", msg);
        ++g_fail;
    } else {
        std::printf("ok   %s\n", msg);
    }
}

static void expect_eq(float got, float want, const char * msg) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s got %.9g want %.9g\n", msg, got, want);
        ++g_fail;
    } else {
        std::printf("ok   %s\n", msg);
    }
}

int main() {
    expect_eq(e4m3_to_f32(0x38), 1.0f, "E4M3 0x38 == 1");
    expect_eq(e4m3_to_f32(0xB8), -1.0f, "E4M3 0xB8 == -1");
    expect_eq(e4m3_to_f32(0x00), 0.0f, "E4M3 0 == 0");
    expect_eq(e4m3_to_f32(0x7F), 0.0f, "E4M3 NaN decodes as 0");
    expect_eq(e4m3_to_f32(0x30), 0.5f, "E4M3 0x30 == 0.5");
    expect_eq(e2m1_to_f32(0x1), 0.5f, "E2M1 nibble 1 == 0.5");
    expect_eq(e2m1_to_f32(0x2), 1.0f, "E2M1 nibble 2 == 1");
    expect_eq(e2m1_to_f32(0x9), -0.5f, "E2M1 nibble 9 == -0.5");

    // Spec A: packed 0x21 is even=0.5, odd=1.0. Scale 1, G=2 -> 0.25 and 0.5.
    {
        const uint8_t packed[1] = {0x21};
        const uint8_t scale[1] = {0x38};
        float dst[2] = {};
        dequant_nvfp4_row(packed, scale, 2.0f, 2, dst);
        expect_eq(dst[0], 0.25f, "NVFP4 0x21 G=2 elem0");
        expect_eq(dst[1], 0.5f, "NVFP4 0x21 G=2 elem1");
    }

    // Spec B: weight 1.0, UE8M0 130 -> 2^(130-127)=8.
    {
        uint8_t w[32];
        uint8_t s[1] = {130};
        for (int i = 0; i < 32; ++i) {
            w[i] = 0x38;
        }
        float dst[32];
        dequant_mxfp8_row(w, s, 32, dst);
        expect_eq(dst[0], 8.0f, "MXFP8 0x38 * 2^3 == 8");
        expect_eq(dst[31], 8.0f, "MXFP8 last of block == 8");
    }

    {
        const uint16_t one = 0x3f80;
        expect_eq(bf16_to_f32(one), 1.0f, "BF16 0x3f80 == 1");
        float dst[1];
        dequant_bf16_row(&one, 1, dst, 1.0f);
        expect_eq(dst[0], 2.0f, "Gemma norm bake +1");
    }

    // G=2, scale 1.0 folds onto E4M3 0.5. G=3 does not. The measured checkpoint
    // pair (scale byte 113 -> 144, G=15123.6923828125) is not on the grid.
    {
        const E4Fit g2 = fit_e4m3(e4m3_to_f32(0x38) / 2.0f);
        expect(g2.exact && g2.byte == 0x30, "G=2 scale 1.0 folds to E4M3 0.5");
        const E4Fit g3 = fit_e4m3(e4m3_to_f32(0x38) / 3.0f);
        expect(!g3.exact, "G=3 scale 1.0 is not E4M3-exact");
        const float G = 15123.6923828125f;
        const E4Fit real = fit_e4m3(e4m3_to_f32(113) / G);
        expect(e4m3_to_f32(113) == 144.0f, "scale byte 113 decodes to 144");
        expect(!real.exact, "checkpoint G does not fold scale 113 into E4M3");
        std::printf("     real target %.9g nearest E4M3 byte %d decoded %.9g\n",
                    real.target, real.byte, real.decoded);
    }

    // One ggml block. Nibbles are not stored as consecutive pairs.
    {
        uint8_t packed[32];
        uint8_t scales[4] = {0x38, 0x30, 0x40, 0x28};
        for (int i = 0; i < 32; ++i) {
            packed[i] = static_cast<uint8_t>(((i * 3) & 0xF) | (((i * 5 + 1) & 0xF) << 4));
        }
        uint8_t blk[36];
        expect(pack_nvfp4_row(packed, scales, 1.0f, 64, blk), "G=1 row packs exactly");
        const uint8_t elem0 = packed[0] & 0x0F;
        const uint8_t elem8 = packed[4] & 0x0F;
        expect(blk[4] == static_cast<uint8_t>(elem0 | (elem8 << 4)),
               "ggml nibble is elem j | elem j+8, not a consecutive pair");
        expect(blk[4] != packed[0], "remap changes the first packed byte");

        float got[64];
        float spec[64];
        dequantize_row_nvfp4(reinterpret_cast<const block_nvfp4 *>(blk), got, 64);
        dequant_nvfp4_row(packed, scales, 1.0f, 64, spec);
        bool match = true;
        for (int i = 0; i < 64; ++i) {
            if (got[i] != spec[i]) {
                std::fprintf(stderr, "FAIL roundtrip i=%d got %.9g spec %.9g\n", i, got[i], spec[i]);
                match = false;
                break;
            }
        }
        expect(match, "dequantize_row_nvfp4 == spec A at G=1");
        expect(ggml_validate_row_data(GGML_TYPE_NVFP4, blk, sizeof(blk)), "packed NVFP4 block validates");
        static_assert(sizeof(block_nvfp4) == 36, "nvfp4 block is 36 bytes");
    }

    {
        uint8_t packed[32] = {};
        uint8_t scales[4] = {0x38, 0x38, 0x38, 0x38};
        uint8_t blk[36];
        expect(!pack_nvfp4_row(packed, scales, 3.0f, 64, blk), "packer refuses G=3");
    }

    // Exact MM-NVFP4 packing: verbatim E4M3 scale bytes + nibble remap, G kept aside.
    // The GGUF block dequantized through the side divisor must equal spec A bit for bit.
    {
        uint8_t packed[32];
        uint8_t scales[4] = {0x71, 0x81, 0x6A, 0x90};   // 144.0, 0.012... (as in the checkpoint), ...
        for (int i = 0; i < 32; ++i) {
            packed[i] = static_cast<uint8_t>(((i * 7 + 3) & 0x0F) | (((i * 11 + 5) & 0x0F) << 4));
        }
        uint8_t blk[36];
        pack_nvfp4_row_exact(packed, scales, 64, blk);
        bool copied = true;
        for (int s = 0; s < 4; ++s) {
            if (blk[s] != scales[s]) {
                copied = false;
            }
        }
        expect(copied, "exact pack copies every E4M3 scale byte verbatim");

        const float G = 15123.6923828125f;
        float got[64];
        float spec[64];
        dequant_ggml_nvfp4_row(blk, G, 64, got);
        dequant_nvfp4_row(packed, scales, G, 64, spec);
        bool match = true;
        for (int i = 0; i < 64; ++i) {
            if (got[i] != spec[i]) {
                std::fprintf(stderr, "FAIL exact pack i=%d got %.9g spec %.9g\n", i, got[i], spec[i]);
                match = false;
                break;
            }
        }
        expect(match, "exact pack: block dequant / G == spec A dequant (0 diff, G=15123.69)");
        expect(ggml_validate_row_data(GGML_TYPE_NVFP4, blk, sizeof(blk)), "exact NVFP4 block validates");
    }

    if (g_fail) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("dequant tests passed\n");
    std::printf("NVFP4: ggml type id %d. MM-NVFP4 copies nibbles (remapped) and E4M3 scale bytes verbatim and\n"
                "stores weight_global_scale as blk.N.ffn_*_exps.scale (f32 [n_expert]). No fold, no gain error.\n",
                GGML_TYPE_NVFP4);
    return 0;
}
