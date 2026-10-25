// src/kernels/cpu/iq_mt_test.cpp - the P13 gate for the multi-token CPU expert row: a column of
// `mt::dot_rows` must equal ggml's own one-pair `vec_dot` for that column, bit for bit (0 ULP).
//
// This is the property the speculative window depends on: if the shared row decode changed even one bit of a
// column's sum, a verify window would be verifying with different arithmetic than the token path and the "greedy is
// the same tokens" claim would be false.  The test builds random rows (quantized with ggml's own quantizer), random
// activations (quantized to the type's vec_dot type with ggml's own from_float, exactly as the engine does), and
// compares 1..8 columns against ggml's dot, plus a same-placement check of the whole window (ggml's per-column loop
// against one MT call, the way gemm_cpu_base used to call it).
//
//   iq_mt_test            # a quick run over the real widths (3072, 6144) and a few hundred rows
//   iq_mt_test --rows 4000 --tol-time
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

#include "strata/kernels/cpu/iq_mt.hpp"

namespace {

int64_t failures = 0, columns = 0, ulp_diffs = 0;

/// The bit pattern of two floats, for the diagnostic when they differ.
uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

/// The engine's one-token path for one (row, activation): ggml's vec_dot, the same bytes and the same call.
float ggml_dot(const ggml_type_traits_cpu* tr, const uint8_t* row, const uint8_t* act, int n) {
    float s = 0.f;
    tr->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

int run(int64_t n_rows, int64_t width, int max_cols, std::mt19937& rng) {
    const ggml_type type = GGML_TYPE_IQ2_XS;
    if (!strata::kernels::mt::supported(type)) {
        std::printf("iq_mt: no multi-token kernel for IQ2_XS in this build\n");
        return 2;
    }
    const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(type);
    if (!tr || !tr->vec_dot) {
        std::printf("iq_mt: ggml has no IQ2_XS vec_dot\n");
        return 2;
    }
    // IQ2_XS itself has no from_float (its quantizer needs an init); the ACTIVATION side is Q8_K, whose from_float
    // is what gemm_cpu_base quantizes with.
    const ggml_type_traits_cpu* act = ggml_get_type_traits_cpu(tr->vec_dot_type);
    if (!act || !act->from_float) {
        std::printf("iq_mt: ggml has no Q8_K from_float\n");
        return 2;
    }
    ggml_quantize_init(type);
    const size_t row_bytes = ggml_row_size(type, width);
    const size_t act_bytes = ggml_row_size(tr->vec_dot_type, width);

    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_int_distribution<int> pick_cols(1, max_cols);
    std::vector<float> row_f((size_t)width), src((size_t)width), imatrix((size_t)width, 1.f);
    std::vector<uint8_t> row(row_bytes), actbuf((size_t)act_bytes * (size_t)max_cols);
    std::vector<const uint8_t*> ptrs((size_t)max_cols);
    std::vector<float> got((size_t)max_cols), want((size_t)max_cols);

    for (int64_t r = 0; r < n_rows; ++r) {
        // a weight row of this shape, quantized by ggml itself
        const float scale = 0.005f + 0.05f * std::abs(u(rng));
        for (int64_t j = 0; j < width; ++j) row_f[(size_t)j] = scale * u(rng);
        ggml_quantize_chunk(type, row_f.data(), row.data(), 0, 1, width, imatrix.data());

        const int ncols = pick_cols(rng);
        for (int c = 0; c < ncols; ++c) {
            // an activation row, quantized to the vec_dot type exactly as gemm_cpu_base quantizes it
            for (int64_t j = 0; j < width; ++j) src[(size_t)j] = u(rng) * 3.f;
            act->from_float(src.data(), actbuf.data() + (size_t)c * act_bytes, width);
            ptrs[(size_t)c] = actbuf.data() + (size_t)c * act_bytes;
        }
        // the reference: ggml's one-pair dot per column, in the order gemm_cpu_base used to call it
        for (int c = 0; c < ncols; ++c) want[(size_t)c] = ggml_dot(tr, row.data(), ptrs[(size_t)c], (int)width);
        // the candidate: one call, the row decoded once
        for (int c = 0; c < max_cols; ++c) got[(size_t)c] = -12345.f;
        strata::kernels::mt::dot_rows(type, row.data(), ptrs.data(), ncols, got.data(), 1, (int)width);
        for (int c = 0; c < ncols; ++c) {
            ++columns;
            if (bits(got[(size_t)c]) != bits(want[(size_t)c])) {
                if (failures < 5)
                    std::printf("MISMATCH width=%lld ncols=%d col=%d: mt=%.9g (0x%08x) ggml=%.9g (0x%08x)\n",
                                (long long)width, ncols, c, got[(size_t)c], bits(got[(size_t)c]),
                                want[(size_t)c], bits(want[(size_t)c]));
                ++failures;
                if (std::fabs(got[(size_t)c] - want[(size_t)c]) > 0.f) ++ulp_diffs;
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int64_t rows = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows = std::atoll(argv[++i]);
    }
    ggml_cpu_init();
    std::mt19937 rng(20261025u);
    // the widths the MiniMax expert matrices use, and the window widths the engine verifies
    const int64_t widths[] = {3072, 6144};
    for (int64_t w : widths) {
        for (int cols : {8, 4, 2, 1}) {
            const int rc = run(rows, w, cols, rng);
            if (rc) return rc;
        }
    }
    std::printf("iq_mt_test: %lld columns over IQ2_XS rows at 3072 and 6144 values, %lld mismatches (%lld with a "
                "value difference)\n",
                (long long)columns, (long long)failures, (long long)ulp_diffs);
    std::printf("%s\n", failures == 0 ? "PASS (bit-identical to ggml's one-pair vec_dot)"
                                      : "FAIL (the multi-token row is not the one-token arithmetic)");
    return failures == 0 ? 0 : 1;
}
