// include/strata/kernels/cpu/iq_mt.hpp - one weight row, several activation rows: the CPU expert pass of a
// speculative window (P13).
//
// A verify window routes each expert to several tokens, and `moe_batch_cpu` used to call ggml's one-pair `vec_dot`
// once per (row, token) - i.e. the IQ2_XS 2-bit grid was looked up again for every token that shares the row.  This
// kernel decodes a row's block ONCE (`iq2xs_grid` lookups, the scale bytes, the sign masks) and then runs the same
// integer multiply-accumulate for each token, so each column's result is BIT-IDENTICAL to ggml's own `vec_dot` on
// that column: the per-column arithmetic is unchanged, only the row decode is shared.
//
// Maya's `MAYA_GLM_CPU_MT` is the same idea (docs/SETTINGS.md there); this is Strata's own implementation against the
// pinned ggml's kernels, checked bit for bit by src/kernels/cpu/iq_mt_test.cpp.
#pragma once

#include <cstdint>

namespace strata::kernels::mt {

/// Whether `dot_rows` has a kernel for this ggml type (IQ2_XS today; the other formats fall back to ggml's one-pair
/// dots, which are unchanged by this header).
bool supported(int ggml_type);

/// out[c * ldy] = <row, aq[c]> for c in 0..ncols-1, `n` values per row.  `aq[c]` is the type's `vec_dot`
/// activation block (Q8_K for IQ2_XS), quantized exactly as the one-token path quantizes it.  Every entry is
/// bit-identical to `ggml_vec_dot_iq2_xs_q8_K(n, &s, 0, row, 0, aq[c], 0, 1)`.
void dot_rows(int ggml_type, const void* row, const uint8_t* const* aq, int ncols, float* out, int64_t ldy, int n);

/// The widest window this kernel takes (the caller falls back above it).
constexpr int kMaxCols = 8;

}  // namespace strata::kernels::mt
