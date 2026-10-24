// include/strata/kernels/cpu/iqk_quants.hpp - codec for the quant formats ported from ik_llama.cpp.
//
// Ported from ik_llama.cpp, commit 804ee29 (MIT).  Copyright (c) 2023-2024 The ggml authors,
// Copyright (c) 2023-2024 The llama.cpp authors, Copyright (c) 2024-2025 The ik_llama.cpp authors.
// See THIRD_PARTY_NOTICES.md.  The codec here is a port, not a copy of the runtime: it keeps ik's
// algorithms, codebooks and tie-breaking so its output is byte-identical.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::ikq {

/// IQ2_KT carries a run-time factor of 1.05 in ik_llama.cpp's matmul that its `to_float` omits; IQ3_KT's
/// `to_float` already has its 1.01.  Strata folds both into `dequantize_row` and `dot_row`, so the two agree
/// and a reference built from ik's `to_float` has to multiply by this to match.  It is 1.0 for IQ3_KT.
float dequant_factor(int type);

/// Dequantize `n` values of one packed row into `dst`.  Bit-exact against ik's codebook, with its run-time
/// factor folded in (see `dequant_factor`).
void dequantize_row(int type, const void* src, float* dst, int64_t n);

/// Dot of one packed row against a ggml `block_q8_K` activation row (4 + 256 + 32 bytes per block), with the
/// same factor `dequantize_row` uses.  `n` is a multiple of 256; the caller rounds wider rows.
float dot_row(int type, const void* src, const void* q8k, int n);

/// Quantize `nrows` rows of `n_per_row` f32 values, `imatrix` being one row of importance weights (may be
/// null).  Requires AVX2, like ik's own quantizer for these formats.  Returns the bytes written, 0 on a
/// width this format cannot hold.
size_t quantize_rows(int type, const float* src, void* dst, int64_t nrows, int64_t n_per_row,
                     const float* imatrix, int nthreads);

}  // namespace strata::kernels::ikq
