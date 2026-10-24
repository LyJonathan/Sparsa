#pragma once

// The trellis types (TQ2_T 144, TQK6 145, TQK7 146) as the quantizer sees them: a name, a block layout and an
// encoder.  ggml (a pinned upstream llama.cpp) has no entry for ids 144-146, so every ggml call that would index
// its type table goes through here instead.

#include "strata/kernels/cpu/tq_encode.hpp"

#include <ggml.h>

#include <cstdint>
#include <string>

namespace mmq {

inline bool is_tq(ggml_type t) {
    const int id = static_cast<int>(t);
    return id == 144 || id == 145 || id == 146;
}

inline const char * type_name(ggml_type t) {
    if (is_tq(t)) {
        return strata::kernels::cpu::tq_encode_type_name(static_cast<int>(t));
    }
    return ggml_type_name(t);
}

inline int blck_size(ggml_type t) {
    return is_tq(t) ? 128 : static_cast<int>(ggml_blck_size(t));
}

inline size_t row_size(ggml_type t, int64_t n_in) {
    if (is_tq(t)) {
        return strata::kernels::cpu::tq_encode_row_bytes(static_cast<int>(t), n_in);
    }
    return ggml_row_size(t, n_in);
}

/// Bytes per 128 weights of a trellis type, 0 otherwise.
inline size_t tq_block_bytes(ggml_type t) {
    return is_tq(t) ? strata::kernels::cpu::tq_encode_block_bytes(static_cast<int>(t)) : 0;
}

}  // namespace mmq
