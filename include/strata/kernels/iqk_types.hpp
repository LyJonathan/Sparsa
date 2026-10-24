// include/strata/kernels/iqk_types.hpp - the ggml type ids and the row geometry of the quant formats
// ported from ik_llama.cpp, in one place.
//
// ik_llama.cpp (MIT) gives its own formats ids 144/145/146 for IQ4_KS/IQ2_KS/IQ4_KSS and 153..158 for the
// trellis (KT) family.  Strata already uses 144/145/146 for TQ2_T/TQK6/TQK7, so the ported formats get
// fresh ids above every id in use: ggml-org llama.cpp stops at 42, ik_llama.cpp at 158, the trellis branch
// (feat/minimax-trellis) at 146.  160..164 collide with nothing in any of them.
//
// A GGUF written with these ids is readable by Strata and by nothing else: neither ggml-org's llama.cpp nor
// ik_llama.cpp knows the ids, and ik_llama.cpp has no MiniMax-M3 architecture to run them in.
//
// The layouts are ik_llama.cpp's, byte for byte.  Each row starts with a row-scope scale (row_meta bytes)
// followed by ne/256 blocks.  The trellis formats may also carry a partial tail block for rows whose width
// is not a multiple of 256.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::ikq {

// Ported from ik_llama.cpp (MIT). See THIRD_PARTY_NOTICES.md.
inline constexpr int kIQ2KT = 160;  // IQ2_KT: 2.125 bpw trellis, 16-bit group index
inline constexpr int kIQ3K  = 161;  // IQ3_K:  3.4375 bpw (reserved, not implemented yet)
inline constexpr int kIQ4KS = 162;  // IQ4_KS: 4.25 bpw   (reserved, not implemented yet)
inline constexpr int kIQ3KT = 163;  // IQ3_KT: 3.125 bpw trellis, abs. values + sign bits
inline constexpr int kIQ2KL = 164;  // IQ2_KL: 2.6875 bpw (reserved, not implemented yet)

/// Block geometry of a packed format.
///
/// `block_elems` values are packed into `block_bytes` bytes; `row_meta` bytes of per-row scale precede the
/// blocks.  `tail_bytes(type, n)` is the extra size of a partial block when the row width is not a multiple
/// of `block_elems` (0 for formats that need whole blocks).
struct Geometry {
    int block_elems;
    int block_bytes;
    int row_meta;
};

/// True for an id this header owns.
inline bool is_iqk_type(int type) noexcept {
    return type >= kIQ2KT && type <= kIQ2KL;
}

/// The ids that this build implements the codec for.
inline bool has_codec(int type) noexcept {
    return type == kIQ2KT || type == kIQ3KT;
}

inline const char* type_name(int type) noexcept {
    switch (type) {
    case kIQ2KT: return "IQ2_KT";
    case kIQ3K: return "IQ3_K";
    case kIQ4KS: return "IQ4_KS";
    case kIQ3KT: return "IQ3_KT";
    case kIQ2KL: return "IQ2_KL";
    default: return "?";
    }
}

/// Block layout, or false for an id this header does not own or has no codec for.
inline bool geometry(int type, Geometry& g) noexcept {
    switch (type) {
    case kIQ2KT: g = {256, 68, 4}; return true;    // scales[4] + ql[64]
    case kIQ3KT: g = {256, 100, 4}; return true;   // scales[4] + ql[64] + qh[32]
    case kIQ3K:  g = {256, 110, 0}; return true;   // d + extra + scales_h + scales_l[8] + qs[64] + qh[32]
    case kIQ4KS: g = {256, 136, 4}; return true;   // scales[8] + qs[128]
    case kIQ2KL: g = {256, 86, 2}; return true;    // scales_h + scales_l[4] + qs[64] + qh[16]
    default: return false;
    }
}

/// Partial-block payload for a row narrower than one whole block. -1 when `n` is not a legal width.
inline long tail_bytes(int type, int64_t n) noexcept {
    const int nt = (int) ((n % 256) / 32);
    if ((n % 32) != 0) {
        return -1;
    }
    if (nt == 0) {
        return 0;
    }
    switch (type) {
    case kIQ2KT: return 4 + 8 * nt;
    case kIQ3KT: return (nt + 1) / 2 + 12 * nt;
    default: return -1;
    }
}

/// Bytes of one packed row of `n` values, or 0 when the width is not legal for this format.
inline size_t row_bytes(int type, int64_t n) noexcept {
    Geometry g{};
    if (n <= 0 || !geometry(type, g) || (n % 32) != 0) {
        return 0;
    }
    const long t = tail_bytes(type, n);
    if (t < 0) {
        return 0;
    }
    const size_t head = (size_t) g.row_meta + (size_t) g.block_bytes * (size_t) (n / g.block_elems);
    const size_t raw = head + (size_t) t;
    return (raw + 3) & ~(size_t) 3;   // ik pads a KT row to 4 bytes
}

}  // namespace strata::kernels::ikq
