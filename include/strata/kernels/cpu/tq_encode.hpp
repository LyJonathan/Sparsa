// include/strata/kernels/cpu/tq_encode.hpp - the ENCODER of the trellis expert formats (TQ2_T / TQK6 / TQK7,
// ggml types 144-146), the counterpart of tq_cpu.hpp.  A Viterbi search over the format's own trellis, so the
// blocks it writes are decoded by the shipped run-time kernels with no format change.
//
// THE FORMAT (third_party/ggml/ggml-common-tq.h is the definition; tq_states.hpp and tq_cpu.hpp are the decode):
// a 128-weight block is 32 trellis steps of four weights.  Step t has a 15-bit state; the state's golden-ratio
// hash picks two points (four values) of a 2048-point fp16 codebook, and the four weights are d * those values.
// TQ2_T writes one byte per step; TQK6/TQK7 bit-pack K = 6 / 7 new bits per step into a circular stream.
//
// THE SEARCH: the step cost of a block depends on a window of the last ceil(16 / K) steps, so the trellis is a
// chain over one step for TQ2_T (the 16-bit window is exactly two adjacent bytes) and over two steps for
// TQK6/TQK7 (a 16-bit window spans three chunks).  Tail-biting Viterbi: two laps around the 32-step circle with
// uniform start metrics, traceback from the best final state, then a least-squares refit of d and a second lap at
// the refitted scale.  Cost is sum_i w_i (x_i - d c_i)^2, w = the imatrix row when the caller passes one.
//
// WHY w IS APPLIED AT THE STEP LEVEL, NOT PER COORDINATE: a Hadamard-folded weight is stored in a rotated basis,
// and after an orthonormal block rotation the diagonal of the (rotated) Hessian is the block mean of the original
// diagonal.  So for a folded tensor the honest per-coordinate weight is the block mean, which is a constant and
// changes nothing; see docs/MINIMAX_TRELLIS.md.  The caller passes a per-coordinate w and gets a per-coordinate
// cost either way; for folded tensors pass `tq_block_importance`'s output, which is what the error metric wants.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::kernels::cpu {

/// Type ids of the trellis formats (the agention llama.cpp fork's GGML_TYPE_* values).
constexpr int kTq2T = 144;
constexpr int kTqk6 = 145;
constexpr int kTqk7 = 146;

/// True for 144/145/146.
bool tq_encode_supported(int ggml_type) noexcept;
/// "TQ2_T" / "TQK6" / "TQK7" (nullptr for other types).
const char* tq_encode_type_name(int ggml_type) noexcept;
/// New bits per trellis step: 8 / 6 / 7 (0 for other types).
int tq_encode_bits(int ggml_type) noexcept;
/// Bytes per 128-weight block: 34 / 26 / 30 (0 for other types).
size_t tq_encode_block_bytes(int ggml_type) noexcept;
/// Encoded bytes of `n` weights (n must be a multiple of 128).
size_t tq_encode_row_bytes(int ggml_type, int64_t n);

struct TqEncodeParams {
    /// Viterbi laps after the first (the first is always run; 1 = rms scale only, 2 = rms + least-squares refit).
    int passes = 2;
    /// Target-scale candidates tried per lap (1 = the least-squares scale only, more = wider search).
    int candidates = 1;
    /// Read `w` as per-weight importance.  With weights the scale refit is weighted too.
    bool weighted = false;
    /// Extra candidates for blocks whose imatrix-weighted error is in the worst `refine_frac` of the tensor.
    int refine_candidates = 0;
    float refine_frac = 0.0f;
};

/// Encodes one 128-weight block.  `w` may be nullptr; it is only read when params.weighted.
void tq_encode_block(int ggml_type, const float* x, const float* w, const TqEncodeParams& params, uint8_t* out);

/// Encodes n weights (n % 128 == 0) into 128-weight blocks.
void tq_encode_row(int ggml_type, const float* x, const float* w, int64_t n, const TqEncodeParams& params,
                   uint8_t* out);

/// Decodes n weights.  This is the format's definition (the same arithmetic as dequantize_row_tq2_t / _tqk of the
/// agention fork and as every Strata trellis kernel): the encoder's reconstruction is exactly this function.
void tq_decode_row(int ggml_type, const uint8_t* in, int64_t n, float* y);

/// The 2048-point codebook as floats: point p is (tq_codebook()[2p], tq_codebook()[2p+1]).
const float* tq_codebook();

/// The block mean of `w` over each 128-wide block: the per-coordinate importance a Hadamard-folded tensor has in
/// the rotated basis (and the weight of that block's squared error in the original basis).
void tq_block_importance(const float* w, int64_t n, float* out);

/// In place: x[i] *= signs[i], then each 128-wide block gets the normalized Sylvester-Walsh-Hadamard transform in
/// the butterfly order of hadamard.cpp, i.e. W' = W diag(signs) H_128^T for a row of x.
void tq_hadamard_rotate(float* x, int64_t n, int block, const float* signs);

/// A deterministic +/-1 sign vector of `width` values (xorshift32 of `seed`): the same call on every machine.
std::vector<float> tq_hadamard_signs(int width, uint32_t seed);

}  // namespace strata::kernels::cpu
