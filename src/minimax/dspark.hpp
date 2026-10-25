// src/minimax/dspark.hpp - MiniMax-M3.1-Flash's own DSpark draft head, driven from the engine.
//
// DSpark is a block drafter (config `dspark_*` in models/MiniMax-M3.1-Flash/dspark/config.json):
//
//   context   the target's residual stream at five of its layers (dspark_target_layer_ids: 3, 17, 31, 45, 59)
//             is concatenated (5 x 6144) and projected by `fc` to the drafter's width; that row is the ONLY thing
//             the drafter knows about the past, and each of its 5 layers turns it into one K/V row of that layer's
//             ring.  There is no autoregressive KV of drafted tokens: the ring holds committed positions only, so a
//             rejected draft leaves nothing to roll back.
//   block     one forward pass of `block_size + 1` rows - the last committed token then `block_size` mask tokens
//             (dspark_mask_token_id) - with full (non-causal) attention inside the block over the ring plus the
//             block's own keys.  Row j is at position p0 + j and its hidden state predicts the token at p0 + j.
//   head      the target's own lm_head (the checkpoint carries no head of its own) plus a rank-256 Markov bias
//             logits[v] += <markov_w1[v], markov_w2[prev]> that couples the block's rows into a chain, and a
//             sigmoid confidence head per row.
//
// The engine only ever uses these tokens as a proposal: the target's own forward pass verifies them (greedy argmax,
// longest matching prefix), so any difference here changes the speed of decoding, never the tokens.
//
// The pack comes from tools/minimax/dspark_pack.py (Q8_0 payload, run through the device MMVQ path; the 1-D
// weights are F16 and read on the host).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::minimax {

struct DsparkGeometry {
    int64_t n_layers = 5;
    int64_t n_embd = 6144;
    int64_t n_head = 64;
    int64_t n_head_kv = 4;
    int64_t head_dim = 128;
    int64_t n_rot = 64;
    int64_t n_ff = 12288;
    int64_t block = 7;        ///< dspark.block_size: mask rows per round (= drafts per round)
    int64_t rank = 256;       ///< dspark.markov_rank
    int64_t mask_id = 200058; ///< dspark.mask_token_id
    int64_t vocab = 200064;
    int64_t window = 1024;    ///< ring rows kept per layer (the checkpoint trains with 4096)
    double theta = 1e7;
    double eps = 1e-6;
    double alpha = 1.702;
    double limit = 7.0;
    bool norm_add1 = true;    ///< use_gemma_norm: the norm weights are applied as (1 + w)
    std::vector<int64_t> taps;///< dspark_target_layer_ids
    /// Block rows per round: the anchor plus one per draft.
    int64_t rows() const { return block + 1; }
    /// Values of one tap row: every tapped layer's residual, concatenated.
    int64_t tap_width() const { return (int64_t)taps.size() * n_embd; }
};

class DsparkDrafter {
public:
    DsparkDrafter();
    ~DsparkDrafter();
    DsparkDrafter(const DsparkDrafter&) = delete;
    DsparkDrafter& operator=(const DsparkDrafter&) = delete;

    /// Read the pack (tools/minimax/dspark_pack.py) onto `device` and allocate. `window` clamps the geometry's
    /// ring; 0 keeps the pack's own sliding window.
    bool load(const std::string& pack_path, int device, int64_t window, std::string& err);
    /// The target's head, as device GGUF rows (Q6_K etc., any type the MMVQ path takes) or `dev == nullptr` to
    /// upload `host`. `rows x cols` with cols = the drafter's width.
    bool bind_head(const void* dev, const void* host, size_t host_bytes, int ggml_type, int64_t rows, int64_t cols,
                   std::string& err);

    /// A new request: the ring is empty.
    void reset();
    /// Committed rows: `taps` is [n][tap_width()] fp32 for positions pos0 .. pos0+n-1 (the target's own taps).
    /// Every position must be seeded exactly once, in order; the ring keeps the last `window` of them.
    bool seed(const float* taps, int64_t n, int64_t pos0, std::string& err);
    /// One draft round for the block anchored at the last committed token `anchor` (position `pos0`, which must
    /// already be seeded): `ids` holds the block's rows (the anchor, then mask ids) as f32 embeddings
    /// [rows()][n_embd].  Fills `drafts[block]` (the tokens of positions pos0+1 .. pos0+block) and `conf[block]`.
    bool draft(const float* ids, int32_t anchor, int64_t pos0, int32_t* drafts, float* conf, std::string& err);

    const DsparkGeometry& geom() const { return geo_; }
    int64_t window() const { return geo_.window; }
    int64_t seeded() const { return seeded_; }
    int64_t vram_bytes() const { return vram_; }

    // stage timings (ms), cumulative
    double ms_seed = 0, ms_block = 0, ms_head = 0;
    int64_t rounds = 0, seed_rows = 0;

private:
    struct Impl;
    Impl* impl_ = nullptr;
    DsparkGeometry geo_;
    int64_t seeded_ = 0;
    int64_t vram_ = 0;
};

}  // namespace strata::minimax
