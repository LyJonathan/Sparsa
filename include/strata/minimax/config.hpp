// include/strata/minimax/config.hpp - MiniMax-M3.1-Flash's geometry, read from a GGUF's `minimax-m3.*` metadata.
//
// Nothing here is a kernel constant: every dimension comes from the file (the key names are llama.cpp's, so a GGUF
// from conversion/minimax.py and a Strata pack describe the same model the same way).  `check_structure` refuses a
// file whose dimensions cannot describe this architecture at all; `check_m3_1` adds the exact fingerprint of
// MiniMax-M3.1-Flash, so a different MiniMax model is refused with the one key that differs rather than mis-run.
//
// The `--minimax-inspect` mode uses the structural check only (it must be able to describe a file it does not run).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::minimax {

/// What this engine knows how to compute.  Read from the GGUF; no field is a compiled-in shape.
struct Config {
    // ---- text tower
    int64_t n_layers = 0;        ///< minimax-m3.block_count
    int64_t n_embd = 0;          ///< minimax-m3.embedding_length
    int64_t n_vocab = 0;         ///< tokenizer.ggml.tokens count, else minimax-m3.vocab_size
    int64_t context = 0;         ///< minimax-m3.context_length
    int64_t n_head = 0;          ///< minimax-m3.attention.head_count
    int64_t n_head_kv = 0;       ///< minimax-m3.attention.head_count_kv
    int64_t head_dim = 0;        ///< minimax-m3.attention.key_length (= value_length)
    int64_t n_rot = 0;           ///< minimax-m3.rope.dimension_count (partial RoPE: the first n_rot dims)
    double rope_theta = 0.0;     ///< minimax-m3.rope.freq_base
    double rms_eps = 0.0;        ///< minimax-m3.attention.layer_norm_rms_epsilon
    int64_t n_ff_dense = 0;      ///< minimax-m3.feed_forward_length (layers 0..leading_dense-1)
    int64_t n_ff_exp = 0;        ///< minimax-m3.expert_feed_forward_length (per routed expert)
    int64_t n_expert = 0;        ///< minimax-m3.expert_count
    int64_t n_expert_used = 0;   ///< minimax-m3.expert_used_count
    int64_t n_shared_experts = 0;///< minimax-m3.expert_shared_count (usually 1)
    int64_t leading_dense = 0;   ///< minimax-m3.leading_dense_block_count (3 for M3.1)
    double routed_scale = 0.0;   ///< minimax-m3.expert_weights_scale (2.0 for M3.1)
    bool expert_weights_norm = false; ///< minimax-m3.expert_weights_norm (renormalize top-k)
    int64_t gating_func = 0;     ///< minimax-m3.expert_gating_func (2 = SIGMOID; the only one implemented)
    // swigluoai's numbers are not in the llama.cpp metadata today; the released model uses these.
    double swiglu_alpha = 1.702;
    double swiglu_limit = 7.0;

    // ---- MSA indexer (all 60 layers for M3.1; `indexer_first_layer` selects the first sparse layer)
    int64_t idx_heads = 0;       ///< minimax-m3.attention.indexer.head_count
    int64_t idx_dim = 0;         ///< minimax-m3.attention.indexer.key_length
    int64_t idx_top_k = 0;       ///< minimax-m3.attention.indexer.top_k (blocks)
    int64_t idx_block = 0;       ///< minimax-m3.attention.indexer.block_size
    int64_t idx_local = 0;       ///< minimax-m3.attention.indexer.local_blocks
    int64_t idx_first_layer = 0; ///< minimax-m3.attention.indexer.first_layer

    /// Number of routed experts on layer `i` (0 on a leading dense layer).
    int64_t experts_on(int64_t layer) const { return layer < leading_dense ? 0 : n_expert; }
    /// Whether layer `i` carries MSA indexer tensors.
    bool has_indexer(int64_t layer) const { return layer >= idx_first_layer; }
    /// Whether layer `i` has a routed/shared MoE block.
    bool is_moe(int64_t layer) const { return layer >= leading_dense; }
};

/// Read a config from the metadata shard of a GGUF.  Throws std::runtime_error with the first missing/malformed key.
Config from_gguf(const GgufFile& g);

/// Cheap structural validation (positivity, divisibility, supported activation/gating).  "" when fine.
std::string check_structure(const Config& c);

/// Differences from the MiniMax-M3.1-Flash fingerprint, one clear line each.  Empty when this is M3.1.
std::vector<std::string> check_m3_1(const Config& c);

/// One-line human summary: "minimax-m3: 60 layers, embd 6144, 64/4 heads x 128, ...".
std::string summary(const Config& c);

}  // namespace strata::minimax
