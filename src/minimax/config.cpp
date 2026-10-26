// src/minimax/config.cpp - read and validate MiniMax-M3.1-Flash's `minimax-m3.*` GGUF metadata.
#include "strata/minimax/config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "strata/artifact/gguf_reader.hpp"

namespace strata::minimax {
namespace {

const char* const kPrefix = "minimax-m3.";

std::string k(const char* suffix) { return std::string(kPrefix) + suffix; }

std::string type_word(MetaType t) {
    switch (t) {
    case MetaType::U8: return "uint8";
    case MetaType::I8: return "int8";
    case MetaType::U16: return "uint16";
    case MetaType::I16: return "int16";
    case MetaType::U32: return "uint32";
    case MetaType::I32: return "int32";
    case MetaType::F32: return "float32";
    case MetaType::BOOL: return "bool";
    case MetaType::STRING: return "string";
    case MetaType::ARRAY: return "array";
    case MetaType::U64: return "uint64";
    case MetaType::I64: return "int64";
    case MetaType::F64: return "float64";
    }
    return "?";
}

int64_t get_int(const GgufFile& g, const char* suffix, bool required, int64_t fallback = 0) {
    const MetaValue* v = g.get(k(suffix));
    if (!v) {
        if (required) throw std::runtime_error("missing " + k(suffix));
        return fallback;
    }
    if (!v->is_num()) throw std::runtime_error(k(suffix) + " is a " + type_word(v->type) + ", not a number");
    if (v->type == MetaType::F32 || v->type == MetaType::F64) {
        const double d = v->num();
        if (d != std::floor(d)) throw std::runtime_error(k(suffix) + " is not a whole number");
        return (int64_t) d;
    }
    if (v->type == MetaType::BOOL) throw std::runtime_error(k(suffix) + " is a bool, not an integer");
    return (int64_t) v->u;
}

double get_real(const GgufFile& g, const char* suffix, bool required, double fallback = 0.0) {
    const MetaValue* v = g.get(k(suffix));
    if (!v) {
        if (required) throw std::runtime_error("missing " + k(suffix));
        return fallback;
    }
    if (!v->is_num()) throw std::runtime_error(k(suffix) + " is a " + type_word(v->type) + ", not a number");
    return v->num();
}

bool get_bool(const GgufFile& g, const char* suffix, bool required, bool fallback = false) {
    const MetaValue* v = g.get(k(suffix));
    if (!v) {
        if (required) throw std::runtime_error("missing " + k(suffix));
        return fallback;
    }
    if (v->type != MetaType::BOOL) throw std::runtime_error(k(suffix) + " is a " + type_word(v->type) + ", not a bool");
    return v->u != 0;
}

// llama.cpp writes expert_feed_forward_length either as one number or as one number per layer (get_key_or_arr).
int64_t get_ff_exp(const GgufFile& g) {
    const MetaValue* v = g.get(k("expert_feed_forward_length"));
    if (!v) throw std::runtime_error("missing " + k("expert_feed_forward_length"));
    if (v->type != MetaType::ARRAY) return get_int(g, "expert_feed_forward_length", true);
    if (v->count == 0 || v->items.empty()) throw std::runtime_error(k("expert_feed_forward_length") + " is an empty array");
    const int64_t first = (int64_t) v->items[0].u;
    for (size_t i = 1; i < v->items.size(); ++i) {
        const int64_t x = (int64_t) v->items[i].u;
        if (x != first)
            throw std::runtime_error(k("expert_feed_forward_length") + " varies by layer (" + std::to_string(first) +
                                     " on layer 0, " + std::to_string(x) + " on layer " + std::to_string(i) +
                                     "): this engine uses one routed-expert width");
    }
    return first;
}

int64_t get_vocab(const GgufFile& g) {
    // The tokenizer's token list is authoritative (it is what the ids index); the model's own width is a fallback.
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens"))
        if (t->type == MetaType::ARRAY && t->count > 0) return (int64_t) t->count;
    return get_int(g, "vocab_size", false, 0);
}

// llama.cpp's converter spells these with an underscore, our packs with a dot.  The dotted key wins when both exist,
// so a file carrying both is read the way it always was.
double get_real_alias(const GgufFile& g, const char* primary, const char* alias, double fallback) {
    if (g.get(k(primary))) return get_real(g, primary, false, fallback);
    return get_real(g, alias, false, fallback);
}

// The metadata key is authoritative.  When it is absent (Unsloth's MiniMax GGUFs omit it), the first layer that
// actually has indexer tensors is the first MSA layer.  A file with no indexer tensors at all keeps the old
// dense-prefix fallback, and the layout check then reports the missing tensors instead of silently running without
// MSA.  `has_indexer(layer)` looks across the whole model, so a split file whose metadata shard has no tensors is
// handled too.
template <class HasIndexer>
Config read_config(const GgufFile& g, const HasIndexer& has_indexer) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) throw std::runtime_error("missing general.architecture");
    if (arch->type != MetaType::STRING) throw std::runtime_error("general.architecture is not a string");
    if (arch->s != "minimax-m3")
        throw std::runtime_error("architecture is '" + arch->s + "', this engine requires 'minimax-m3'");

    Config c;
    c.n_layers = get_int(g, "block_count", true);
    c.n_embd = get_int(g, "embedding_length", true);
    c.n_vocab = get_vocab(g);
    c.context = get_int(g, "context_length", false, 1048576);
    c.n_head = get_int(g, "attention.head_count", true);
    c.n_head_kv = get_int(g, "attention.head_count_kv", true);
    c.head_dim = get_int(g, "attention.key_length", true);
    const int64_t value_dim = get_int(g, "attention.value_length", false, c.head_dim);
    if (value_dim != c.head_dim)
        throw std::runtime_error(k("attention.value_length") + " = " + std::to_string(value_dim) + " and " +
                                 k("attention.key_length") + " = " + std::to_string(c.head_dim) +
                                 ": this engine uses one head width for both");
    c.n_rot = get_int(g, "rope.dimension_count", false, c.head_dim);
    c.rope_theta = get_real(g, "rope.freq_base", false, 1.0e7);
    c.rms_eps = get_real(g, "attention.layer_norm_rms_epsilon", true);
    c.n_ff_dense = get_int(g, "feed_forward_length", true);
    c.n_ff_exp = get_ff_exp(g);
    c.n_expert = get_int(g, "expert_count", true);
    c.n_expert_used = get_int(g, "expert_used_count", true);
    c.n_shared_experts = get_int(g, "expert_shared_count", true);
    c.leading_dense = get_int(g, "leading_dense_block_count", false, 0);
    c.routed_scale = get_real(g, "expert_weights_scale", false, 1.0);
    c.expert_weights_norm = get_bool(g, "expert_weights_norm", false, false);
    c.gating_func = get_int(g, "expert_gating_func", true);
    c.swiglu_alpha = get_real_alias(g, "swiglu.alpha", "swiglu_alpha", 1.702);
    c.swiglu_limit = get_real_alias(g, "swiglu.limit", "swiglu_limit", 7.0);

    c.idx_heads = get_int(g, "attention.indexer.head_count", true);
    c.idx_dim = get_int(g, "attention.indexer.key_length", true);
    c.idx_top_k = get_int(g, "attention.indexer.top_k", true);
    c.idx_block = get_int(g, "attention.indexer.block_size", true);
    c.idx_local = get_int(g, "attention.indexer.local_blocks", true);
    if (g.get(k("attention.indexer.first_layer"))) {
        c.idx_first_layer = get_int(g, "attention.indexer.first_layer", true);
    } else {
        int64_t inferred = c.leading_dense;
        for (int64_t l = 0; l < c.n_layers; ++l) {
            if (has_indexer(l)) {
                inferred = l;
                break;
            }
        }
        c.idx_first_layer = inferred;
    }
    return c;
}

}  // namespace

Config from_gguf(const GgufFile& g) {
    return read_config(g, [&g](int64_t layer) {
        return g.find("blk." + std::to_string(layer) + ".indexer.q_proj.weight") != nullptr;
    });
}

Config from_gguf(const GgufFile& g, const GgufModel& model) {
    return read_config(g, [&model](int64_t layer) {
        return model.find("blk." + std::to_string(layer) + ".indexer.q_proj.weight") != nullptr;
    });
}

std::string check_structure(const Config& c) {
    auto bad = [](const std::string& what) { return std::string("invalid geometry: ") + what; };
    if (c.n_layers < 1) return bad("minimax-m3.block_count must be at least 1");
    if (c.n_embd < 1) return bad("minimax-m3.embedding_length must be positive");
    if (c.n_vocab < 1) return bad("the tokenizer/vocabulary is empty");
    if (c.context < 1) return bad("minimax-m3.context_length must be positive");
    if (c.n_head < 1 || c.n_head_kv < 1) return bad("attention.head_count and head_count_kv must be positive");
    if (c.n_head % c.n_head_kv != 0)
        return bad("attention.head_count (" + std::to_string(c.n_head) + ") is not a multiple of head_count_kv (" +
                   std::to_string(c.n_head_kv) + ")");
    if (c.head_dim < 1) return bad("attention.key_length must be positive");
    if (c.n_rot < 0 || c.n_rot > c.head_dim) return bad("rope.dimension_count must be in 0..head_dim");
    if (c.n_rot % 2 != 0) return bad("rope.dimension_count must be even (rotary pairs)");
    if (!(c.rope_theta > 0.0)) return bad("rope.freq_base must be positive");
    if (!(c.rms_eps > 0.0)) return bad("attention.layer_norm_rms_epsilon must be positive");
    if (c.leading_dense < 0 || c.leading_dense > c.n_layers)
        return bad("leading_dense_block_count must be in 0..block_count");
    if (c.n_ff_dense < 1 && c.leading_dense > 0) return bad("feed_forward_length must be positive (leading dense MLP)");
    if (c.n_ff_exp < 1) return bad("expert_feed_forward_length must be positive");
    if (c.n_expert < 1) return bad("expert_count must be positive");
    if (c.n_expert_used < 1 || c.n_expert_used > c.n_expert)
        return bad("expert_used_count must be in 1..expert_count");
    if (c.n_shared_experts < 0) return bad("expert_shared_count must not be negative");
    if (!(c.routed_scale > 0.0)) return bad("expert_weights_scale must be positive");
    if (c.gating_func != 2)
        return bad("expert_gating_func = " + std::to_string(c.gating_func) +
                   "; only SIGMOID (2) is implemented for MiniMax-M3");
    if (c.idx_heads < 1 || c.idx_dim < 1) return bad("the MSA indexer needs positive head_count/key_length");
    if (c.idx_top_k < 1) return bad("attention.indexer.top_k must be positive");
    if (c.idx_block < 1) return bad("attention.indexer.block_size must be positive");
    if (c.idx_local < 0) return bad("attention.indexer.local_blocks must not be negative");
    if (c.idx_first_layer < 0 || c.idx_first_layer > c.n_layers)
        return bad("attention.indexer.first_layer must be in 0..block_count");
    if (!(c.swiglu_alpha > 0.0)) return bad("swiglu.alpha must be positive");
    if (!(c.swiglu_limit > 0.0)) return bad("swiglu.limit must be positive");
    return {};
}

std::vector<std::string> check_m3_1(const Config& c) {
    std::vector<std::string> out;
    auto expect = [&](const char* key, int64_t got, int64_t want) {
        if (got != want)
            out.push_back(std::string("minimax-m3.") + key + " = " + std::to_string(got) + ", expected " +
                          std::to_string(want) + " for MiniMax-M3.1-Flash");
    };
    auto expect_real = [&](const char* key, double got, double want) {
        // GGUF writes f32: 1e-6 is not exact in binary, so compare with a tolerance rather than refusing a real file.
        const double tol = 1e-9 * (std::max)(1.0, std::fabs(want));
        if (std::fabs(got - want) > tol)
            out.push_back(std::string("minimax-m3.") + key + " = " + std::to_string(got) + ", expected " +
                          std::to_string(want) + " for MiniMax-M3.1-Flash");
    };
    expect("block_count", c.n_layers, 60);
    expect("embedding_length", c.n_embd, 6144);
    expect("vocab_size", c.n_vocab, 200064);
    expect("attention.head_count", c.n_head, 64);
    expect("attention.head_count_kv", c.n_head_kv, 4);
    expect("attention.key_length", c.head_dim, 128);
    expect("rope.dimension_count", c.n_rot, 64);
    expect_real("rope.freq_base", c.rope_theta, 1.0e7);
    expect_real("attention.layer_norm_rms_epsilon", c.rms_eps, 1.0e-6);
    expect("feed_forward_length", c.n_ff_dense, 12288);
    expect("expert_feed_forward_length", c.n_ff_exp, 3072);
    expect("expert_count", c.n_expert, 128);
    expect("expert_used_count", c.n_expert_used, 4);
    expect("expert_shared_count", c.n_shared_experts, 1);
    expect("leading_dense_block_count", c.leading_dense, 3);
    expect_real("expert_weights_scale", c.routed_scale, 2.0);
    expect("attention.indexer.head_count", c.idx_heads, 4);
    expect("attention.indexer.key_length", c.idx_dim, 128);
    expect("attention.indexer.top_k", c.idx_top_k, 16);
    expect("attention.indexer.block_size", c.idx_block, 128);
    expect("attention.indexer.local_blocks", c.idx_local, 1);
    expect("attention.indexer.first_layer", c.idx_first_layer, 0);
    if (!c.expert_weights_norm)
        out.push_back("minimax-m3.expert_weights_norm = false, expected true for MiniMax-M3.1-Flash");
    return out;
}

std::string summary(const Config& c) {
    std::string s = "minimax-m3: " + std::to_string(c.n_layers) + " layers, embd " + std::to_string(c.n_embd) +
                    ", " + std::to_string(c.n_head) + "/" + std::to_string(c.n_head_kv) + " heads x " +
                    std::to_string(c.head_dim) + " (rope " + std::to_string(c.n_rot) + "), vocab " +
                    std::to_string(c.n_vocab) + ", dense " + std::to_string(c.n_ff_dense) + " on " +
                    std::to_string(c.leading_dense) + " layers, " + std::to_string(c.n_expert) + " experts top-" +
                    std::to_string(c.n_expert_used) + " x " + std::to_string(c.n_ff_exp) + " + " +
                    std::to_string(c.n_shared_experts) + " shared, indexer " + std::to_string(c.idx_heads) + "x" +
                    std::to_string(c.idx_dim) + " top-" + std::to_string(c.idx_top_k) + " blocks of " +
                    std::to_string(c.idx_block) + " from layer " + std::to_string(c.idx_first_layer);
    return s;
}

}  // namespace strata::minimax
