// tests/minimax/minimax_config_test.cpp - the MiniMax-M3 config read from a synthetic GGUF: no model, no GPU.
//
// The bug this guards: `attention.indexer.first_layer` was optional and defaulted to the dense prefix (3).  Unsloth's
// MiniMax-M3.1-Flash GGUFs omit the key but keep the indexer on all 60 layers, so layers 0-2 would have run without
// MSA.  The key is now authoritative when present (our own packs write it and must not move) and inferred from the
// indexer tensors when absent.  The same file also guards the `swiglu_alpha` / `swiglu_limit` spellings.
#include "../core/gguf_fixture.hpp"

#include "strata/minimax/config.hpp"
#include "strata/minimax/gguf.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-86s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-minimax-config-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
const uint32_t F32 = 0;

/// A complete minimax-m3 geometry: small enough for a fixture, with every required key.
std::vector<Kv> metadata(bool with_first_layer = false, int64_t first_layer = 0) {
    std::vector<Kv> kv = {
        fixture::str("general.architecture", "minimax-m3"),
        fixture::u32("minimax-m3.block_count", 4),
        fixture::u32("minimax-m3.embedding_length", 16),
        fixture::u32("minimax-m3.vocab_size", 32),
        fixture::u32("minimax-m3.context_length", 512),
        fixture::u32("minimax-m3.attention.head_count", 2),
        fixture::u32("minimax-m3.attention.head_count_kv", 1),
        fixture::u32("minimax-m3.attention.key_length", 8),
        fixture::u32("minimax-m3.attention.value_length", 8),
        fixture::u32("minimax-m3.rope.dimension_count", 8),
        fixture::f32("minimax-m3.rope.freq_base", 1.0e7f),
        fixture::f32("minimax-m3.attention.layer_norm_rms_epsilon", 1.0e-6f),
        fixture::u32("minimax-m3.feed_forward_length", 16),
        fixture::u32("minimax-m3.expert_feed_forward_length", 16),
        fixture::u32("minimax-m3.expert_count", 4),
        fixture::u32("minimax-m3.expert_used_count", 2),
        fixture::u32("minimax-m3.expert_shared_count", 1),
        fixture::u32("minimax-m3.expert_gating_func", 2),
        fixture::u32("minimax-m3.leading_dense_block_count", 3),
        fixture::u32("minimax-m3.attention.indexer.head_count", 2),
        fixture::u32("minimax-m3.attention.indexer.key_length", 8),
        fixture::u32("minimax-m3.attention.indexer.top_k", 2),
        fixture::u32("minimax-m3.attention.indexer.block_size", 8),
        fixture::u32("minimax-m3.attention.indexer.local_blocks", 1),
    };
    if (with_first_layer)
        kv.push_back(fixture::u32("minimax-m3.attention.indexer.first_layer", (uint64_t) first_layer));
    return kv;
}

/// Only the one tensor the inference looks for: `indexer.q_proj.weight` on layers [first, n_layers).
std::vector<fixture::Tensor> indexer_tensors(int64_t first, int64_t n_layers) {
    std::vector<fixture::Tensor> ts;
    uint8_t seed = 1;
    for (int64_t l = first; l < n_layers; ++l)
        ts.push_back({"blk." + std::to_string(l) + ".indexer.q_proj.weight", {8}, F32, seed++});
    return ts;
}

/// Every tensor `minimax::load_model` expects for the 4-layer metadata above (all F32 fixtures).
std::vector<fixture::Tensor> full_model_tensors(int64_t first_indexer) {
    const uint64_t ne = 16, vocab = 32, hq = 16, hkv = 8, head = 8, ff = 16, ffe = 16, shexp = 16, nxp = 4;
    std::vector<fixture::Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, std::vector<uint64_t> shape) {
        ts.push_back({name, std::move(shape), F32, seed++});
    };
    add("token_embd.weight", {ne, vocab});
    add("output_norm.weight", {ne});
    add("output.weight", {ne, vocab});
    for (int64_t l = 0; l < 4; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        add(b + "attn_norm.weight", {ne});
        add(b + "attn_q.weight", {ne, hq});
        add(b + "attn_k.weight", {ne, hkv});
        add(b + "attn_v.weight", {ne, hkv});
        add(b + "attn_output.weight", {hq, ne});
        add(b + "attn_q_norm.weight", {head});
        add(b + "attn_k_norm.weight", {head});
        add(b + "ffn_norm.weight", {ne});
        if (l < 3) {
            add(b + "ffn_gate.weight", {ne, ff});
            add(b + "ffn_up.weight", {ne, ff});
            add(b + "ffn_down.weight", {ff, ne});
        } else {
            add(b + "ffn_gate_inp.weight", {ne, nxp});
            add(b + "exp_probs_b.bias", {nxp});
            add(b + "ffn_gate_exps.weight", {ne, ffe, nxp});
            add(b + "ffn_down_exps.weight", {ffe, ne, nxp});
            add(b + "ffn_up_exps.weight", {ne, ffe, nxp});
            add(b + "ffn_gate_shexp.weight", {ne, shexp});
            add(b + "ffn_down_shexp.weight", {shexp, ne});
            add(b + "ffn_up_shexp.weight", {ne, shexp});
        }
        if (l >= first_indexer) {
            add(b + "indexer.q_proj.weight", {ne, hq});
            add(b + "indexer.k_proj.weight", {ne, head});
            add(b + "indexer.q_norm.weight", {head});
            add(b + "indexer.k_norm.weight", {head});
        }
    }
    return ts;
}

strata::minimax::Config read(const fs::path& p) {
    return strata::minimax::from_gguf(strata::GgufFile(p.string()));
}

}  // namespace

int main() {
    std::printf("minimax_config_test\n");
    TempDir d;
    {
        // Our own packs write the key.  It wins even when every layer has an indexer tensor (0) and even when the
        // tensors start later than the key says (3): the metadata is authoritative, not a hint.
        const fs::path explicit0 = d.path / "explicit0.gguf";
        fixture::write(explicit0, metadata(true, 0), indexer_tensors(0, 4));
        check(read(explicit0).idx_first_layer == 0, "explicit first_layer=0 is kept (our D3 pack)");

        const fs::path explicit3 = d.path / "explicit3.gguf";
        fixture::write(explicit3, metadata(true, 3), indexer_tensors(0, 4));
        check(read(explicit3).idx_first_layer == 3, "explicit first_layer=3 is kept even with indexer tensors on 0-2");

        const fs::path explicit2 = d.path / "explicit2.gguf";
        fixture::write(explicit2, metadata(true, 2), indexer_tensors(2, 4));
        check(read(explicit2).idx_first_layer == 2, "explicit first_layer=2 is kept");
    }
    {
        // Absent key: infer from the tensors.  Unsloth's GGUFs are the first case (all 60 layers carry the indexer).
        const fs::path all = d.path / "all.gguf";
        fixture::write(all, metadata(), indexer_tensors(0, 4));
        check(read(all).idx_first_layer == 0, "absent key + indexer tensors on all layers -> 0 (Unsloth's layout)");

        const fs::path late = d.path / "late.gguf";
        fixture::write(late, metadata(), indexer_tensors(3, 4));
        check(read(late).idx_first_layer == 3, "absent key + indexer tensors from layer 3 -> 3 (the old default, now inferred)");

        const fs::path none = d.path / "none.gguf";
        fixture::write(none, metadata(), {});
        check(read(none).idx_first_layer == 3, "absent key + no indexer tensors -> the dense prefix, not a silent 0");
    }
    {
        // A split model: shard 1 is metadata only, the tensors live in shard 2.  The model overload must see them.
        const fs::path s1 = d.path / "split-1.gguf", s2 = d.path / "split-2.gguf";
        std::vector<Kv> kv1 = metadata();
        for (const Kv& k : fixture::split_keys(0, 2, 4)) kv1.push_back(k);
        std::vector<Kv> kv2 = fixture::split_keys(1, 2, 4);
        fixture::write(s1, kv1, {});
        fixture::write(s2, kv2, indexer_tensors(0, 4));
        const strata::GgufModel m({s1.string(), s2.string()});
        check(strata::minimax::from_gguf(m.meta(), m).idx_first_layer == 0,
              "split model: the indexer tensors in shard 2 make first_layer 0");
        // Documents why the model overload exists: the metadata shard alone has nothing to look at.
        check(strata::minimax::from_gguf(m.meta()).idx_first_layer == 3,
              "the metadata-only overload cannot see shard 2 and keeps the dense-prefix fallback");
    }
    {
        // End to end through the engine's own loader: a file with no first_layer key and the indexer on layer 0 must
        // map blk.0's indexer tensors, not treat layer 0 as a dense/no-MSA prefix.
        const fs::path full = d.path / "full.gguf";
        fixture::write(full, metadata(), full_model_tensors(0));
        const strata::minimax::Model m = strata::minimax::load_model(full.string(), false);
        check(m.config.idx_first_layer == 0 && m.config.has_indexer(0),
              "load_model infers first_layer 0 and expects MSA on layer 0");
        bool saw_layer0_indexer = false;
        for (const strata::minimax::TensorRecord& t : m.tensors)
            if (t.name == "blk.0.indexer.q_proj.weight") saw_layer0_indexer = true;
        check(saw_layer0_indexer, "the loader maps blk.0.indexer.q_proj.weight");
    }
    {
        // llama.cpp's converter spells the swiglu keys with underscores; our packs use dots.
        std::vector<Kv> under = metadata();
        under.push_back(fixture::f32("minimax-m3.swiglu_alpha", 1.5f));
        under.push_back(fixture::f32("minimax-m3.swiglu_limit", 5.0f));
        const fs::path up = d.path / "swiglu_under.gguf";
        fixture::write(up, under, indexer_tensors(0, 4));
        const strata::minimax::Config cu = read(up);
        check(cu.swiglu_alpha == 1.5 && cu.swiglu_limit == 5.0, "swiglu_alpha / swiglu_limit are read");

        std::vector<Kv> both = under;
        both.push_back(fixture::f32("minimax-m3.swiglu.alpha", 1.6f));
        both.push_back(fixture::f32("minimax-m3.swiglu.limit", 6.0f));
        const fs::path bp = d.path / "swiglu_both.gguf";
        fixture::write(bp, both, indexer_tensors(0, 4));
        const strata::minimax::Config cb = read(bp);
        check(std::fabs(cb.swiglu_alpha - 1.6) < 1e-6 && std::fabs(cb.swiglu_limit - 6.0) < 1e-6,
              "the dotted spelling wins when both are present");

        const fs::path dp = d.path / "swiglu_default.gguf";
        fixture::write(dp, metadata(), indexer_tensors(0, 4));
        const strata::minimax::Config cd = read(dp);
        check(cd.swiglu_alpha == 1.702 && cd.swiglu_limit == 7.0, "neither spelling present -> the model's defaults");
    }
    {
        // MiniMax-M3.1-Flash's exact fingerprint with no first_layer key and the indexer on all 60 layers: this is
        // what an Unsloth file looks like, and it must pass check_m3_1 with first_layer 0.
        std::vector<Kv> m31 = {
            fixture::str("general.architecture", "minimax-m3"),
            fixture::u32("minimax-m3.block_count", 60),
            fixture::u32("minimax-m3.embedding_length", 6144),
            fixture::u32("minimax-m3.vocab_size", 200064),
            fixture::u32("minimax-m3.attention.head_count", 64),
            fixture::u32("minimax-m3.attention.head_count_kv", 4),
            fixture::u32("minimax-m3.attention.key_length", 128),
            fixture::u32("minimax-m3.attention.value_length", 128),
            fixture::u32("minimax-m3.rope.dimension_count", 64),
            fixture::f32("minimax-m3.rope.freq_base", 1.0e7f),
            fixture::f32("minimax-m3.attention.layer_norm_rms_epsilon", 1.0e-6f),
            fixture::u32("minimax-m3.feed_forward_length", 12288),
            fixture::u32("minimax-m3.expert_feed_forward_length", 3072),
            fixture::u32("minimax-m3.expert_count", 128),
            fixture::u32("minimax-m3.expert_used_count", 4),
            fixture::u32("minimax-m3.expert_shared_count", 1),
            fixture::u32("minimax-m3.expert_gating_func", 2),
            fixture::u32("minimax-m3.leading_dense_block_count", 3),
            fixture::f32("minimax-m3.expert_weights_scale", 2.0f),
            fixture::u32("minimax-m3.attention.indexer.head_count", 4),
            fixture::u32("minimax-m3.attention.indexer.key_length", 128),
            fixture::u32("minimax-m3.attention.indexer.top_k", 16),
            fixture::u32("minimax-m3.attention.indexer.block_size", 128),
            fixture::u32("minimax-m3.attention.indexer.local_blocks", 1),
        };
        Kv norm = fixture::make_kv("minimax-m3.expert_weights_norm", 7);   // BOOL
        norm.u = 1;
        m31.push_back(norm);
        const fs::path p = d.path / "m31.gguf";
        fixture::write(p, m31, indexer_tensors(0, 60));
        const strata::minimax::Config c = read(p);
        check(c.idx_first_layer == 0, "M3.1 with no key and the indexer on all 60 layers -> first_layer 0");
        check(strata::minimax::check_structure(c).empty(), "the inferred config passes check_structure");
        check(strata::minimax::check_m3_1(c).empty(), "and passes the MiniMax-M3.1-Flash fingerprint");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    std::fflush(stdout);
    return g_fail ? 1 : 0;
}
