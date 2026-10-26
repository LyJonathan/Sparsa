#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace mmq {

struct Options {
    std::filesystem::path model;
    std::filesystem::path recipe;
    std::filesystem::path out;
    std::filesystem::path imatrix;
    std::filesystem::path write_imatrix;
    std::filesystem::path expert_profile;
    std::string dump_name;
    std::filesystem::path dump_out;
    std::filesystem::path mmproj_out;
    std::filesystem::path llama_cpp;
    std::filesystem::path exe_dir;
    int max_layers = -1;
    std::vector<int> layers;
    // Range form --layers A-B keeps blk.{source} names and writes only that
    // range. Comma form remaps onto blk.0.. for a loadable smoke model.
    bool keep_indices = false;
    // Output-index filter on a remapped plan (comma/--max-layers). -1 = all.
    int emit_lo = -1;
    int emit_hi = -1;
    // llama-gguf-split part. part is 0-based; count >= 2 enables split output.
    // tensors < 0 means "count a full plan". Part 0 carries KV metadata.
    int split_part = -1;
    int split_count = 0;
    int split_tensors = -1;
    bool include_globals = false;
    bool include_globals_set = false;
    bool no_globals = false;
    int threads = 0;
    std::string dense_type;
    std::string hot_mode = "record";
    bool hot_mode_set = false;
    bool placeholder = false;
    bool dry_run = false;
    bool bench = false;
    int bench_experts = 8;
    bool nvfp4_check = false;
    float placeholder_value = 0.02f;
    // Imatrix tail blending. Experts with count < imatrix_min_count (0 = every
    // expert) get w = (c*w_e + K*w_layer)/(c + K) with K = imatrix_blend.
    // imatrix_blend <= 0 disables blending.
    int imatrix_min_count = 0;
    float imatrix_blend = 0.f;
    // Experiment mode: one MoE layer, all experts, measured error per matrix.
    int exp_layer = -1;
    std::string exp_type;
    std::filesystem::path exp_out;
    std::filesystem::path exp_x;
    std::filesystem::path exp_dumps;
    int exp_experts = 0;  // 0 = all
    // Rounding method of the experiment and of a layer pack:
    //   old   - llama.cpp's imatrix rounding, the tier's own imatrix (the baseline)
    //   diag  - llama.cpp's imatrix rounding with a per-expert diagonal built from the dumps
    //   gptq  - GPTQ-style error feedback (gptq.hpp)
    std::string exp_method = "old";
    std::filesystem::path exp_pack;   // write this layer's expert tensors (GGUF) for --expert-source
    double gptq_prior = 16384.0;      // tokens' worth of the layer-wide Hessian mixed into each expert's
    double gptq_damp = 0.1;           // diagonal damping, fraction of mean(diag)
    double exp_holdout = 0.2;         // share of each expert's rows the error is measured on (never in H)
    int gptq_selftest = 0;            // > 0: check chol/inverse against a naive reference at this size, exit
    std::vector<std::pair<std::string, std::string>> overrides;
};

int run(const Options & opt);

}  // namespace mmq
