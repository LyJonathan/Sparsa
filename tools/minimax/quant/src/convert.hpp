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
    // --tq-eval: the trellis format comparison on one real MoE layer
    int tq_eval_layer = -1;
    int tq_eval_experts = 8;
    int tq_eval_expert = 0;
    int tq_eval_rows = 0;   // 0 = every row of each matrix, else the first N rows (fast TQK6/TQK7 cells)
    std::filesystem::path tq_eval_dumps;
    std::vector<std::pair<std::string, std::string>> overrides;
};

int run(const Options & opt);

}  // namespace mmq
