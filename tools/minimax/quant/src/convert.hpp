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
    std::vector<std::pair<std::string, std::string>> overrides;
};

int run(const Options & opt);

}  // namespace mmq
