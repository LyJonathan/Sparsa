#include "convert.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

void usage() {
    std::fputs(
        "strata-mm-quant --model DIR --recipe NAME|FILE [options]\n"
        "\n"
        "Stream MiniMax-M3.1 safetensors into a minimax-m3 GGUF.\n"
        "Dequantizes NVFP4 / MXFP8 / BF16 in memory and quantizes with ggml.\n"
        "Does not write a full-model BF16 or F32 intermediate.\n"
        "\n"
        "  --model DIR                 checkpoint directory (config.json + shards)\n"
        "  --recipe NAME|FILE          recipes/MM-IQ2_XS.json or a path\n"
        "  --out FILE                  GGUF to write\n"
        "  --imatrix FILE              llama.cpp GGUF imatrix\n"
        "  --imatrix-min-count N       blend experts with count < N (0 = every expert)\n"
        "  --imatrix-blend K           blend weight K for --imatrix-min-count (0 = off)\n"
        "  --write-test-imatrix FILE   uniform TEST-ONLY imatrix for the selected layers\n"
        "  --max-layers N              block_count=N, source layers 0..N-1 (all must exist)\n"
        "  --layers 3,4                source layers, renumbered to blk.0.. (not with --max-layers)\n"
        "  --layers A-B                keep blk.{source} names; metadata block_count stays the full model\n"
        "  --emit A-B                  on a remapped plan, write only those output layers\n"
        "  --split-part K              0-based llama-gguf-split part (with --split-count)\n"
        "  --split-count N             number of final parts; N>=2 writes name-0000k-of-0000N.gguf\n"
        "  --split-tensors N           total tensors across every part (default: count a full plan)\n"
        "  --include-globals           write token_embd, output_norm, output in this part\n"
        "  --no-globals                skip those three tensors\n"
        "  --threads N                 default: all logical cores\n"
        "  --dense-type Q8_0|Q6_K|Q4_K dense, attention, indexer, shared-expert type\n"
        "  --override KEY=TYPE         KEY is experts, ffn_down_exps, hot_experts, attention,\n"
        "                              indexer, dense_ffn, shared_experts, router, norms,\n"
        "                              token_embd, output. experts= also sets down and hot.\n"
        "  --expert-profile FILE       {\"mode\":\"record\"|\"promote-layer\",\"layers\":{\"3\":[12]}}\n"
        "  --hot-expert-mode MODE      record (default) or promote-layer; overrides the file\n"
        "  --placeholder-missing       stand in for a missing token_embd, output, output_norm\n"
        "  --dump-dequant NAME OUT.npy dequant one tensor (HF name or layers.N.experts.E.w1)\n"
        "  --dry-run                   print output sizes, do not write a model\n"
        "  --bench                     time expert dequant+quant (no GGUF)\n"
        "  --bench-experts N           default 8\n"
        "  --exp-layer N               experiment: quantize every expert of MoE layer N\n"
        "  --exp-type TYPE             experiment expert type (default: recipe experts)\n"
        "  --exp-out FILE              experiment JSON report\n"
        "  --exp-method old|diag|gptq  experiment rounding: imatrix, per-expert diagonal, GPTQ feedback\n"
        "  --exp-pack FILE             experiment: also write the layer's expert tensors as a GGUF\n"
        "                              (a donor for the engine's --expert-source 30-30=FILE)\n"
        "  --exp-pack-frag DIR         experiment: write one raw file per (kind, expert) instead,\n"
        "                              so a pass can be split across turns\n"
        "  --exp-start N               experiment: first expert of the pass (default 0)\n"
        "  --pack-merge DIR            assemble the layer pack from --exp-pack-frag's directory\n"
        "  --gptq-prior N              tokens' worth of the layer-wide Hessian mixed into each expert (16384)\n"
        "  --gptq-damp X               GPTQ diagonal damping as a fraction of mean(diag) (0.1)\n"
        "  --exp-holdout X             share of each expert's rows held out of the Hessian (0.2)\n"
        "  --gptq-selftest N           check the Cholesky/inverse against a naive reference and exit\n"
        "  --exp-x FILE.npy            experiment: routed-expert input [tokens, n_embd]\n"
        "  --exp-dumps DIR             experiment: dump dir with expert_XXX_hidden/token_idx.npy\n"
        "  --exp-experts N             experiment: limit to the first N experts (default all)\n"
        "  --nvfp4-check               report whether expert scales fold into GGML_TYPE_NVFP4\n"
        "  --mmproj OUT.gguf           F16 vision projector via llama.cpp (skips if shards are absent)\n"
        "  --llama-cpp DIR             llama.cpp tree for --mmproj\n",
        stderr);
}

std::filesystem::path exe_dir() {
#ifdef _WIN32
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) {
        return {};
    }
    buf.resize(n);
    return std::filesystem::path(buf).parent_path();
#else
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = 0;
        return std::filesystem::path(buf).parent_path();
    }
    return {};
#endif
}

bool take(int & i, int argc, char ** argv, const char * key, std::string & out) {
    const std::string a = argv[i];
    const std::string pref = std::string(key) + "=";
    if (a == key) {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + key);
        }
        out = argv[++i];
        return true;
    }
    if (a.compare(0, pref.size(), pref) == 0) {
        out = a.substr(pref.size());
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        mmq::Options opt;
        opt.exe_dir = exe_dir();
        if (argc < 2) {
            usage();
            return 2;
        }
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "-h" || a == "--help") {
                usage();
                return 0;
            }
            std::string val;
            if (take(i, argc, argv, "--model", val)) {
                opt.model = val;
            } else if (take(i, argc, argv, "--recipe", val)) {
                opt.recipe = val;
            } else if (take(i, argc, argv, "--out", val)) {
                opt.out = val;
            } else if (take(i, argc, argv, "--imatrix", val)) {
                opt.imatrix = val;
            } else if (take(i, argc, argv, "--imatrix-min-count", val)) {
                opt.imatrix_min_count = std::stoi(val);
            } else if (take(i, argc, argv, "--imatrix-blend", val)) {
                opt.imatrix_blend = std::stof(val);
            } else if (take(i, argc, argv, "--exp-layer", val)) {
                opt.exp_layer = std::stoi(val);
            } else if (take(i, argc, argv, "--exp-type", val)) {
                opt.exp_type = val;
            } else if (take(i, argc, argv, "--exp-out", val)) {
                opt.exp_out = val;
            } else if (take(i, argc, argv, "--exp-x", val)) {
                opt.exp_x = val;
            } else if (take(i, argc, argv, "--exp-dumps", val)) {
                opt.exp_dumps = val;
            } else if (take(i, argc, argv, "--exp-experts", val)) {
                opt.exp_experts = std::stoi(val);
            } else if (take(i, argc, argv, "--exp-method", val)) {
                opt.exp_method = val;
            } else if (take(i, argc, argv, "--exp-pack", val)) {
                opt.exp_pack = val;
            } else if (take(i, argc, argv, "--exp-pack-frag", val)) {
                opt.exp_pack_frag = val;
            } else if (take(i, argc, argv, "--pack-merge", val)) {
                opt.pack_merge = val;
            } else if (take(i, argc, argv, "--exp-start", val)) {
                opt.exp_start = std::stoi(val);
            } else if (take(i, argc, argv, "--gptq-prior", val)) {
                opt.gptq_prior = std::stod(val);
            } else if (take(i, argc, argv, "--gptq-damp", val)) {
                opt.gptq_damp = std::stod(val);
            } else if (take(i, argc, argv, "--exp-holdout", val)) {
                opt.exp_holdout = std::stod(val);
            } else if (take(i, argc, argv, "--gptq-selftest", val)) {
                opt.gptq_selftest = std::stoi(val);
            } else if (take(i, argc, argv, "--write-test-imatrix", val)) {
                opt.write_imatrix = val;
            } else if (take(i, argc, argv, "--max-layers", val)) {
                opt.max_layers = std::stoi(val);
            } else if (take(i, argc, argv, "--layers", val)) {
                opt.layers.clear();
                opt.keep_indices = false;
                if (val.find(',') == std::string::npos && val.find('-') != std::string::npos) {
                    const auto dash = val.find('-');
                    if (dash == 0 || dash + 1 >= val.size()) {
                        throw std::runtime_error("--layers range expects A-B");
                    }
                    const int lo = std::stoi(val.substr(0, dash));
                    const int hi = std::stoi(val.substr(dash + 1));
                    if (hi < lo) {
                        throw std::runtime_error("--layers range is empty");
                    }
                    opt.keep_indices = true;
                    for (int layer = lo; layer <= hi; ++layer) {
                        opt.layers.push_back(layer);
                    }
                } else {
                    std::string cur;
                    for (size_t k = 0; k <= val.size(); ++k) {
                        if (k == val.size() || val[k] == ',') {
                            if (!cur.empty()) {
                                opt.layers.push_back(std::stoi(cur));
                                cur.clear();
                            }
                        } else {
                            cur.push_back(val[k]);
                        }
                    }
                }
            } else if (take(i, argc, argv, "--emit", val)) {
                const auto dash = val.find('-');
                if (dash == std::string::npos || dash == 0 || dash + 1 >= val.size()) {
                    throw std::runtime_error("--emit expects A-B");
                }
                opt.emit_lo = std::stoi(val.substr(0, dash));
                opt.emit_hi = std::stoi(val.substr(dash + 1));
            } else if (take(i, argc, argv, "--split-part", val)) {
                opt.split_part = std::stoi(val);
            } else if (take(i, argc, argv, "--split-count", val)) {
                opt.split_count = std::stoi(val);
            } else if (take(i, argc, argv, "--split-tensors", val)) {
                opt.split_tensors = std::stoi(val);
            } else if (a == "--include-globals") {
                opt.include_globals = true;
                opt.include_globals_set = true;
            } else if (a == "--no-globals") {
                opt.no_globals = true;
            } else if (take(i, argc, argv, "--threads", val)) {
                opt.threads = std::stoi(val);
            } else if (take(i, argc, argv, "--dense-type", val)) {
                opt.dense_type = val;
            } else if (take(i, argc, argv, "--override", val)) {
                const auto eq = val.find('=');
                if (eq == std::string::npos) {
                    throw std::runtime_error("--override expects KEY=TYPE");
                }
                opt.overrides.emplace_back(val.substr(0, eq), val.substr(eq + 1));
            } else if (take(i, argc, argv, "--expert-profile", val)) {
                opt.expert_profile = val;
            } else if (take(i, argc, argv, "--hot-expert-mode", val)) {
                opt.hot_mode = val;
                opt.hot_mode_set = true;
            } else if (a == "--placeholder-missing") {
                opt.placeholder = true;
            } else if (take(i, argc, argv, "--dump-dequant", val)) {
                opt.dump_name = val;
                if (i + 1 >= argc) {
                    throw std::runtime_error("--dump-dequant needs an output .npy path");
                }
                opt.dump_out = argv[++i];
            } else if (a == "--dry-run") {
                opt.dry_run = true;
            } else if (a == "--bench") {
                opt.bench = true;
            } else if (take(i, argc, argv, "--bench-experts", val)) {
                opt.bench_experts = std::stoi(val);
            } else if (a == "--nvfp4-check") {
                opt.nvfp4_check = true;
            } else if (take(i, argc, argv, "--mmproj", val)) {
                opt.mmproj_out = val;
            } else if (take(i, argc, argv, "--llama-cpp", val)) {
                opt.llama_cpp = val;
            } else {
                throw std::runtime_error("unknown argument " + a);
            }
        }
        if (opt.model.empty()) {
            throw std::runtime_error("--model is required");
        }
        return mmq::run(opt);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
