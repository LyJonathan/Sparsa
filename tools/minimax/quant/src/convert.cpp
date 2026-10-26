#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include "convert.hpp"

#include "dequant.hpp"
#include "ggufw.hpp"
#include "gptq.hpp"
#include "json.hpp"
#include "quantize.hpp"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace mmq {
namespace {

namespace fs = std::filesystem;

constexpr const char * kTestNote =
    "TEST-ONLY uniform importance; not a real activation imatrix";

void logf(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fflush(stderr);
}

[[noreturn]] void die(const std::string & m) { throw std::runtime_error(m); }

int threads_of(const Options & opt) {
    if (opt.threads > 0) {
        return opt.threads;
    }
    const unsigned n = std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 1;
}

std::string human(uint64_t b) {
    char buf[80];
    std::snprintf(buf, sizeof(buf), "%.3f GB (%.3f GiB)",
                  static_cast<double>(b) / 1e9,
                  static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

std::string upper(std::string s) {
    for (char & c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

ggml_type parse_type(const std::string & s) {
    const std::string u = upper(s);
    if (u == "F32" || u == "FP32") return GGML_TYPE_F32;
    if (u == "Q8_0") return GGML_TYPE_Q8_0;
    if (u == "Q6_K") return GGML_TYPE_Q6_K;
    if (u == "Q4_K" || u == "Q4_K_M" || u == "Q4_K_S") return GGML_TYPE_Q4_K;
    if (u == "IQ1_S") return GGML_TYPE_IQ1_S;
    if (u == "IQ1_M") return GGML_TYPE_IQ1_M;
    if (u == "IQ2_XXS") return GGML_TYPE_IQ2_XXS;
    if (u == "IQ2_XS") return GGML_TYPE_IQ2_XS;
    if (u == "IQ2_S") return GGML_TYPE_IQ2_S;
    if (u == "IQ3_XXS") return GGML_TYPE_IQ3_XXS;
    if (u == "IQ3_S") return GGML_TYPE_IQ3_S;
    if (u == "IQ4_XS") return GGML_TYPE_IQ4_XS;
    if (u == "NVFP4") return GGML_TYPE_NVFP4;
    // Ported from ik_llama.cpp; ids from include/strata/kernels/iqk_types.hpp. See THIRD_PARTY_NOTICES.md.
    if (u == "IQ2_KT") return (ggml_type) strata::kernels::ikq::kIQ2KT;
    if (u == "IQ3_K") return (ggml_type) strata::kernels::ikq::kIQ3K;
    if (u == "IQ4_KS") return (ggml_type) strata::kernels::ikq::kIQ4KS;
    if (u == "IQ3_KT") return (ggml_type) strata::kernels::ikq::kIQ3KT;
    if (u == "IQ2_KL") return (ggml_type) strata::kernels::ikq::kIQ2KL;
    die("unknown quant type " + s);
}

uint32_t ftype_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return 0;
        case GGML_TYPE_Q8_0: return 7;
        case GGML_TYPE_Q4_K: return 15;
        case GGML_TYPE_Q6_K: return 18;
        case GGML_TYPE_IQ2_XXS: return 19;
        case GGML_TYPE_IQ2_XS: return 20;
        case GGML_TYPE_IQ3_XXS: return 23;
        case GGML_TYPE_IQ1_S: return 24;
        case GGML_TYPE_IQ3_S: return 26;
        case GGML_TYPE_IQ2_S: return 28;
        case GGML_TYPE_IQ4_XS: return 30;
        case GGML_TYPE_IQ1_M: return 31;
        case GGML_TYPE_NVFP4: return 39;
        // ik_llama.cpp's GGML_FTYPE_MOSTLY_* numbers, so general.file_type reads the same in both tools
        case (ggml_type) 160: return 142;   // IQ2_KT
        case (ggml_type) 161: return 131;   // IQ3_K
        case (ggml_type) 162: return 137;   // IQ4_KS
        case (ggml_type) 163: return 143;   // IQ3_KT
        case (ggml_type) 164: return 146;   // IQ2_KL
        default: return 0;
    }
}

uint64_t mat_bytes(ggml_type t, int64_t n_in, int64_t n_out, int64_t n_ex = 1) {
    if (n_in <= 0 || n_out <= 0 || n_ex <= 0) {
        die("bad mat shape");
    }
    const size_t row = row_size(t, n_in);
    if (row == 0) {
        die(std::string("no row size for ") + type_name(t) + " at " + std::to_string(n_in) + " values");
    }
    return static_cast<uint64_t>(row) * static_cast<uint64_t>(n_out) * static_cast<uint64_t>(n_ex);
}

// ---------------------------------------------------------------------------
// Checkpoint hparams. Read from text_config; the numbers below are M3.1 defaults
// used only when a key is absent.

struct Hyp {
    int n_layer = 60;
    int n_embd = 6144;
    int n_ff = 12288;
    int n_ff_exp = 3072;
    int n_head = 64;
    int n_head_kv = 4;
    int head_dim = 128;
    int n_rot = 64;
    int n_expert = 128;
    int n_used = 4;
    int n_shared = 1;
    int n_vocab = 200064;
    uint32_t n_ctx = 1048576;
    float rope = 10000000.f;
    float eps = 1e-6f;
    float exp_scale = 2.f;
    bool exp_norm = true;
    bool gemma = true;
    float swiglu_a = 1.702f;
    float swiglu_l = 7.f;
    int idx_heads = 4;
    int idx_key = 128;
    int idx_topk = 16;
    int idx_block = 128;
    int idx_local = 1;
    int idx_first = 0;
    int n_dense = 3;
    std::vector<int> moe_freq;
    std::string name = "MiniMax-M3.1-Flash";

    bool is_moe(int i) const {
        if (i < 0 || i >= static_cast<int>(moe_freq.size())) {
            die("layer id out of range");
        }
        return moe_freq[static_cast<size_t>(i)] != 0;
    }
};

int64_t jget_i(const Json & j, const char * k, int64_t dflt, bool required = false) {
    const Json * p = j.find(k);
    if (!p) {
        if (required) {
            die(std::string("config missing ") + k);
        }
        return dflt;
    }
    return p->as_i();
}

double jget_f(const Json & j, const char * k, double dflt) {
    const Json * p = j.find(k);
    return p ? p->as_f() : dflt;
}

Hyp load_hyp(const fs::path & model) {
    const fs::path cfg = model / "config.json";
    Json root = parse_json(read_bytes(cfg));
    const Json & tc = root.at("text_config");
    Hyp h;
    h.n_embd = static_cast<int>(jget_i(tc, "hidden_size", h.n_embd, true));
    h.n_layer = static_cast<int>(jget_i(tc, "num_hidden_layers", h.n_layer, true));
    h.n_head = static_cast<int>(jget_i(tc, "num_attention_heads", h.n_head, true));
    h.n_head_kv = static_cast<int>(jget_i(tc, "num_key_value_heads", h.n_head_kv, true));
    h.head_dim = static_cast<int>(jget_i(tc, "head_dim", h.head_dim, true));
    h.n_vocab = static_cast<int>(jget_i(tc, "vocab_size", h.n_vocab, true));
    h.n_ctx = static_cast<uint32_t>(jget_i(tc, "max_position_embeddings", h.n_ctx, true));
    h.n_ff_exp = static_cast<int>(jget_i(tc, "intermediate_size", h.n_ff_exp, true));
    h.n_ff = static_cast<int>(jget_i(tc, "dense_intermediate_size", h.n_ff_exp));
    h.n_rot = static_cast<int>(jget_i(tc, "rotary_dim", h.head_dim / 2));
    h.rope = static_cast<float>(jget_f(tc, "rope_theta", h.rope));
    h.eps = static_cast<float>(jget_f(tc, "rms_norm_eps", h.eps));
    h.n_expert = static_cast<int>(jget_i(tc, "num_local_experts", h.n_expert, true));
    h.n_used = static_cast<int>(jget_i(tc, "num_experts_per_tok", h.n_used, true));
    h.n_shared = static_cast<int>(jget_i(tc, "n_shared_experts", h.n_shared));
    h.exp_scale = static_cast<float>(jget_f(tc, "routed_scaling_factor", h.exp_scale));
    h.swiglu_a = static_cast<float>(jget_f(tc, "swiglu_alpha", h.swiglu_a));
    h.swiglu_l = static_cast<float>(jget_f(tc, "swiglu_limit", h.swiglu_l));
    if (const Json * g = tc.find("use_gemma_norm")) {
        h.gemma = g->as_b();
    }
    if (const Json * s = tc.find("scoring_func")) {
        if (s->as_s() != "sigmoid") {
            die("scoring_func is " + s->as_s() + "; this tool only writes sigmoid gating");
        }
    }
    if (const Json * freq = tc.find("moe_layer_freq")) {
        for (const Json & x : freq->arr) {
            h.moe_freq.push_back(static_cast<int>(x.as_i()));
        }
    }
    if (static_cast<int>(h.moe_freq.size()) != h.n_layer) {
        die("moe_layer_freq length does not match num_hidden_layers");
    }
    h.n_dense = 0;
    while (h.n_dense < h.n_layer && h.moe_freq[static_cast<size_t>(h.n_dense)] == 0) {
        ++h.n_dense;
    }
    if (const Json * sp = tc.find("sparse_attention_config")) {
        h.idx_heads = static_cast<int>(jget_i(*sp, "sparse_num_index_heads", h.idx_heads));
        h.idx_key = static_cast<int>(jget_i(*sp, "sparse_index_dim", h.idx_key));
        h.idx_topk = static_cast<int>(jget_i(*sp, "sparse_topk_blocks", h.idx_topk));
        h.idx_block = static_cast<int>(jget_i(*sp, "sparse_block_size", h.idx_block));
        h.idx_local = static_cast<int>(jget_i(*sp, "sparse_local_block", h.idx_local));
        if (const Json * f = sp->find("sparse_attention_freq")) {
            int first = static_cast<int>(f->arr.size());
            for (size_t i = 0; i < f->arr.size(); ++i) {
                if (f->arr[i].as_i() != 0) {
                    first = static_cast<int>(i);
                    break;
                }
            }
            for (size_t i = static_cast<size_t>(first); i < f->arr.size(); ++i) {
                if (f->arr[i].as_i() == 0) {
                    die("sparse_attention_freq must be a dense prefix followed by MSA layers");
                }
            }
            h.idx_first = first;
        }
    }
    return h;
}

// ---------------------------------------------------------------------------

struct Recipe {
    std::string name;
    uint32_t file_type = 7;
    bool require_imatrix = false;
    bool dynamic_down = false;
    bool disabled = false;
    std::string disabled_reason;
    ggml_type experts = GGML_TYPE_Q8_0;
    ggml_type ffn_down = GGML_TYPE_Q8_0;
    ggml_type hot = GGML_TYPE_Q8_0;
    ggml_type attention = GGML_TYPE_Q8_0;
    ggml_type indexer = GGML_TYPE_Q8_0;
    ggml_type dense = GGML_TYPE_Q8_0;
    ggml_type shared = GGML_TYPE_Q8_0;
    ggml_type router = GGML_TYPE_F32;
    ggml_type norms = GGML_TYPE_F32;
    ggml_type embd = GGML_TYPE_Q8_0;
    ggml_type output = GGML_TYPE_Q6_K;
};

ggml_type recipe_type(const Json & types, const char * key, ggml_type dflt) {
    const Json * p = types.find(key);
    return p ? parse_type(p->as_s()) : dflt;
}

fs::path find_recipe(const Options & opt) {
    const fs::path given = opt.recipe;
    if (given.empty()) {
        die("--recipe is required");
    }
    if (fs::is_regular_file(given)) {
        return given;
    }
    std::string name = given.filename().string();
    if (name.size() < 5 || name.substr(name.size() - 5) != ".json") {
        name += ".json";
    }
    std::vector<fs::path> cands;
    if (!opt.exe_dir.empty()) {
        cands.push_back(opt.exe_dir / "recipes" / name);
    }
    cands.push_back(fs::current_path() / "recipes" / name);
    cands.push_back(fs::current_path() / "tools" / "minimax" / "quant" / "recipes" / name);
    std::string tried;
    for (const fs::path & c : cands) {
        tried += "\n  " + c.string();
        if (fs::is_regular_file(c)) {
            return c;
        }
    }
    die("recipe not found: " + given.string() + tried);
}

void recompute_recipe(Recipe & r) {
    r.file_type = ftype_of(r.experts);
    r.require_imatrix = type_needs_imatrix(r.experts) || type_needs_imatrix(r.ffn_down) ||
                        type_needs_imatrix(r.hot) || type_needs_imatrix(r.attention) ||
                        type_needs_imatrix(r.indexer) || type_needs_imatrix(r.dense) ||
                        type_needs_imatrix(r.shared) || type_needs_imatrix(r.embd) ||
                        type_needs_imatrix(r.output);
}

Recipe load_recipe(const Options & opt) {
    const Json root = parse_json(read_bytes(find_recipe(opt)));
    Recipe r;
    r.name = root.find("name") ? root.at("name").as_s() : opt.recipe.stem().string();
    if (const Json * dis = root.find("disabled")) {
        r.disabled = dis->as_b();
    }
    if (const Json * why = root.find("disabled_reason")) {
        r.disabled_reason = why->as_s();
    }
    if (r.disabled) {
        std::string msg = "recipe " + r.name + " is disabled";
        if (!r.disabled_reason.empty()) {
            msg += ": " + r.disabled_reason;
        }
        die(msg);
    }
    r.dynamic_down = root.find("dynamic_down") ? root.at("dynamic_down").as_b() : false;
    const Json & t = root.at("types");
    r.experts = recipe_type(t, "experts", r.experts);
    r.ffn_down = recipe_type(t, "ffn_down_exps", r.experts);
    r.hot = recipe_type(t, "hot_experts", r.ffn_down);
    r.attention = recipe_type(t, "attention", r.attention);
    r.indexer = recipe_type(t, "indexer", r.indexer);
    r.dense = recipe_type(t, "dense_ffn", r.dense);
    r.shared = recipe_type(t, "shared_experts", r.shared);
    r.router = recipe_type(t, "router", r.router);
    r.norms = recipe_type(t, "norms", r.norms);
    r.embd = recipe_type(t, "token_embd", r.embd);
    r.output = recipe_type(t, "output", r.output);
    if (!opt.dense_type.empty()) {
        const ggml_type d = parse_type(opt.dense_type);
        r.attention = r.indexer = r.dense = r.shared = d;
    }
    for (const auto & ov : opt.overrides) {
        const ggml_type ty = parse_type(ov.second);
        const std::string & k = ov.first;
        if (k == "experts") {
            r.experts = r.ffn_down = r.hot = ty;
        } else if (k == "ffn_down_exps") {
            r.ffn_down = ty;
        } else if (k == "hot_experts") {
            r.hot = ty;
        } else if (k == "attention") {
            r.attention = ty;
        } else if (k == "indexer") {
            r.indexer = ty;
        } else if (k == "dense_ffn") {
            r.dense = ty;
        } else if (k == "shared_experts") {
            r.shared = ty;
        } else if (k == "router") {
            r.router = ty;
        } else if (k == "norms") {
            r.norms = ty;
        } else if (k == "token_embd") {
            r.embd = ty;
        } else if (k == "output") {
            r.output = ty;
        } else {
            die("unknown --override key " + k);
        }
    }
    recompute_recipe(r);
    return r;
}

std::string recipe_meta(const Recipe & r, const std::string & hot_mode) {
    std::ostringstream o;
    o << "{\"name\":\"" << r.name << "\",\"file_type\":" << r.file_type
      << ",\"require_imatrix\":" << (r.require_imatrix ? "true" : "false")
      << ",\"dynamic_down\":" << (r.dynamic_down ? "true" : "false")
      << ",\"hot_expert_mode\":\"" << hot_mode << "\",\"types\":{"
      << "\"experts\":\"" << type_name(r.experts) << "\","
      << "\"ffn_down_exps\":\"" << type_name(r.ffn_down) << "\","
      << "\"hot_experts\":\"" << type_name(r.hot) << "\","
      << "\"attention\":\"" << type_name(r.attention) << "\","
      << "\"indexer\":\"" << type_name(r.indexer) << "\","
      << "\"dense_ffn\":\"" << type_name(r.dense) << "\","
      << "\"shared_experts\":\"" << type_name(r.shared) << "\","
      << "\"router\":\"" << type_name(r.router) << "\","
      << "\"norms\":\"" << type_name(r.norms) << "\","
      << "\"token_embd\":\"" << type_name(r.embd) << "\","
      << "\"output\":\"" << type_name(r.output) << "\"}}";
    return o.str();
}

struct Profile {
    std::string raw;
    std::string mode;
    std::map<int, std::vector<int>> layers;
};

Profile load_profile(const Options & opt) {
    Profile p;
    if (opt.expert_profile.empty()) {
        return p;
    }
    p.raw = read_bytes(opt.expert_profile);
    const Json root = parse_json(p.raw);
    if (const Json * m = root.find("mode")) {
        p.mode = m->as_s();
    }
    if (const Json * layers = root.find("layers")) {
        for (const auto & kv : layers->obj) {
            int id = std::stoi(kv.first);
            std::vector<int> ids;
            for (const Json & x : kv.second.arr) {
                ids.push_back(static_cast<int>(x.as_i()));
            }
            p.layers.emplace(id, std::move(ids));
        }
    }
    return p;
}

// ---------------------------------------------------------------------------

struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t * data = nullptr;
    uint64_t nbytes = 0;
};

struct Catalog {
    std::vector<MappedFile> files;
    std::unordered_map<std::string, StTensor> map;
    const StTensor * find(const std::string & n) const {
        const auto it = map.find(n);
        return it == map.end() ? nullptr : &it->second;
    }
};

std::string shape_of(const StTensor & t) {
    std::string s = "[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i) {
            s += ",";
        }
        s += std::to_string(t.shape[i]);
    }
    return s + "]";
}

void load_shard(Catalog & cat, const fs::path & path) {
    MappedFile mf;
    mf.open(path);
    if (mf.size < 8) {
        die("short safetensors " + path.string());
    }
    uint64_t hlen = 0;
    std::memcpy(&hlen, mf.data, 8);
    if (hlen > mf.size - 8) {
        die("safetensors header exceeds file " + path.string());
    }
    const std::string header(reinterpret_cast<const char *>(mf.data + 8), static_cast<size_t>(hlen));
    Json root;
    try {
        root = parse_json(header);
    } catch (const std::exception & e) {
        die(path.filename().string() + ": " + e.what());
    }
    const uint64_t data0 = 8 + hlen;
    size_t added = 0;
    for (const auto & kv : root.obj) {
        if (kv.first.rfind("__", 0) == 0) {
            continue;
        }
        if (kv.second.type != Json::Object) {
            continue;
        }
        const Json * dtype = kv.second.find("dtype");
        const Json * shape = kv.second.find("shape");
        const Json * offs = kv.second.find("data_offsets");
        if (!dtype || !shape || !offs || offs->arr.size() < 2) {
            continue;
        }
        const int64_t a = offs->arr[0].as_i();
        const int64_t b = offs->arr[1].as_i();
        if (a < 0 || b < a) {
            die(kv.first + " bad data_offsets");
        }
        if (data0 + static_cast<uint64_t>(b) > mf.size) {
            die(kv.first + " extends past " + path.filename().string());
        }
        if (cat.map.find(kv.first) != cat.map.end()) {
            die("duplicate tensor " + kv.first);
        }
        StTensor t;
        t.dtype = dtype->as_s();
        for (const Json & d : shape->arr) {
            t.shape.push_back(d.as_i());
        }
        t.nbytes = static_cast<uint64_t>(b - a);
        t.data = mf.data + data0 + static_cast<uint64_t>(a);
        cat.map.emplace(kv.first, std::move(t));
        ++added;
    }
    logf("mapped %s  tensors %zu  bytes %s\n", path.filename().string().c_str(), added, human(mf.size).c_str());
    cat.files.push_back(std::move(mf));
}

// True when this shard is complete and (if needles is non-empty) its header names
// one of the requested tensors. A file that is still downloading is skipped.
bool shard_selected(const fs::path & path, const std::vector<std::string> & needles, bool & incomplete) {
    incomplete = false;
    std::error_code ec;
    const uint64_t sz = fs::file_size(path, ec);
    if (ec || sz < 8) {
        incomplete = true;
        logf("skip incomplete shard %s\n", path.filename().string().c_str());
        return false;
    }
    FILE * f = open_file(path, "rb");
    if (!f) {
        incomplete = true;
        logf("skip unreadable shard %s\n", path.filename().string().c_str());
        return false;
    }
    uint64_t hlen = 0;
    const bool got = std::fread(&hlen, 1, 8, f) == 8;
    if (!got || hlen > sz - 8 || hlen > (512ull << 20)) {
        std::fclose(f);
        incomplete = true;
        logf("skip incomplete shard %s (header not finished)\n", path.filename().string().c_str());
        return false;
    }
    std::string header(static_cast<size_t>(hlen), '\0');
    const size_t nread = std::fread(header.data(), 1, header.size(), f);
    std::fclose(f);
    if (nread != header.size()) {
        incomplete = true;
        logf("skip incomplete shard %s (short header)\n", path.filename().string().c_str());
        return false;
    }
    if (!needles.empty()) {
        bool hit = false;
        for (const std::string & n : needles) {
            if (header.find(n) != std::string::npos) {
                hit = true;
                break;
            }
        }
        if (!hit) {
            return false;
        }
    }
    uint64_t max_end = 0;
    try {
        const Json root = parse_json(header);
        for (const auto & kv : root.obj) {
            if (kv.second.type != Json::Object) {
                continue;
            }
            const Json * offs = kv.second.find("data_offsets");
            if (!offs || offs->arr.size() < 2) {
                continue;
            }
            const int64_t b = offs->arr[1].as_i();
            if (b > 0 && static_cast<uint64_t>(b) > max_end) {
                max_end = static_cast<uint64_t>(b);
            }
        }
    } catch (const std::exception & e) {
        die(path.filename().string() + ": " + e.what());
    }
    if (sz < 8 + hlen + max_end) {
        incomplete = true;
        logf("skip incomplete shard %s (payload short)\n", path.filename().string().c_str());
        return false;
    }
    return true;
}

Catalog load_catalog(const fs::path & model, const std::vector<std::string> & needles) {
    std::vector<fs::path> shards;
    for (const fs::directory_entry & ent : fs::directory_iterator(model)) {
        if (!ent.is_regular_file()) {
            continue;
        }
        const std::string name = ent.path().filename().string();
        if (name.rfind("model-", 0) != 0) {
            continue;
        }
        if (name.size() < 13 || name.compare(name.size() - 12, 12, ".safetensors") != 0) {
            continue;
        }
        shards.push_back(ent.path());
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) {
        die("no model-*-of-*.safetensors in " + model.string());
    }
    Catalog cat;
    cat.files.reserve(shards.size());
    cat.map.reserve(static_cast<size_t>(shards.size()) * 4000);
    int skipped = 0;
    for (const fs::path & p : shards) {
        bool incomplete = false;
        if (!shard_selected(p, needles, incomplete)) {
            if (incomplete) {
                ++skipped;
            }
            continue;
        }
        load_shard(cat, p);
    }
    if (cat.files.empty()) {
        die("no completed model-*-of-*.safetensors matched the selected layers in " + model.string() +
            (skipped ? " (" + std::to_string(skipped) + " still downloading)" : ""));
    }
    return cat;
}

// ---------------------------------------------------------------------------

struct Weight {
    enum class Store { Missing, F32, BF16, MXFP8, NVFP4 } store = Store::Missing;
    const StTensor * val = nullptr;
    const StTensor * scale = nullptr;
    float G = 1.f;
    int64_t val_stride = 0;
    int64_t scale_stride = 0;
};

float read_f32(const StTensor & t) {
    if (t.nbytes < 4 || t.dtype != "F32") {
        die("expected F32 scalar, got " + t.dtype);
    }
    float g = 0.f;
    std::memcpy(&g, t.data, 4);
    return g;
}

Weight resolve_weight(const Catalog & cat, const std::string & base, bool exact) {
    Weight w;
    if (!exact) {
        if (const StTensor * packed = cat.find(base + ".weight_packed")) {
            w.store = Weight::Store::NVFP4;
            w.val = packed;
            w.scale = cat.find(base + ".weight_scale");
            const StTensor * g = cat.find(base + ".weight_global_scale");
            if (!w.scale || !g) {
                die(base + " is NVFP4 but weight_scale or weight_global_scale is missing");
            }
            w.G = read_f32(*g);
            if (!(w.G > 0.f) || !std::isfinite(w.G)) {
                die(base + " weight_global_scale is not a positive finite float");
            }
            return w;
        }
    }
    const std::string name = exact ? base : base + ".weight";
    const StTensor * t = cat.find(name);
    if (!t) {
        return w;
    }
    w.val = t;
    if (t->dtype == "F8_E4M3" || t->dtype == "F8_E4M3FN") {
        w.scale = cat.find(base + ".weight_scale_inv");
        if (!w.scale) {
            die(name + " is F8_E4M3 but weight_scale_inv is missing");
        }
        w.store = Weight::Store::MXFP8;
    } else if (t->dtype == "BF16") {
        w.store = Weight::Store::BF16;
    } else if (t->dtype == "F32") {
        w.store = Weight::Store::F32;
    } else {
        die(name + " has unsupported dtype " + t->dtype);
    }
    return w;
}

void bind_shape(Weight & w, int64_t n_in, int64_t n_out, const std::string & what) {
    if (!w.val) {
        die(what + " has no data");
    }
    const StTensor & v = *w.val;
    if (v.shape.size() == 1) {
        if (n_out != 1 || v.shape[0] != n_in) {
            die(what + " shape " + shape_of(v) + " expected vector " + std::to_string(n_in));
        }
        w.val_stride = static_cast<int64_t>(v.nbytes);
        return;
    }
    if (v.shape.size() != 2 || v.shape[0] != n_out) {
        die(what + " shape " + shape_of(v) + " expected rows " + std::to_string(n_out));
    }
    if (w.store == Weight::Store::NVFP4) {
        if (v.shape[1] * 2 != n_in) {
            die(what + " packed width " + std::to_string(v.shape[1]) + " is not I/2 for I=" + std::to_string(n_in));
        }
        if (!w.scale || w.scale->shape.size() != 2 || w.scale->shape[0] != n_out || w.scale->shape[1] * 16 != n_in) {
            die(what + " weight_scale shape does not match [O, I/16]");
        }
        w.val_stride = v.shape[1];
        w.scale_stride = w.scale->shape[1];
    } else if (w.store == Weight::Store::MXFP8) {
        if (v.shape[1] != n_in) {
            die(what + " width " + std::to_string(v.shape[1]) + " != " + std::to_string(n_in));
        }
        if (!w.scale || w.scale->shape.size() != 2 || w.scale->shape[0] != n_out || w.scale->shape[1] * 32 != n_in) {
            die(what + " weight_scale_inv shape does not match [O, I/32]");
        }
        w.val_stride = n_in;
        w.scale_stride = w.scale->shape[1];
    } else if (w.store == Weight::Store::BF16) {
        if (v.shape[1] != n_in) {
            die(what + " width mismatch");
        }
        w.val_stride = n_in * 2;
    } else if (w.store == Weight::Store::F32) {
        if (v.shape[1] != n_in) {
            die(what + " width mismatch");
        }
        w.val_stride = n_in * 4;
    } else {
        die(what + " unbound store");
    }
}

struct ScaleFail {
    bool bad = false;
    int64_t row = 0;
    int64_t group = 0;
    uint8_t byte = 0;
    float target = 0.f;
    int nearest = 0;
    float nearest_v = 0.f;
};

ScaleFail first_inexact(const Weight & w, int64_t n_in, int64_t n_out, bool first_only) {
    ScaleFail f;
    if (w.store != Weight::Store::NVFP4 || !w.scale) {
        die("NVFP4 check on a non-NVFP4 tensor");
    }
    const int64_t ng = first_only ? 1 : n_in / 16;
    const int64_t nr = first_only ? 1 : n_out;
    for (int64_t r = 0; r < nr; ++r) {
        const uint8_t * sc = w.scale->data + static_cast<uint64_t>(r) * static_cast<uint64_t>(w.scale_stride);
        for (int64_t g = 0; g < ng; ++g) {
            const float target = e4m3_to_f32(sc[g]) / w.G;
            const E4Fit fit = fit_e4m3(target);
            if (!fit.exact) {
                f.bad = true;
                f.row = r;
                f.group = g;
                f.byte = sc[g];
                f.target = target;
                f.nearest = fit.byte;
                f.nearest_v = fit.decoded;
                return f;
            }
        }
    }
    return f;
}

void dequant_one(const Weight & w, int64_t row, int64_t n_in, float * dst, float add) {
    const uint8_t * src = w.val->data + static_cast<uint64_t>(row) * static_cast<uint64_t>(w.val_stride);
    if (w.store == Weight::Store::NVFP4) {
        const uint8_t * sc = w.scale->data + static_cast<uint64_t>(row) * static_cast<uint64_t>(w.scale_stride);
        dequant_nvfp4_row(src, sc, w.G, n_in, dst);
    } else if (w.store == Weight::Store::MXFP8) {
        const uint8_t * sc = w.scale->data + static_cast<uint64_t>(row) * static_cast<uint64_t>(w.scale_stride);
        dequant_mxfp8_row(src, sc, n_in, dst);
    } else if (w.store == Weight::Store::BF16) {
        dequant_bf16_row(reinterpret_cast<const uint16_t *>(src), n_in, dst, add);
        return;
    } else if (w.store == Weight::Store::F32) {
        const float * f = reinterpret_cast<const float *>(src);
        for (int64_t i = 0; i < n_in; ++i) {
            dst[i] = f[i] + add;
        }
        return;
    } else {
        die("dequant of missing weight");
    }
    if (add != 0.f) {
        for (int64_t i = 0; i < n_in; ++i) {
            dst[i] += add;
        }
    }
}

void dequant_rows(const Weight & w, int64_t row0, int64_t nrows, int64_t n_in, float * dst, float add, int threads) {
    if (threads < 1) {
        threads = 1;
    }
    if (static_cast<int64_t>(threads) > nrows) {
        threads = static_cast<int>(nrows);
    }
    if (threads <= 1) {
        for (int64_t r = 0; r < nrows; ++r) {
            dequant_one(w, row0 + r, n_in, dst + r * n_in, add);
        }
        return;
    }
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(static_cast<size_t>(threads));
    const int64_t chunk = (nrows + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int64_t a = static_cast<int64_t>(t) * chunk;
        if (a >= nrows) {
            break;
        }
        const int64_t b = std::min(nrows, a + chunk);
        workers.emplace_back([&, t, a, b]() {
            try {
                for (int64_t r = a; r < b; ++r) {
                    dequant_one(w, row0 + r, n_in, dst + r * n_in, add);
                }
            } catch (...) {
                errors[static_cast<size_t>(t)] = std::current_exception();
            }
        });
    }
    for (auto & th : workers) {
        th.join();
    }
    for (const auto & e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

// ---------------------------------------------------------------------------

enum class Role {
    Embd, Out, OutNorm,
    ANorm, Q, K, V, O, Qn, Kn, FNorm,
    Iq, Ik, Iqn, Ikn,
    Dg, Du, Dd,
    Router, Bias,
    Sg, Su, Sd,
    Eg, Eu, Ed
};

struct Item {
    std::string gguf;
    std::string hf;
    std::string wkey;
    std::string imx;           ///< imatrix row name when the output layer is remapped (else = gguf)
    bool exact = false;
    bool side_scale = false;   ///< f32 [n_expert]: per-expert weight_global_scale for an NVFP4 expert tensor
    Role role = Role::Embd;
    int src = -1;
    int out_i = -1;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    float add = 0.f;
    bool placeholder_ok = false;
    bool used_placeholder = false;
    uint64_t nbytes = 0;
};

std::string layer_base(int src) {
    return "language_model.model.layers." + std::to_string(src);
}

std::string blk_name(int i, const char * leaf, const char * suf = "weight") {
    return "blk." + std::to_string(i) + "." + leaf + "." + suf;
}

void push_item(std::vector<Item> & items, Item it) {
    it.nbytes = tensor_nbytes(it.type, it.ne);
    items.push_back(std::move(it));
}

std::vector<int> select_layers(const Options & opt, const Hyp & h) {
    if (opt.max_layers >= 0 && !opt.layers.empty()) {
        die("--max-layers and --layers are mutually exclusive");
    }
    std::vector<int> sel;
    if (!opt.layers.empty()) {
        for (int s : opt.layers) {
            if (s < 0 || s >= h.n_layer) {
                die("layer " + std::to_string(s) + " is outside 0.." + std::to_string(h.n_layer - 1));
            }
            sel.push_back(s);
        }
    } else if (opt.max_layers >= 0) {
        if (opt.max_layers <= 0 || opt.max_layers > h.n_layer) {
            die("--max-layers out of range");
        }
        for (int i = 0; i < opt.max_layers; ++i) {
            sel.push_back(i);
        }
    } else {
        for (int i = 0; i < h.n_layer; ++i) {
            sel.push_back(i);
        }
    }
    if (sel.empty()) {
        die("no layers selected");
    }
    return sel;
}

int leading_dense(const Hyp & h, const std::vector<int> & sel) {
    int lead = 0;
    bool seen_moe = false;
    for (int src : sel) {
        if (!h.is_moe(src)) {
            if (seen_moe) {
                die("output has a dense layer after a MoE layer; llama.cpp requires a dense prefix");
            }
            ++lead;
        } else {
            seen_moe = true;
        }
    }
    return lead;
}

std::vector<Item> build_plan(const Hyp & h, const Recipe & r, const std::vector<int> & sel,
                             const std::string & hot_mode, const Profile & prof, bool keep_indices) {
    const int meta_blocks = keep_indices ? h.n_layer : static_cast<int>(sel.size());
    if (h.idx_first > meta_blocks) {
        die("attention.indexer.first_layer would exceed block_count and llama.cpp would reject the file");
    }
    std::vector<Item> items;
    const float gadd = h.gemma ? 1.f : 0.f;

    Item emb;
    emb.gguf = "token_embd.weight";
    emb.hf = "language_model.model.embed_tokens";
    emb.role = Role::Embd;
    emb.type = r.embd;
    emb.ne[0] = h.n_embd;
    emb.ne[1] = h.n_vocab;
    emb.placeholder_ok = true;
    push_item(items, emb);

    Item on;
    on.gguf = "output_norm.weight";
    on.hf = "language_model.model.norm";
    on.role = Role::OutNorm;
    on.type = r.norms;
    on.ne[0] = h.n_embd;
    on.add = gadd;
    on.placeholder_ok = true;
    push_item(items, on);

    Item out;
    out.gguf = "output.weight";
    out.hf = "language_model.lm_head";
    out.role = Role::Out;
    out.type = r.output;
    out.ne[0] = h.n_embd;
    out.ne[1] = h.n_vocab;
    out.placeholder_ok = true;
    push_item(items, out);

    for (int oi = 0; oi < static_cast<int>(sel.size()); ++oi) {
        const int src = sel[static_cast<size_t>(oi)];
        const int blk = keep_indices ? src : oi;
        const std::string lb = layer_base(src);
        const bool hot = hot_mode == "promote-layer" && prof.layers.find(src) != prof.layers.end();
        auto add2 = [&](const std::string & gguf, const std::string & hf, Role role, ggml_type ty,
                        int64_t n_in, int64_t n_out, float add = 0.f) {
            Item it;
            it.gguf = gguf;
            it.hf = hf;
            it.role = role;
            it.src = src;
            it.out_i = oi;
            it.type = ty;
            it.ne[0] = n_in;
            it.ne[1] = n_out;
            it.add = add;
            push_item(items, it);
        };
        auto add1 = [&](const std::string & gguf, const std::string & hf, Role role, float add) {
            Item it;
            it.gguf = gguf;
            it.hf = hf;
            it.role = role;
            it.src = src;
            it.out_i = oi;
            it.type = r.norms;
            it.ne[0] = role == Role::ANorm || role == Role::FNorm ? h.n_embd : (role == Role::Qn || role == Role::Kn ? h.head_dim : h.idx_key);
            it.add = add;
            push_item(items, it);
        };

        add1(blk_name(blk, "attn_norm"), lb + ".input_layernorm", Role::ANorm, gadd);
        add2(blk_name(blk, "attn_q"), lb + ".self_attn.q_proj", Role::Q, r.attention, h.n_embd, static_cast<int64_t>(h.n_head) * h.head_dim);
        add2(blk_name(blk, "attn_k"), lb + ".self_attn.k_proj", Role::K, r.attention, h.n_embd, static_cast<int64_t>(h.n_head_kv) * h.head_dim);
        add2(blk_name(blk, "attn_v"), lb + ".self_attn.v_proj", Role::V, r.attention, h.n_embd, static_cast<int64_t>(h.n_head_kv) * h.head_dim);
        add2(blk_name(blk, "attn_output"), lb + ".self_attn.o_proj", Role::O, r.attention, static_cast<int64_t>(h.n_head) * h.head_dim, h.n_embd);
        add1(blk_name(blk, "attn_q_norm"), lb + ".self_attn.q_norm", Role::Qn, gadd);
        add1(blk_name(blk, "attn_k_norm"), lb + ".self_attn.k_norm", Role::Kn, gadd);
        add1(blk_name(blk, "ffn_norm"), lb + ".post_attention_layernorm", Role::FNorm, gadd);

        if ((keep_indices ? src : oi) >= h.idx_first) {
            add2(blk_name(blk, "indexer.q_proj"), lb + ".self_attn.index_q_proj", Role::Iq, r.indexer, h.n_embd,
                 static_cast<int64_t>(h.idx_heads) * h.idx_key);
            add2(blk_name(blk, "indexer.k_proj"), lb + ".self_attn.index_k_proj", Role::Ik, r.indexer, h.n_embd, h.idx_key);
            add1(blk_name(blk, "indexer.q_norm"), lb + ".self_attn.index_q_norm", Role::Iqn, gadd);
            add1(blk_name(blk, "indexer.k_norm"), lb + ".self_attn.index_k_norm", Role::Ikn, gadd);
        }

        if (!h.is_moe(src)) {
            add2(blk_name(blk, "ffn_gate"), lb + ".mlp.gate_proj", Role::Dg, r.dense, h.n_embd, h.n_ff);
            add2(blk_name(blk, "ffn_up"), lb + ".mlp.up_proj", Role::Du, r.dense, h.n_embd, h.n_ff);
            add2(blk_name(blk, "ffn_down"), lb + ".mlp.down_proj", Role::Dd, r.dense, h.n_ff, h.n_embd);
        } else {
            add2(blk_name(blk, "ffn_gate_inp"), lb + ".block_sparse_moe.gate", Role::Router, r.router, h.n_embd, h.n_expert);
            Item bias;
            bias.gguf = blk_name(blk, "exp_probs_b", "bias");
            bias.hf = lb + ".block_sparse_moe.e_score_correction_bias";
            bias.exact = true;
            bias.role = Role::Bias;
            bias.src = src;
            bias.out_i = oi;
            bias.type = r.router;
            bias.ne[0] = h.n_expert;
            push_item(items, bias);

            const int64_t shexp = static_cast<int64_t>(h.n_ff_exp) * h.n_shared;
            const std::string sh = lb + ".block_sparse_moe.shared_experts.";
            add2(blk_name(blk, "ffn_gate_shexp"), sh + "gate_proj", Role::Sg, r.shared, h.n_embd, shexp);
            add2(blk_name(blk, "ffn_up_shexp"), sh + "up_proj", Role::Su, r.shared, h.n_embd, shexp);
            add2(blk_name(blk, "ffn_down_shexp"), sh + "down_proj", Role::Sd, r.shared, shexp, h.n_embd);

            const ggml_type gate_ty = hot ? r.hot : r.experts;
            auto add_ex = [&](const char * leaf, const char * wkey, Role role, ggml_type ty, int64_t n_in, int64_t n_out) {
                Item it;
                it.gguf = blk_name(blk, leaf);
                // The calibration imatrix is keyed by the checkpoint's own layer names, so a remapped layer
                // (`--layers 40,40` writes blk.0) still reads the importance of its source layer.
                it.imx = blk_name(src, leaf);
                it.hf = lb + ".block_sparse_moe.experts";
                it.wkey = wkey;
                it.role = role;
                it.src = src;
                it.out_i = oi;
                it.type = ty;
                it.ne[0] = n_in;
                it.ne[1] = n_out;
                it.ne[2] = h.n_expert;
                push_item(items, it);
                if (ty == GGML_TYPE_NVFP4) {
                    // Lossless experts: the per-expert divisor is a side tensor, not a fold into the scale byte.
                    Item sc;
                    sc.gguf = blk_name(blk, leaf, "scale");
                    sc.hf = it.hf;
                    sc.wkey = wkey;
                    sc.side_scale = true;
                    sc.role = role;
                    sc.src = src;
                    sc.out_i = oi;
                    sc.type = GGML_TYPE_F32;
                    sc.ne[0] = h.n_expert;
                    push_item(items, sc);
                }
            };
            add_ex("ffn_gate_exps", "w1", Role::Eg, gate_ty, h.n_embd, h.n_ff_exp);
            add_ex("ffn_up_exps", "w3", Role::Eu, gate_ty, h.n_embd, h.n_ff_exp);
            add_ex("ffn_down_exps", "w2", Role::Ed, r.ffn_down, h.n_ff_exp, h.n_embd);
        }
    }
    return items;
}

struct Sizes {
    uint64_t embd = 0, output = 0, norms = 0, router = 0;
    uint64_t attn = 0, indexer = 0, dense = 0, shared = 0;
    uint64_t exps = 0, exps_down = 0;
    uint64_t total() const {
        return embd + output + norms + router + attn + indexer + dense + shared + exps + exps_down;
    }
};

Sizes project_full(const Hyp & h, const Recipe & r) {
    Sizes s;
    const int n_moe = h.n_layer - h.n_dense;
    s.embd = mat_bytes(r.embd, h.n_embd, h.n_vocab);
    s.output = mat_bytes(r.output, h.n_embd, h.n_vocab);
    s.norms = mat_bytes(r.norms, h.n_embd, 1);
    const uint64_t per_layer_norm =
        mat_bytes(r.norms, h.n_embd, 1) * 2 +
        mat_bytes(r.norms, h.head_dim, 1) * 2 +
        mat_bytes(r.norms, h.idx_key, 1) * 2;
    s.norms += per_layer_norm * static_cast<uint64_t>(h.n_layer);
    const int64_t q = static_cast<int64_t>(h.n_head) * h.head_dim;
    const int64_t kv = static_cast<int64_t>(h.n_head_kv) * h.head_dim;
    const uint64_t attn1 = mat_bytes(r.attention, h.n_embd, q) + mat_bytes(r.attention, h.n_embd, kv) * 2 +
                           mat_bytes(r.attention, q, h.n_embd);
    s.attn = attn1 * static_cast<uint64_t>(h.n_layer);
    const int64_t iq = static_cast<int64_t>(h.idx_heads) * h.idx_key;
    const uint64_t idx1 = mat_bytes(r.indexer, h.n_embd, iq) + mat_bytes(r.indexer, h.n_embd, h.idx_key);
    s.indexer = idx1 * static_cast<uint64_t>(h.n_layer - h.idx_first);
    const uint64_t dense1 = mat_bytes(r.dense, h.n_embd, h.n_ff) * 2 + mat_bytes(r.dense, h.n_ff, h.n_embd);
    s.dense = dense1 * static_cast<uint64_t>(h.n_dense);
    const int64_t sh = static_cast<int64_t>(h.n_ff_exp) * h.n_shared;
    const uint64_t sh1 = mat_bytes(r.shared, h.n_embd, sh) * 2 + mat_bytes(r.shared, sh, h.n_embd);
    s.shared = sh1 * static_cast<uint64_t>(n_moe);
    s.router = (mat_bytes(r.router, h.n_embd, h.n_expert) + mat_bytes(r.router, h.n_expert, 1)) * static_cast<uint64_t>(n_moe);
    s.exps = mat_bytes(r.experts, h.n_embd, h.n_ff_exp, h.n_expert) * 2 * static_cast<uint64_t>(n_moe);
    s.exps_down = mat_bytes(r.ffn_down, h.n_ff_exp, h.n_embd, h.n_expert) * static_cast<uint64_t>(n_moe);
    return s;
}

void print_sizes(const char * title, const Sizes & s) {
    logf("%s\n", title);
    logf("  token_embd     %s\n", human(s.embd).c_str());
    logf("  output         %s\n", human(s.output).c_str());
    logf("  norms          %s\n", human(s.norms).c_str());
    logf("  router+bias    %s\n", human(s.router).c_str());
    logf("  attention      %s\n", human(s.attn).c_str());
    logf("  indexer        %s\n", human(s.indexer).c_str());
    logf("  dense ffn      %s\n", human(s.dense).c_str());
    logf("  shared experts %s\n", human(s.shared).c_str());
    logf("  routed exps    %s\n", human(s.exps).c_str());
    logf("  routed down    %s\n", human(s.exps_down).c_str());
    logf("  total tensors  %s\n", human(s.total()).c_str());
}

Sizes sizes_of_plan(const std::vector<Item> & items) {
    Sizes s;
    for (const Item & it : items) {
        switch (it.role) {
            case Role::Embd: s.embd += it.nbytes; break;
            case Role::Out: s.output += it.nbytes; break;
            case Role::OutNorm:
            case Role::ANorm:
            case Role::Qn:
            case Role::Kn:
            case Role::FNorm:
            case Role::Iqn:
            case Role::Ikn: s.norms += it.nbytes; break;
            case Role::Router:
            case Role::Bias: s.router += it.nbytes; break;
            case Role::Q:
            case Role::K:
            case Role::V:
            case Role::O: s.attn += it.nbytes; break;
            case Role::Iq:
            case Role::Ik: s.indexer += it.nbytes; break;
            case Role::Dg:
            case Role::Du:
            case Role::Dd: s.dense += it.nbytes; break;
            case Role::Sg:
            case Role::Su:
            case Role::Sd: s.shared += it.nbytes; break;
            case Role::Eg:
            case Role::Eu: s.exps += it.nbytes; break;
            case Role::Ed: s.exps_down += it.nbytes; break;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------

struct Imp {
    std::unordered_map<std::string, std::vector<float>> rows;
    // Per-expert calibration token counts, only for stacked expert tensors.
    std::unordered_map<std::string, std::vector<float>> counts;
    std::vector<std::string> datasets;
    bool test_only = false;
    int blended_experts = 0;
    int empty_experts = 0;

    const float * at(const std::string & name, int64_t n_in, int64_t nmat, int64_t expert) const {
        const auto it = rows.find(name);
        if (it == rows.end()) {
            return nullptr;
        }
        const std::vector<float> & v = it->second;
        if (static_cast<int64_t>(v.size()) == n_in) {
            return v.data();
        }
        if (nmat > 0 && static_cast<int64_t>(v.size()) == n_in * nmat) {
            return v.data() + expert * n_in;
        }
        die("imatrix length for " + name + " is " + std::to_string(v.size()) + ", expected " +
            std::to_string(n_in) + " or " + std::to_string(n_in * nmat));
    }

    const std::vector<float> * counts_of(const std::string & name) const {
        const auto it = counts.find(name);
        return it == counts.end() ? nullptr : &it->second;
    }
};

void write_test_imatrix(const fs::path & path, const std::vector<Item> & items) {
    GgufWriter w;
    w.alignment = 32;
    w.open(path);
    w.add_str("general.type", "imatrix");
    w.add_u32("general.alignment", 32);
    w.add_str_array("imatrix.datasets", {kTestNote});
    w.add_u32("imatrix.chunk_count", 1);
    w.add_u32("imatrix.chunk_size", 512);
    struct Part {
        std::string name;
        int64_t n_in = 0;
        int64_t nmat = 1;
    };
    std::vector<Part> parts;
    for (const Item & it : items) {
        if (it.type == GGML_TYPE_F32) {
            continue;
        }
        Part p;
        p.name = it.gguf;
        p.n_in = it.ne[0];
        p.nmat = it.ne[2] > 0 ? it.ne[2] : 1;
        parts.push_back(p);
        {
            TensorDesc td;
            td.name = p.name + ".in_sum2";
            td.ne[0] = p.n_in;
            td.ne[1] = p.nmat;
            td.n_dims = gguf_n_dims(td.ne);
            td.type = GGML_TYPE_F32;
            td.nbytes = static_cast<uint64_t>(p.n_in) * static_cast<uint64_t>(p.nmat) * 4;
            w.add_tensor(td);
        }
        {
            TensorDesc td;
            td.name = p.name + ".counts";
            td.ne[0] = 1;
            td.ne[1] = p.nmat;
            td.n_dims = gguf_n_dims(td.ne);
            td.type = GGML_TYPE_F32;
            td.nbytes = static_cast<uint64_t>(p.nmat) * 4;
            w.add_tensor(td);
        }
    }
    w.write_header();
    for (const Part & p : parts) {
        std::vector<float> row(static_cast<size_t>(p.n_in), 1.f);
        for (int64_t m = 0; m < p.nmat; ++m) {
            w.write_payload(row.data(), row.size() * sizeof(float));
        }
        w.end_tensor();
        std::vector<float> counts(static_cast<size_t>(p.nmat), 1.f);
        w.write_payload(counts.data(), counts.size() * sizeof(float));
        w.end_tensor();
    }
    w.close();
    logf("wrote TEST-ONLY imatrix %s (%zu tensors)\n", path.string().c_str(), parts.size());
    logf("WARNING: %s\n", kTestNote);
}

Imp load_imatrix(const fs::path & path, int min_count, float blend) {
    ggml_context * gctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx = &gctx;
    gguf_context * gf = gguf_init_from_file(path.string().c_str(), params);
    if (!gf || !gctx) {
        if (gf) {
            gguf_free(gf);
        }
        if (gctx) {
            ggml_free(gctx);
        }
        die("failed to read imatrix " + path.string());
    }
    Imp imp;
    const int64_t dk = gguf_find_key(gf, "imatrix.datasets");
    if (dk >= 0 && gguf_get_kv_type(gf, dk) == GGUF_TYPE_ARRAY) {
        const int64_t n = static_cast<int64_t>(gguf_get_arr_n(gf, dk));
        for (int64_t i = 0; i < n; ++i) {
            imp.datasets.emplace_back(gguf_get_arr_str(gf, dk, static_cast<size_t>(i)));
        }
    }
    for (const std::string & d : imp.datasets) {
        logf("imatrix dataset: %s\n", d.c_str());
        if (d.find("TEST-ONLY") != std::string::npos) {
            imp.test_only = true;
        }
    }
    if (imp.test_only) {
        logf("WARNING: imatrix is TEST-ONLY uniform importance, not a real activation imatrix\n");
    }
    std::unordered_map<std::string, const float *> sums;
    std::unordered_map<std::string, std::pair<const float *, int64_t>> count_tensors;
    std::unordered_map<std::string, int64_t> sum_n;
    for (ggml_tensor * cur = ggml_get_first_tensor(gctx); cur; cur = ggml_get_next_tensor(gctx, cur)) {
        std::string name = cur->name;
        const std::string s2 = ".in_sum2";
        const std::string cs = ".counts";
        if (name.size() > s2.size() && name.compare(name.size() - s2.size(), s2.size(), s2) == 0) {
            name.resize(name.size() - s2.size());
            sums[name] = static_cast<const float *>(cur->data);
            sum_n[name] = ggml_nelements(cur);
        } else if (name.size() > cs.size() && name.compare(name.size() - cs.size(), cs.size(), cs) == 0) {
            name.resize(name.size() - cs.size());
            count_tensors[name] = {static_cast<const float *>(cur->data), ggml_nelements(cur)};
        }
    }
    for (const auto & kv : sums) {
        const auto cit = count_tensors.find(kv.first);
        if (cit == count_tensors.end() || cit->second.second <= 0) {
            die("imatrix missing counts for " + kv.first);
        }
        const int64_t ncounts = cit->second.second;
        const int64_t nsum = sum_n[kv.first];
        if (nsum % ncounts != 0) {
            die("imatrix sum/count mismatch for " + kv.first);
        }
        std::vector<float> row(static_cast<size_t>(nsum), 1.f);
        const float * s = kv.second;
        const float * c = cit->second.first;
        const int64_t ne0 = nsum / ncounts;
        for (int64_t j = 0; j < ncounts; ++j) {
            const float count = c[j];
            const float div = count > 0.f ? count : 1.f;
            for (int64_t i = 0; i < ne0; ++i) {
                row[static_cast<size_t>(j * ne0 + i)] = s[j * ne0 + i] / div;
            }
        }
        if (ncounts > 1) {
            // Stacked expert tensor. The per-layer mean is the average of every
            // expert row that saw at least one token.
            std::vector<double> mean(static_cast<size_t>(ne0), 0.0);
            int used = 0;
            for (int64_t j = 0; j < ncounts; ++j) {
                if (!(c[j] > 0.f)) {
                    continue;
                }
                for (int64_t i = 0; i < ne0; ++i) {
                    mean[static_cast<size_t>(i)] += row[static_cast<size_t>(j * ne0 + i)];
                }
                ++used;
            }
            if (used == 0) {
                logf("WARNING: imatrix %s has no calibration tokens on any expert; using uniform importance\n",
                     kv.first.c_str());
                std::fill(mean.begin(), mean.end(), 1.0);
            } else {
                for (double & v : mean) {
                    v /= used;
                }
            }
            std::vector<float> mean_f(static_cast<size_t>(ne0));
            for (int64_t i = 0; i < ne0; ++i) {
                mean_f[static_cast<size_t>(i)] = static_cast<float>(mean[static_cast<size_t>(i)]);
            }
            std::vector<float> counts_f(static_cast<size_t>(ncounts));
            for (int64_t j = 0; j < ncounts; ++j) {
                counts_f[static_cast<size_t>(j)] = c[j];
                float * r = row.data() + static_cast<size_t>(j * ne0);
                const float count = c[j];
                if (!(count > 0.f)) {
                    // Rule from the calibration study: a zero-count expert must
                    // not fall back to 1.0 across the board.
                    std::memcpy(r, mean_f.data(), static_cast<size_t>(ne0) * sizeof(float));
                    ++imp.empty_experts;
                    continue;
                }
                if (blend > 0.f && (min_count <= 0 || count < static_cast<float>(min_count))) {
                    const float a = count / (count + blend);
                    for (int64_t i = 0; i < ne0; ++i) {
                        r[i] = a * r[i] + (1.f - a) * mean_f[static_cast<size_t>(i)];
                    }
                    ++imp.blended_experts;
                }
            }
            imp.counts.emplace(kv.first, std::move(counts_f));
        }
        imp.rows.emplace(kv.first, std::move(row));
    }
    gguf_free(gf);
    ggml_free(gctx);
    logf("loaded imatrix entries %zu", imp.rows.size());
    if (blend > 0.f) {
        logf("  blend K=%.0f min-count=%d blended=%d zero-count=%d", blend, min_count,
             imp.blended_experts, imp.empty_experts);
    } else if (imp.empty_experts > 0) {
        logf("  zero-count experts filled from the layer mean: %d", imp.empty_experts);
    }
    logf("\n");
    return imp;
}

// ---------------------------------------------------------------------------

void write_npy(const fs::path & path, const float * data, int64_t rows, int64_t cols) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) +
                         ", " + std::to_string(cols) + "), }";
    int pad = static_cast<int>((64 - ((10 + header.size() + 1) % 64)) % 64);
    header.append(static_cast<size_t>(pad), ' ');
    header.push_back('\n');
    FILE * f = open_file(path, "wb");
    if (!f) {
        die("cannot create " + path.string());
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    std::fwrite(magic, 1, 8, f);
    const uint16_t hlen = static_cast<uint16_t>(header.size());
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(header.data(), 1, header.size(), f);
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (std::fwrite(data, sizeof(float), n, f) != n) {
        std::fclose(f);
        die("short npy write");
    }
    std::fclose(f);
}

bool parse_expert_shorthand(const std::string & s, int & layer, int & expert, std::string & wkey) {
    int n = 0;
    if (std::sscanf(s.c_str(), "layers.%d.experts.%d.%n", &layer, &expert, &n) == 2 && n > 0) {
        wkey = s.substr(static_cast<size_t>(n));
        return wkey == "w1" || wkey == "w2" || wkey == "w3";
    }
    n = 0;
    if (std::sscanf(s.c_str(), "layer%d.expert%d.%n", &layer, &expert, &n) == 2 && n > 0) {
        wkey = s.substr(static_cast<size_t>(n));
        return wkey == "w1" || wkey == "w2" || wkey == "w3";
    }
    return false;
}

std::string strip_weight_suffix(std::string s) {
    const char * suf[] = {".weight_packed", ".weight_global_scale", ".weight_scale_inv", ".weight_scale", ".weight"};
    for (const char * x : suf) {
        const size_t n = std::strlen(x);
        if (s.size() > n && s.compare(s.size() - n, n, x) == 0) {
            s.resize(s.size() - n);
            return s;
        }
    }
    return s;
}

void dump_dequant(const Catalog & cat, const Options & opt, int threads) {
    int layer = 0, expert = 0;
    std::string wkey;
    Weight w;
    std::string label = opt.dump_name;
    if (parse_expert_shorthand(opt.dump_name, layer, expert, wkey)) {
        const std::string base = layer_base(layer) + ".block_sparse_moe.experts." + std::to_string(expert) + "." + wkey;
        w = resolve_weight(cat, base, false);
        label = base;
    } else {
        const std::string base = strip_weight_suffix(opt.dump_name);
        w = resolve_weight(cat, base, false);
        if (w.store == Weight::Store::Missing) {
            w = resolve_weight(cat, opt.dump_name, true);
        }
        label = base;
    }
    if (w.store == Weight::Store::Missing) {
        die("dump tensor not found: " + opt.dump_name);
    }
    int64_t n_out = 1, n_in = 0;
    if (w.val->shape.size() == 1) {
        n_in = w.val->shape[0];
        n_out = 1;
    } else if (w.store == Weight::Store::NVFP4) {
        n_out = w.val->shape[0];
        n_in = w.val->shape[1] * 2;
    } else {
        n_out = w.val->shape[0];
        n_in = w.val->shape[1];
    }
    bind_shape(w, n_in, n_out, label);
    std::vector<float> dst(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
    dequant_rows(w, 0, n_out, n_in, dst.data(), 0.f, threads);
    if (!opt.dump_out.parent_path().empty()) {
        fs::create_directories(opt.dump_out.parent_path());
    }
    write_npy(opt.dump_out, dst.data(), n_out, n_in);
    logf("dumped %s shape (%lld, %lld) store G=%.9g -> %s\n", label.c_str(),
         static_cast<long long>(n_out), static_cast<long long>(n_in),
         w.store == Weight::Store::NVFP4 ? w.G : 0.0, opt.dump_out.string().c_str());
}

void nvfp4_check(const Catalog & cat, const Hyp & h) {
    int layer = -1;
    for (int i = 0; i < h.n_layer; ++i) {
        if (!h.is_moe(i)) {
            continue;
        }
        const std::string b = layer_base(i) + ".block_sparse_moe.experts.0.w1.weight_packed";
        if (cat.find(b)) {
            layer = i;
            break;
        }
    }
    if (layer < 0) {
        die("nvfp4-check: no expert NVFP4 tensor in the completed shards");
    }
    logf("NVFP4 check on source layer %d. GGML_TYPE_NVFP4 id %d, block 36 bytes, QK 64.\n", layer,
         static_cast<int>(GGML_TYPE_NVFP4));
    logf("MM-NVFP4 copies the packed nibbles and the E4M3 scale bytes verbatim and stores the divisor\n");
    logf("weight_global_scale as blk.N.ffn_{gate,up,down}_exps.scale (f32 [n_expert]). No fold, no gain error.\n");
    const char * keys[] = {"w1", "w2", "w3"};
    int64_t n_in_of[3] = {h.n_embd, h.n_ff_exp, h.n_embd};
    int64_t n_out_of[3] = {h.n_ff_exp, h.n_embd, h.n_ff_exp};
    int checked = 0;
    bool any = false;
    for (int e = 0; e < 2; ++e) {
        for (int k = 0; k < 3; ++k) {
            const std::string base = layer_base(layer) + ".block_sparse_moe.experts." + std::to_string(e) + "." + keys[k];
            Weight w = resolve_weight(cat, base, false);
            if (w.store != Weight::Store::NVFP4) {
                die(base + " is not NVFP4; nothing to copy");
            }
            any = true;
            bind_shape(w, n_in_of[k], n_out_of[k], base);
            const ScaleFail bad = first_inexact(w, n_in_of[k], n_out_of[k], false);
            logf("  %s G=%.9g  scale bytes fold into E4M3: %s (side tensor keeps them exact either way)\n", base.c_str(),
                 w.G, bad.bad ? "no" : "yes");
            ++checked;
            bool neg = false;
            for (int64_t r = 0; r < n_out_of[k] && !neg; ++r) {
                const uint8_t * sc = w.scale->data + static_cast<uint64_t>(r) * static_cast<uint64_t>(w.scale_stride);
                for (int64_t g = 0; g < n_in_of[k] / 16; ++g) {
                    if (sc[g] & 0x80) {
                        neg = true;
                        break;
                    }
                }
            }
            if (neg) {
                die(base + " has a negative E4M3 scale byte; block_nvfp4 stores unsigned scales");
            }
            if (w.val->nbytes != static_cast<uint64_t>(n_out_of[k] * (n_in_of[k] / 2))) {
                die(base + " packed size is not O * I/2");
            }
        }
    }
    if (!any) {
        die("nvfp4-check: expert tensors were not NVFP4");
    }
    logf("%d sampled matrices: rows are a multiple of 64 and every scale byte is a non-negative E4M3 value.\n", checked);
    logf("MM-NVFP4 can be written exactly for this checkpoint.\n");
}

// ---------------------------------------------------------------------------

const float * importance_for(const Imp * imp, const Item & it, int64_t expert) {
    if (!imp) {
        return nullptr;
    }
    const int64_t nmat = it.ne[2] > 0 ? it.ne[2] : 1;
    return imp->at(it.imx.empty() ? it.gguf : it.imx, it.ne[0], nmat, expert);
}

void require_importance(const Item & it, const float * im) {
    if (type_needs_imatrix(it.type) && !im) {
        die(std::string(type_name(it.type)) + " requires an imatrix entry for " + it.gguf);
    }
}

void emit_constant(GgufWriter & w, ggml_type type, int64_t n_in, int64_t nrows, float value, const float * im) {
    if (type == GGML_TYPE_F32) {
        std::vector<float> row(static_cast<size_t>(n_in), value);
        const size_t bytes = row.size() * sizeof(float);
        const int64_t chunk = 64;
        std::vector<float> buf(static_cast<size_t>(chunk) * static_cast<size_t>(n_in));
        for (int64_t i = 0; i < chunk; ++i) {
            std::memcpy(buf.data() + static_cast<size_t>(i) * static_cast<size_t>(n_in), row.data(), bytes);
        }
        int64_t left = nrows;
        while (left > 0) {
            const int64_t n = std::min(left, chunk);
            w.write_payload(buf.data(), static_cast<size_t>(n) * bytes);
            left -= n;
        }
        return;
    }
    std::vector<float> row(static_cast<size_t>(n_in), value);
    const size_t row_q = row_size(type, n_in);
    std::vector<uint8_t> q(row_q);
    quantize_rows(type, row.data(), q.data(), 1, n_in, im, 1);
    const int64_t chunk = 128;
    std::vector<uint8_t> buf(row_q * static_cast<size_t>(chunk));
    for (int64_t i = 0; i < chunk; ++i) {
        std::memcpy(buf.data() + static_cast<size_t>(i) * row_q, q.data(), row_q);
    }
    int64_t left = nrows;
    while (left > 0) {
        const int64_t n = std::min(left, chunk);
        w.write_payload(buf.data(), row_q * static_cast<size_t>(n));
        left -= n;
    }
}

void emit_weight(GgufWriter & w, const Weight & wt, ggml_type type, int64_t n_in, int64_t n_out,
                 const float * im, int threads, float add) {
    const size_t row_q = type == GGML_TYPE_F32 ? static_cast<size_t>(n_in) * 4 : row_size(type, n_in);
    if (type == GGML_TYPE_F32 && wt.store == Weight::Store::F32 && add == 0.f) {
        const uint64_t n = static_cast<uint64_t>(n_in) * static_cast<uint64_t>(n_out) * 4;
        if (wt.val->nbytes < n) {
            die("F32 tensor shorter than expected");
        }
        uint64_t off = 0;
        while (off < n) {
            const size_t c = static_cast<size_t>(std::min<uint64_t>(n - off, 8ull * 1024ull * 1024ull));
            w.write_payload(wt.val->data + off, c);
            off += c;
        }
        return;
    }
    if (type == GGML_TYPE_NVFP4) {
        if (wt.store != Weight::Store::NVFP4) {
            die("NVFP4 output requested for a tensor that is not NVFP4");
        }
        if (n_in % 64 != 0) {
            die("NVFP4 row length is not a multiple of 64");
        }
        std::vector<uint8_t> q(row_q);
        for (int64_t r = 0; r < n_out; ++r) {
            const uint8_t * packed = wt.val->data + static_cast<uint64_t>(r) * static_cast<uint64_t>(wt.val_stride);
            const uint8_t * sc = wt.scale->data + static_cast<uint64_t>(r) * static_cast<uint64_t>(wt.scale_stride);
            // Exact: nibble remap + verbatim E4M3 scale bytes. G goes to the side tensor.
            pack_nvfp4_row_exact(packed, sc, n_in, q.data());
            w.write_payload(q.data(), q.size());
        }
        return;
    }
    const int64_t budget = 64 * 1024 * 1024;
    int64_t batch = std::max<int64_t>(1, budget / (n_in * static_cast<int64_t>(sizeof(float))));
    if (batch > n_out) {
        batch = n_out;
    }
    std::vector<float> f32(static_cast<size_t>(batch) * static_cast<size_t>(n_in));
    std::vector<uint8_t> q(static_cast<size_t>(batch) * row_q);
    for (int64_t r = 0; r < n_out; r += batch) {
        const int64_t n = std::min(batch, n_out - r);
        dequant_rows(wt, r, n, n_in, f32.data(), add, threads);
        quantize_rows(type, f32.data(), q.data(), n, n_in, im, threads);
        w.write_payload(q.data(), row_q * static_cast<size_t>(n));
    }
}

std::string expert_base(const Item & it, int e) {
    return it.hf + "." + std::to_string(e) + "." + it.wkey;
}

bool is_expert(const Item & it) {
    return it.role == Role::Eg || it.role == Role::Eu || it.role == Role::Ed;
}

// blk.N.ffn_*_exps.scale: f32 [n_expert], one weight_global_scale per expert, in expert
// order. The value is the divisor from spec A (W = e2m1 * e4m3(scale) / G).
void emit_side_scale(GgufWriter & w, const Catalog & cat, const Item & it, const Hyp & h) {
    std::vector<float> g(static_cast<size_t>(h.n_expert));
    for (int e = 0; e < h.n_expert; ++e) {
        Weight wt = resolve_weight(cat, expert_base(it, e), false);
        if (wt.store != Weight::Store::NVFP4) {
            die(it.gguf + ": expert " + std::to_string(e) + " is not NVFP4; a side scale needs one");
        }
        g[static_cast<size_t>(e)] = wt.G;
    }
    w.write_payload(g.data(), g.size() * sizeof(float));
}

void preflight(const Catalog & cat, std::vector<Item> & items, const Hyp & h, bool placeholder) {
    std::vector<std::string> missing;
    auto note = [&](const std::string & n) {
        if (missing.size() < 20) {
            missing.push_back(n);
        } else if (missing.size() == 20) {
            missing.push_back("...");
        }
    };
    for (Item & it : items) {
        if (it.side_scale) {
            continue;
        }
        if (is_expert(it)) {
            for (int e = 0; e < h.n_expert; ++e) {
                const std::string base = expert_base(it, e);
                Weight w = resolve_weight(cat, base, false);
                if (w.store == Weight::Store::Missing) {
                    note(base);
                    continue;
                }
                bind_shape(w, it.ne[0], it.ne[1], base);
                if (it.type == GGML_TYPE_NVFP4 && w.store != Weight::Store::NVFP4) {
                    die(base + " is not NVFP4; " + it.gguf +
                        " can only be the exact copy of an NVFP4 expert (MM-NVFP4 stores G as a side tensor)");
                }
            }
            continue;
        }
        Weight w = resolve_weight(cat, it.hf, it.exact);
        if (w.store == Weight::Store::Missing) {
            if (placeholder && it.placeholder_ok) {
                it.used_placeholder = true;
                continue;
            }
            note(it.exact ? it.hf : it.hf + ".weight");
            continue;
        }
        bind_shape(w, it.ne[0], it.ne[1], it.gguf);
    }
    if (!missing.empty()) {
        std::string msg = "missing tensors (refusing to write a partial GGUF):\n";
        for (const std::string & n : missing) {
            msg += "  " + n + "\n";
        }
        msg += "Layer tensors are never synthesized. --placeholder-missing covers only token_embd, output, and output_norm.";
        die(msg);
    }
}

uint64_t disk_free(const fs::path & path) {
    fs::path dir = path.parent_path();
    if (dir.empty()) {
        dir = ".";
    }
#ifdef _WIN32
    ULARGE_INTEGER avail{};
    if (!GetDiskFreeSpaceExW(dir.c_str(), &avail, nullptr, nullptr)) {
        die("GetDiskFreeSpaceExW failed for " + dir.string());
    }
    return avail.QuadPart;
#else
    struct statvfs st {};
    if (statvfs(dir.c_str(), &st) != 0) {
        die("statvfs failed for " + dir.string());
    }
    return static_cast<uint64_t>(st.f_bavail) * static_cast<uint64_t>(st.f_frsize);
#endif
}

struct Vocab {
    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    std::vector<std::string> merges;
    uint32_t bos = 200019;
    uint32_t eos = 200020;
    std::string chat;
};

std::string encode_merge_piece(const std::string & part) {
    std::string o;
    o.reserve(part.size());
    for (unsigned char c : part) {
        if (c == ' ') {
            // llama.cpp stores a space inside a merge piece as U+0120.
            o.push_back(static_cast<char>(0xC4));
            o.push_back(static_cast<char>(0xA0));
        } else {
            o.push_back(static_cast<char>(c));
        }
    }
    return o;
}

bool token_looks_special(const std::string & token) {
    if (token == "<pad>" || token == "<mask>" || token == "<2mass>" || token == "[@BOS@]") {
        return true;
    }
    if (token.size() >= 4 && token.compare(0, 2, "<|") == 0 && token.compare(token.size() - 2, 2, "|>") == 0) {
        return true;
    }
    if (token.size() >= 8 && token.compare(0, 7, "<unused") == 0 && token.back() == '>') {
        return true;
    }
    return false;
}

bool load_tokenizer_json(const fs::path & path, Vocab & v, int n_vocab) {
    logf("parsing %s\n", path.string().c_str());
    const Json root = parse_json(read_bytes(path));
    const Json * model = root.find("model");
    if (!model) {
        die("tokenizer.json has no model object");
    }
    const Json * vocab = model->find("vocab");
    if (!vocab || vocab->type != Json::Object) {
        die("tokenizer.json has no model.vocab");
    }
    v.tokens.assign(static_cast<size_t>(n_vocab), "");
    v.types.assign(static_cast<size_t>(n_vocab), 5);
    for (int i = 0; i < n_vocab; ++i) {
        v.tokens[static_cast<size_t>(i)] = "[PAD" + std::to_string(i) + "]";
    }
    for (const auto & kv : vocab->obj) {
        const int64_t id = kv.second.as_i();
        if (id < 0 || id >= n_vocab) {
            die("tokenizer vocab id out of range: " + kv.first);
        }
        v.tokens[static_cast<size_t>(id)] = kv.first;
        v.types[static_cast<size_t>(id)] = 1;
    }
    if (const Json * added = root.find("added_tokens")) {
        if (added->type == Json::Array) {
            for (const Json & a : added->arr) {
                const Json * idj = a.find("id");
                const Json * content = a.find("content");
                if (!idj || !content) {
                    continue;
                }
                const int64_t id = idj->as_i();
                if (id < 0 || id >= n_vocab) {
                    die("tokenizer added token id out of range");
                }
                const std::string & text = content->as_s();
                v.tokens[static_cast<size_t>(id)] = text;
                const Json * special = a.find("special");
                const bool is_special =
                    (special && special->type == Json::Bool && special->as_b()) || token_looks_special(text);
                v.types[static_cast<size_t>(id)] = is_special ? 3 : 4;
            }
        }
    }
    v.merges.clear();
    if (const Json * merges = model->find("merges")) {
        if (merges->type == Json::Array && !merges->arr.empty()) {
            if (merges->arr[0].type == Json::String) {
                for (const Json & m : merges->arr) {
                    v.merges.push_back(m.as_s());
                }
            } else if (merges->arr[0].type == Json::Array) {
                for (const Json & pair : merges->arr) {
                    if (pair.arr.size() != 2) {
                        die("tokenizer merge pair is not length 2");
                    }
                    v.merges.push_back(encode_merge_piece(pair.arr[0].as_s()) + " " +
                                       encode_merge_piece(pair.arr[1].as_s()));
                }
            } else {
                die("tokenizer merges have an unknown shape");
            }
        }
    }
    return true;
}

Vocab load_vocab(const fs::path & model, int n_vocab) {
    Vocab v;
    v.tokens.resize(static_cast<size_t>(n_vocab));
    v.types.assign(static_cast<size_t>(n_vocab), 1);
    for (int i = 0; i < n_vocab; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "t%06d", i);
        v.tokens[static_cast<size_t>(i)] = buf;
    }
    bool real_vocab = false;
    const fs::path tokj = model / "tokenizer.json";
    if (fs::is_regular_file(tokj)) {
        real_vocab = load_tokenizer_json(tokj, v, n_vocab);
    }
    const fs::path gen = model / "generation_config.json";
    if (fs::is_regular_file(gen)) {
        const Json g = parse_json(read_bytes(gen));
        if (const Json * p = g.find("bos_token_id")) {
            v.bos = static_cast<uint32_t>(p->as_i());
        }
        if (const Json * p = g.find("eos_token_id")) {
            v.eos = static_cast<uint32_t>(p->as_i());
        }
    }
    const fs::path added = model / "added_tokens.json";
    if (!fs::is_regular_file(added)) {
        die("added_tokens.json is missing; refusing a vocab that cannot spell bos/eos");
    }
    const Json a = parse_json(read_bytes(added));
    for (const auto & kv : a.obj) {
        const int64_t id = kv.second.as_i();
        if (id < 0 || id >= n_vocab) {
            die("added token id out of range: " + kv.first);
        }
        v.tokens[static_cast<size_t>(id)] = kv.first;
        const bool special = static_cast<uint32_t>(id) == v.bos || static_cast<uint32_t>(id) == v.eos;
        v.types[static_cast<size_t>(id)] = special ? 3 : 4;
    }
    if (v.bos < static_cast<uint32_t>(n_vocab)) {
        v.types[v.bos] = 3;
    }
    if (v.eos < static_cast<uint32_t>(n_vocab)) {
        v.types[v.eos] = 3;
    }
    std::unordered_set<std::string> seen;
    seen.reserve(v.tokens.size() * 2);
    for (const std::string & t : v.tokens) {
        if (t.empty() || !seen.insert(t).second) {
            die("duplicate or empty token string \"" + t + "\"");
        }
    }
    const fs::path chat = model / "chat_template.jinja";
    if (fs::is_regular_file(chat)) {
        v.chat = read_bytes(chat);
    }
    if (real_vocab) {
        logf("vocab %d tokens from tokenizer.json, merges %zu, bos %u (%s), eos %u (%s)\n",
             n_vocab, v.merges.size(), v.bos, v.tokens[v.bos].c_str(), v.eos, v.tokens[v.eos].c_str());
    } else {
        logf("vocab %d tokens, bos %u (%s), eos %u (%s), merges empty (tokenizer.json not in the checkpoint yet)\n",
             n_vocab, v.bos, v.tokens[v.bos].c_str(), v.eos, v.tokens[v.eos].c_str());
    }
    return v;
}

bool want_globals(const Options & opt) {
    if (opt.no_globals) {
        return false;
    }
    if (opt.include_globals_set) {
        return opt.include_globals;
    }
    if (opt.split_count >= 2) {
        return opt.split_part == 0;
    }
    return true;
}

std::vector<Item> filter_items(const std::vector<Item> & items, const Options & opt) {
    const bool glob = want_globals(opt);
    std::vector<Item> out;
    out.reserve(items.size());
    for (const Item & it : items) {
        if (it.src < 0) {
            if (glob) {
                out.push_back(it);
            }
            continue;
        }
        if (opt.emit_lo >= 0 && (it.out_i < opt.emit_lo || it.out_i > opt.emit_hi)) {
            continue;
        }
        out.push_back(it);
    }
    return out;
}

fs::path split_output_path(const Options & opt) {
    if (opt.split_count < 2) {
        return opt.out;
    }
    char suf[64];
    std::snprintf(suf, sizeof(suf), "-%05d-of-%05d.gguf", opt.split_part + 1, opt.split_count);
    const std::string s = opt.out.string();
    const size_t n = std::strlen(suf);
    if (s.size() >= n && s.compare(s.size() - n, n, suf) == 0) {
        return opt.out;
    }
    std::string prefix = s;
    if (prefix.size() >= 5 && (prefix.compare(prefix.size() - 5, 5, ".gguf") == 0 ||
                               prefix.compare(prefix.size() - 5, 5, ".GGUF") == 0)) {
        prefix.resize(prefix.size() - 5);
    }
    return fs::path(prefix + suf);
}

std::string hot_list(const Profile & p) {
    std::ostringstream o;
    for (const auto & kv : p.layers) {
        o << kv.first << ":";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            if (i) {
                o << ",";
            }
            o << kv.second[i];
        }
        o << ";";
    }
    return o.str();
}

void write_model(const Options & opt, const Hyp & h, const Recipe & r, const Profile & prof,
                 const std::string & hot_mode, const std::vector<int> & sel, const std::vector<Item> & items_in,
                 const Catalog & cat, const Imp * imp, int threads) {
    std::vector<Item> items = items_in;
    preflight(cat, items, h, opt.placeholder);
    const bool splitting = opt.split_count >= 2;
    const bool write_kv = !splitting || opt.split_part == 0;
    const int meta_blocks = opt.keep_indices ? h.n_layer : static_cast<int>(sel.size());
    const int lead = opt.keep_indices ? h.n_dense : leading_dense(h, sel);
    const fs::path out_path = split_output_path(opt);
    if (splitting && opt.split_tensors < 0) {
        die("internal: split.tensors.count was not set");
    }
    if (splitting && static_cast<int>(items.size()) > opt.split_tensors) {
        die("this split part has more tensors than split.tensors.count");
    }
    uint64_t total = 0;
    bool ph_emb = false, ph_out = false, ph_norm = false;
    for (const Item & it : items) {
        total += it.nbytes;
        if (it.used_placeholder && it.role == Role::Embd) ph_emb = true;
        if (it.used_placeholder && it.role == Role::Out) ph_out = true;
        if (it.used_placeholder && it.role == Role::OutNorm) ph_norm = true;
    }
    if (!out_path.parent_path().empty()) {
        fs::create_directories(out_path.parent_path());
    }
    const uint64_t freeb = disk_free(out_path);
    const uint64_t reserve = 8ull << 30;
    logf("output tensor bytes %s, free %s\n", human(total).c_str(), human(freeb).c_str());
    if (freeb < total + reserve) {
        die("not enough free disk for the GGUF plus an 8 GiB reserve (download is still running)");
    }
    if (r.require_imatrix && !imp) {
        die("this recipe needs --imatrix (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS)");
    }

    Vocab vocab;
    if (write_kv) {
        vocab = load_vocab(opt.model, h.n_vocab);
    }
    struct PartialOut {
        fs::path path;
        bool armed = false;
        ~PartialOut() {
            if (armed) {
                std::error_code ec;
                fs::remove(path, ec);
            }
        }
    } partial;
    partial.path = out_path;
    GgufWriter w;
    w.alignment = 32;
    w.open(out_path);
    partial.armed = true;
    if (write_kv) {
    const std::string arch = "minimax-m3";
    w.add_str("general.architecture", arch);
    w.add_str("general.name", h.name);
    w.add_u32("general.file_type", r.file_type);
    w.add_u32("general.quantization_version", GGML_QNT_VERSION);
    w.add_u32("general.alignment", 32);
    w.add_str("general.source.huggingface.repository", "MiniMaxAI/MiniMax-M3.1-Flash");
    auto key = [&](const std::string & k) { return arch + "." + k; };
    w.add_u32(key("block_count"), static_cast<uint32_t>(meta_blocks));
    w.add_u32(key("context_length"), h.n_ctx);
    w.add_u32(key("embedding_length"), static_cast<uint32_t>(h.n_embd));
    w.add_u32(key("feed_forward_length"), static_cast<uint32_t>(h.n_ff));
    w.add_u32(key("expert_feed_forward_length"), static_cast<uint32_t>(h.n_ff_exp));
    w.add_u32(key("attention.head_count"), static_cast<uint32_t>(h.n_head));
    w.add_u32(key("attention.head_count_kv"), static_cast<uint32_t>(h.n_head_kv));
    w.add_u32(key("attention.key_length"), static_cast<uint32_t>(h.head_dim));
    w.add_u32(key("attention.value_length"), static_cast<uint32_t>(h.head_dim));
    w.add_f32(key("rope.freq_base"), h.rope);
    w.add_u32(key("rope.dimension_count"), static_cast<uint32_t>(h.n_rot));
    w.add_f32(key("attention.layer_norm_rms_epsilon"), h.eps);
    w.add_u32(key("expert_count"), static_cast<uint32_t>(h.n_expert));
    w.add_u32(key("expert_used_count"), static_cast<uint32_t>(h.n_used));
    w.add_u32(key("expert_shared_count"), static_cast<uint32_t>(h.n_shared));
    w.add_f32(key("expert_weights_scale"), h.exp_scale);
    w.add_bool(key("expert_weights_norm"), h.exp_norm);
    w.add_u32(key("expert_gating_func"), 2);
    w.add_u32(key("leading_dense_block_count"), static_cast<uint32_t>(lead));
    w.add_u32(key("attention.indexer.head_count"), static_cast<uint32_t>(h.idx_heads));
    w.add_u32(key("attention.indexer.key_length"), static_cast<uint32_t>(h.idx_key));
    w.add_u32(key("attention.indexer.top_k"), static_cast<uint32_t>(h.idx_topk));
    w.add_u32(key("attention.indexer.block_size"), static_cast<uint32_t>(h.idx_block));
    w.add_u32(key("attention.indexer.local_blocks"), static_cast<uint32_t>(h.idx_local));
    w.add_u32(key("attention.indexer.first_layer"), static_cast<uint32_t>(h.idx_first));
    w.add_f32(key("swiglu_alpha"), h.swiglu_a);
    w.add_f32(key("swiglu_limit"), h.swiglu_l);
    w.add_str("strata.recipe", recipe_meta(r, hot_mode));
    w.add_str("strata.tool", "strata-mm-quant");
    w.add_str("strata.expert_mix",
              "GGUF stores one ggml type per tensor, so a per-expert mix is not representable. "
              "dynamic_down steps the whole ffn_down_exps tensor. promote-layer steps one layer's "
              "ffn_gate_exps and ffn_up_exps. The hot-expert id list is recorded either way.");
    w.add_str("strata.hot_expert_mode", hot_mode);
    w.add_str("strata.hot_experts", hot_list(prof));
    if (!prof.raw.empty()) {
        w.add_str("strata.expert_profile", prof.raw);
    }
    w.add_str("strata.nvfp4",
              "MM-NVFP4 experts are GGML_TYPE_NVFP4 (40) bytes: nibbles remapped to the ggml block layout and "
              "the checkpoint's E4M3 scale bytes copied verbatim. block_nvfp4 has no global-scale field, so the "
              "per-tensor divisor weight_global_scale is stored beside each expert tensor as "
              "blk.N.ffn_{gate,up,down}_exps.scale, f32 [n_expert], expert order. Dequant is "
              "e2m1 * e4m3(scale) / scale[e], bit-for-bit equal to spec A. Not readable by llama.cpp or ik_llama: "
              "both dequantize NVFP4 without the side tensor (and neither knows this key). Strata engine only.");
    w.add_str("strata.expert_global_scale",
              "blk.N.ffn_{gate,up,down}_exps.scale = f32 [n_expert]: weight_global_scale of that expert, a DIVISOR "
              "(spec A.3). Applied exactly once, after the dot product.");
    w.add_str("strata.indexer_quant",
              "Indexer projections use the recipe indexer type (default Q8_0). "
              "llama.cpp's converter forces .indexer. to F32.");
    std::vector<int32_t> srcs;
    for (int s : sel) {
        srcs.push_back(s);
    }
    w.add_i32_array("strata.source_layers", srcs);
    w.add_bool("strata.placeholder_embed", ph_emb);
    w.add_bool("strata.placeholder_output", ph_out);
    w.add_bool("strata.placeholder_output_norm", ph_norm);
    w.add_f32("strata.placeholder_value", opt.placeholder_value);
    w.add_str("tokenizer.ggml.model", "gpt2");
    w.add_str("tokenizer.ggml.pre", "minimax-m2");
    w.add_str_array("tokenizer.ggml.tokens", vocab.tokens);
    w.add_i32_array("tokenizer.ggml.token_type", vocab.types);
    w.add_str_array("tokenizer.ggml.merges", vocab.merges);
    w.add_u32("tokenizer.ggml.bos_token_id", vocab.bos);
    w.add_u32("tokenizer.ggml.eos_token_id", vocab.eos);
    w.add_bool("tokenizer.ggml.add_bos_token", false);
    if (!vocab.chat.empty()) {
        w.add_str("tokenizer.chat_template", vocab.chat);
    }
    }
    if (splitting) {
        if (opt.split_part < 0 || opt.split_part > 65535 || opt.split_count < 2 || opt.split_count > 65535) {
            die("--split-part and --split-count must fit in uint16");
        }
        w.add_u16("split.no", static_cast<uint16_t>(opt.split_part));
        w.add_u16("split.count", static_cast<uint16_t>(opt.split_count));
        w.add_i32("split.tensors.count", opt.split_tensors);
    }
    for (const Item & it : items) {
        TensorDesc td;
        td.name = it.gguf;
        std::memcpy(td.ne, it.ne, sizeof(td.ne));
        td.n_dims = gguf_n_dims(td.ne);
        td.type = static_cast<uint32_t>(it.type);
        td.nbytes = it.nbytes;
        w.add_tensor(std::move(td));
    }
    w.write_header();

    const auto t_all = std::chrono::steady_clock::now();
    uint64_t written = 0;
    bool logged_g = false;
    for (const Item & it : items) {
        const auto t0 = std::chrono::steady_clock::now();
        if (it.used_placeholder) {
            const float value = it.role == Role::OutNorm ? 1.f : opt.placeholder_value;
            const float * im = importance_for(imp, it, 0);
            require_importance(it, im);
            const int64_t nrows = it.ne[1] * it.ne[2];
            emit_constant(w, it.type, it.ne[0], nrows, value, im);
        } else if (it.side_scale) {
            emit_side_scale(w, cat, it, h);
        } else if (is_expert(it)) {
            for (int e = 0; e < h.n_expert; ++e) {
                const std::string base = expert_base(it, e);
                Weight wt = resolve_weight(cat, base, false);
                bind_shape(wt, it.ne[0], it.ne[1], base);
                if (!logged_g && wt.store == Weight::Store::NVFP4) {
                    logf("weight_global_scale %s = %.9g\n", base.c_str(), wt.G);
                    logged_g = true;
                }
                const float * im = importance_for(imp, it, e);
                require_importance(it, im);
                emit_weight(w, wt, it.type, it.ne[0], it.ne[1], im, threads, 0.f);
            }
        } else {
            Weight wt = resolve_weight(cat, it.hf, it.exact);
            bind_shape(wt, it.ne[0], it.ne[1], it.gguf);
            const float * im = importance_for(imp, it, 0);
            require_importance(it, im);
            emit_weight(w, wt, it.type, it.ne[0], it.ne[1], im, threads, it.add);
        }
        w.end_tensor();
        written += it.nbytes;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        logf("  %-32s %-8s %6.1fs  %s\n", it.gguf.c_str(), type_name(it.type), sec, human(it.nbytes).c_str());
    }
    w.close();
    partial.armed = false;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    logf("wrote %s  %s in %.1fs  (%.2f GB/s of output)\n", out_path.string().c_str(), human(written).c_str(), sec,
         sec > 0 ? (static_cast<double>(written) / 1e9) / sec : 0.0);
    logf("block_count %d, leading_dense %d, indexer.first_layer %d, file_type %u\n",
         meta_blocks, lead, h.idx_first, r.file_type);
    if (splitting) {
        logf("split.no %d  split.count %d  split.tensors.count %d  tensors in this file %zu\n",
             opt.split_part, opt.split_count, opt.split_tensors, items.size());
    }
    if (ph_emb || ph_out || ph_norm) {
        logf("placeholders: embed=%d output=%d output_norm=%d (embed/output value %.4f, output_norm is 1)\n",
             ph_emb ? 1 : 0, ph_out ? 1 : 0, ph_norm ? 1 : 0, opt.placeholder_value);
    }
}

// ---------------------------------------------------------------------------

void bench(const Catalog & cat, const Hyp & h, const Recipe & r, const std::vector<int> & sel,
           const Imp * imp, const Options & opt, int threads) {
    int layer = -1;
    for (int src : sel) {
        if (!h.is_moe(src)) {
            continue;
        }
        if (cat.find(layer_base(src) + ".block_sparse_moe.experts.0.w1.weight_packed")) {
            layer = src;
            break;
        }
    }
    if (layer < 0) {
        die("bench: no MoE expert weights in the selected layers");
    }
    int out_i = 0;
    for (int i = 0; i < static_cast<int>(sel.size()); ++i) {
        if (sel[static_cast<size_t>(i)] == layer) {
            out_i = i;
        }
    }
    const int nexp = std::max(1, opt.bench_experts);
    const int64_t n_in = h.n_embd;
    const int64_t n_out = h.n_ff_exp;
    const std::string gguf_name = blk_name(out_i, "ffn_gate_exps");
    std::vector<float> uniform;
    const float * im0 = nullptr;
    if (imp) {
        Item fake;
        fake.gguf = gguf_name;
        fake.ne[0] = n_in;
        fake.ne[2] = h.n_expert;
        im0 = importance_for(imp, fake, 0);
    }
    if (!im0 && type_needs_imatrix(r.experts)) {
        uniform.assign(static_cast<size_t>(n_in), 1.f);
        im0 = uniform.data();
        logf("WARNING: bench synthesized a uniform importance row. %s\n", kTestNote);
    }
    if (type_needs_imatrix(r.experts) && !im0) {
        die("bench needs an imatrix for " + std::string(type_name(r.experts)));
    }
    std::vector<float> f32(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
    const size_t row_q = row_size(r.experts, n_in);
    std::vector<uint8_t> q(row_q * static_cast<size_t>(n_out));
    int done = 0;
    uint64_t logical = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int e = 0; e < h.n_expert && done < nexp; ++e) {
        const std::string base = layer_base(layer) + ".block_sparse_moe.experts." + std::to_string(e) + ".w1";
        Weight wt = resolve_weight(cat, base, false);
        if (wt.store == Weight::Store::Missing) {
            break;
        }
        bind_shape(wt, n_in, n_out, base);
        const float * im = im0;
        if (imp) {
            Item fake;
            fake.gguf = gguf_name;
            fake.ne[0] = n_in;
            fake.ne[2] = h.n_expert;
            if (const float * slice = importance_for(imp, fake, e)) {
                im = slice;
            }
        }
        dequant_rows(wt, 0, n_out, n_in, f32.data(), 0.f, threads);
        quantize_rows(r.experts, f32.data(), q.data(), n_out, n_in, im, threads);
        logical += static_cast<uint64_t>(n_out) * static_cast<uint64_t>(n_in) * 4;
        ++done;
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (done == 0 || sec <= 0) {
        die("bench produced no samples");
    }
    const double bps = static_cast<double>(logical) / sec;
    logf("bench layer %d -> blk.%d  %s  experts %d  logical f32 %s  %.2fs  %.3f GB/s\n",
         layer, out_i, type_name(r.experts), done, human(logical).c_str(), sec, bps / 1e9);
    const double n_moe = static_cast<double>(h.n_layer - h.n_dense);
    const double expert_params = n_moe * h.n_expert * 3.0 * h.n_ff_exp * h.n_embd;
    const double expert_f32 = expert_params * 4.0;
    const double expert_sec = expert_f32 / bps;
    logf("projected full routed-expert quant: %.0f params, logical f32 %.3f TB, %.1f min at this rate\n",
         expert_params, expert_f32 / 1e12, expert_sec / 60.0);
    logf("That projection is the expert matrices only. Q8_0 attention and dense tensors are faster, so a full-file estimate that uses this rate for every tensor is an upper bound.\n");

    const std::string qbase = layer_base(layer) + ".self_attn.q_proj";
    Weight qwt = resolve_weight(cat, qbase, false);
    if (qwt.store != Weight::Store::Missing) {
        const int64_t q_out = static_cast<int64_t>(h.n_head) * h.head_dim;
        bind_shape(qwt, h.n_embd, q_out, qbase);
        std::vector<float> qf(static_cast<size_t>(q_out) * static_cast<size_t>(h.n_embd));
        std::vector<uint8_t> qq(ggml_row_size(GGML_TYPE_Q8_0, h.n_embd) * static_cast<size_t>(q_out));
        const auto a0 = std::chrono::steady_clock::now();
        dequant_rows(qwt, 0, q_out, h.n_embd, qf.data(), 0.f, threads);
        quantize_rows(GGML_TYPE_Q8_0, qf.data(), qq.data(), q_out, h.n_embd, nullptr, threads);
        const double asec = std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
        const double alogical = static_cast<double>(q_out) * h.n_embd * 4.0;
        logf("bench attn q_proj MXFP8->Q8_0  %.2fs  %.3f GB/s logical f32\n", asec, asec > 0 ? (alogical / 1e9) / asec : 0);
    }
}

// ---------------------------------------------------------------------------
// Experiment: one MoE layer, every expert, measured error per matrix kind.

struct NpyArray {
    std::string descr;
    std::vector<int64_t> shape;
    std::vector<uint8_t> raw;
};

NpyArray read_npy(const fs::path & path) {
    FILE * f = std::fopen(path.string().c_str(), "rb");
    if (!f) {
        die("cannot open " + path.string());
    }
    unsigned char magic[8] = {0};
    if (std::fread(magic, 1, 8, f) != 8 || magic[0] != 0x93 || std::memcmp(magic + 1, "NUMPY", 5) != 0) {
        std::fclose(f);
        die("not an npy file: " + path.string());
    }
    const int major = magic[6];
    if (major != 1 && major != 2 && major != 3) {
        std::fclose(f);
        die("unsupported npy version in " + path.string());
    }
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        if (std::fread(&h16, 2, 1, f) != 1) {
            std::fclose(f);
            die("short npy header");
        }
        hlen = h16;
    } else if (std::fread(&hlen, 4, 1, f) != 1) {
        std::fclose(f);
        die("short npy header");
    }
    std::string h(static_cast<size_t>(hlen), '\0');
    if (std::fread(h.data(), 1, hlen, f) != hlen) {
        std::fclose(f);
        die("short npy header");
    }
    NpyArray a;
    const size_t dk = h.find("'descr':");
    const size_t q1 = dk == std::string::npos ? std::string::npos : h.find('\'', dk + 8);
    const size_t q2 = q1 == std::string::npos ? std::string::npos : h.find('\'', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) {
        std::fclose(f);
        die("unreadable npy header in " + path.string());
    }
    a.descr = h.substr(q1 + 1, q2 - q1 - 1);
    if (h.find("'fortran_order': True") != std::string::npos) {
        std::fclose(f);
        die("npy file is Fortran-ordered; expected C order: " + path.string());
    }
    const size_t op = h.find('(', q2);
    const size_t cp = op == std::string::npos ? std::string::npos : h.find(')', op);
    if (op == std::string::npos || cp == std::string::npos) {
        std::fclose(f);
        die("npy header has no shape: " + path.string());
    }
    std::string dims = h.substr(op + 1, cp - op - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) {
            ++pos;
        }
        if (pos >= dims.size()) {
            break;
        }
        char * end = nullptr;
        const long long v = std::strtoll(dims.c_str() + pos, &end, 10);
        if (end == dims.c_str() + pos) {
            break;
        }
        a.shape.push_back(static_cast<int64_t>(v));
        pos = static_cast<size_t>(end - dims.c_str());
    }
    int64_t n = 1;
    for (int64_t d : a.shape) {
        n *= d;
    }
    int64_t itemsize = 1;
    if (a.descr.size() >= 3) {
        const char k = a.descr[1];
        if (k == 'f') {
            itemsize = 4;
        } else if (k == 'i' || k == 'u') {
            itemsize = 8;
        } else if (k == 'b') {
            itemsize = 1;
        }
    }
    a.raw.resize(static_cast<size_t>(n) * static_cast<size_t>(itemsize));
    if (std::fread(a.raw.data(), 1, a.raw.size(), f) != a.raw.size()) {
        std::fclose(f);
        die("short npy payload in " + path.string());
    }
    std::fclose(f);
    return a;
}

std::vector<float> npy_f32(const fs::path & path, int64_t & n0, int64_t & n1) {
    NpyArray a = read_npy(path);
    if (a.descr != "<f4" && a.descr != "|f4" && a.descr != "=f4") {
        die("expected float32 npy: " + path.string() + " has " + a.descr);
    }
    n0 = a.shape.empty() ? 0 : a.shape[0];
    n1 = a.shape.size() > 1 ? a.shape[1] : 1;
    if (static_cast<int64_t>(a.raw.size()) != n0 * n1 * 4) {
        die("float32 npy shape does not match payload: " + path.string());
    }
    std::vector<float> out(a.raw.size() / 4);
    std::memcpy(out.data(), a.raw.data(), out.size() * sizeof(float));
    return out;
}

std::vector<int64_t> npy_i64(const fs::path & path, int64_t & n) {
    NpyArray a = read_npy(path);
    if (a.descr != "<i8" && a.descr != "|i8" && a.descr != "=i8") {
        die("expected int64 npy: " + path.string() + " has " + a.descr);
    }
    std::vector<int64_t> out(a.raw.size() / 8);
    std::memcpy(out.data(), a.raw.data(), out.size() * sizeof(int64_t));
    n = static_cast<int64_t>(out.size());
    return out;
}

struct KindResult {
    std::string name;
    std::string tensor;
    int64_t experts = 0;
    double plain_num = 0, plain_den = 0;
    double plain_mean = 0;      // mean of per-expert relative Frobenius
    double weighted_mean = 0;   // mean of per-row weighted ratio
    double weighted_num = 0, weighted_den = 0;
    double out_num = 0, out_den = 0;
    int64_t out_tokens = 0;
    int64_t out_experts = 0;
    double quant_sec = 0;
};

// ||W x - Q x|| / ||W x|| accumulated over the given input rows.
void matvec_error(const float * w, const float * qw, int64_t n_out, int64_t n_in,
                  const std::vector<const float *> & rows, int threads, double & num, double & den) {
    if (threads < 1) {
        threads = 1;
    }
    if (static_cast<int64_t>(threads) > n_out) {
        threads = static_cast<int>(n_out);
    }
    std::vector<double> pn(static_cast<size_t>(threads), 0.0);
    std::vector<double> pd(static_cast<size_t>(threads), 0.0);
    auto worker = [&](int t, int64_t a, int64_t b) {
        double ln = 0.0, ld = 0.0;
        for (int64_t r = a; r < b; ++r) {
            const float * wr = w + r * n_in;
            const float * qr = qw + r * n_in;
            for (const float * x : rows) {
                float y = 0.f, yq = 0.f;
                for (int64_t i = 0; i < n_in; ++i) {
                    y += wr[i] * x[i];
                    yq += qr[i] * x[i];
                }
                const double d = static_cast<double>(y) - static_cast<double>(yq);
                ln += d * d;
                ld += static_cast<double>(y) * static_cast<double>(y);
            }
        }
        pn[static_cast<size_t>(t)] = ln;
        pd[static_cast<size_t>(t)] = ld;
    };
    if (threads <= 1) {
        worker(0, 0, n_out);
    } else {
        const int64_t chunk = (n_out + threads - 1) / threads;
        std::vector<std::thread> ws;
        for (int t = 0; t < threads; ++t) {
            const int64_t a = static_cast<int64_t>(t) * chunk;
            if (a >= n_out) {
                break;
            }
            ws.emplace_back(worker, t, a, std::min(n_out, a + chunk));
        }
        for (std::thread & th : ws) {
            th.join();
        }
    }
    for (int t = 0; t < threads; ++t) {
        num += pn[static_cast<size_t>(t)];
        den += pd[static_cast<size_t>(t)];
    }
}

enum class ExpMethod { Old, Diag, Gptq };

ExpMethod parse_exp_method(const std::string & s) {
    if (s == "old") {
        return ExpMethod::Old;
    }
    if (s == "diag") {
        return ExpMethod::Diag;
    }
    if (s == "gptq") {
        return ExpMethod::Gptq;
    }
    die("--exp-method must be old, diag or gptq (got '" + s + "')");
    return ExpMethod::Old;
}

/// H += sum_rows x x^T, lower triangle only, tiled: a 128x128 tile is accumulated over every row
/// before it is written back, so the matrix is streamed once per tile instead of once per row (the
/// row-at-a-time form moved ~130 GB for 431 rows on layer 40 and took 30 s).
void add_outer(float * h, int64_t n, const std::vector<const float *> & rows, int threads) {
    if (rows.empty()) {
        return;
    }
    constexpr int64_t kT = 128;
    const int64_t nt = (n + kT - 1) / kT;
    std::vector<std::pair<int64_t, int64_t>> tiles;
    tiles.reserve(static_cast<size_t>(nt * (nt + 1) / 2));
    for (int64_t ti = 0; ti < nt; ++ti) {
        for (int64_t tj = 0; tj <= ti; ++tj) {
            tiles.emplace_back(ti, tj);
        }
    }
    std::vector<double> acc(static_cast<size_t>(kT) * static_cast<size_t>(kT));
    gptq::parallel_for(static_cast<int64_t>(tiles.size()), threads, [&](int64_t t0, int64_t t1) {
        std::vector<double> a(static_cast<size_t>(kT) * static_cast<size_t>(kT));
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t ti = tiles[static_cast<size_t>(t)].first;
            const int64_t tj = tiles[static_cast<size_t>(t)].second;
            const int64_t i0 = ti * kT, i1 = std::min(n, i0 + kT);
            const int64_t j0 = tj * kT, j1 = std::min(n, j0 + kT);
            const int64_t ni = i1 - i0, nj = j1 - j0;
            std::fill(a.begin(), a.end(), 0.0);
            for (const float * x : rows) {
                for (int64_t i = 0; i < ni; ++i) {
                    const double xi = x[i0 + i];
                    double * ar = a.data() + i * nj;
                    for (int64_t j = 0; j < nj; ++j) {
                        ar[j] += xi * static_cast<double>(x[j0 + j]);
                    }
                }
            }
            for (int64_t i = 0; i < ni; ++i) {
                float * hr = h + (i0 + i) * n;
                const double * ar = a.data() + i * nj;
                for (int64_t j = 0; j < nj; ++j) {
                    if (j0 + j <= i0 + i) {   // lower triangle only
                        hr[j0 + j] += static_cast<float>(ar[j]);
                    }
                }
            }
        }
    });
    (void) acc;
}

/// H += scale * src (n x n), for the layer-wide prior.
void add_scaled(float * h, const float * src, int64_t n, double scale, int threads) {
    gptq::parallel_for(n, threads, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                h[i * n + j] += static_cast<float>(scale * static_cast<double>(src[i * n + j]));
            }
        }
    });
}

/// The expert's rows from the dumps, split deterministically into fit and held-out rows:
/// every `stride`-th row (stride = round(1/holdout), 0 = no split) is held out, so the
/// split is spread over the sequence rather than being its tail.
void split_holdout(const std::vector<int64_t> & all, double holdout, std::vector<int64_t> & fit,
                   std::vector<int64_t> & held) {
    fit.clear();
    held.clear();
    int stride = 0;
    if (holdout > 0.0) {
        stride = static_cast<int>(std::lround(1.0 / holdout));
        if (stride < 2) {
            stride = 2;
        }
    }
    for (size_t i = 0; i < all.size(); ++i) {
        if (stride > 0 && static_cast<int>(i) % stride == 0) {
            held.push_back(all[i]);
        } else {
            fit.push_back(all[i]);
        }
    }
}

/// A 1-D importance vector from rows: mean x^2 per column.
void diag_importance(const std::vector<const float *> & rows, int64_t n, std::vector<float> & out) {
    out.assign(static_cast<size_t>(n), 1.f);
    if (rows.empty()) {
        return;
    }
    std::vector<double> ss(static_cast<size_t>(n), 0.0);
    for (const float * x : rows) {
        for (int64_t i = 0; i < n; ++i) {
            ss[static_cast<size_t>(i)] += static_cast<double>(x[i]) * x[i];
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = static_cast<float>(ss[static_cast<size_t>(i)] / static_cast<double>(rows.size()));
    }
}

void experiment_selftest(int n, int threads) {
    std::vector<float> a(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<float> ref(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<float> g(static_cast<size_t>(n) * static_cast<size_t>(n), 0.f);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const float u = static_cast<float>(std::sin(0.37 * (i + 1) * (j + 2) + 0.11 * i));
            g[i * n + j] = u;
        }
    }
    // A = G G^T / n + I (symmetric positive definite, a mild condition number)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                s += static_cast<double>(g[i * n + k]) * g[j * n + k];
            }
            ref[i * n + j] = static_cast<float>(s / n);
            a[i * n + j] = ref[i * n + j] + (i == j ? 1.f : 0.f);
        }
    }
    std::vector<float> l = a;
    gptq::chol_lower(l.data(), n, threads);
    // check L L^T == A
    double worst = 0.0, scale = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k <= j; ++k) {
                s += static_cast<double>(l[i * n + k]) * l[j * n + k];
            }
            worst = std::max(worst, std::fabs(s - static_cast<double>(a[i * n + j])));
            scale = std::max(scale, std::fabs(static_cast<double>(a[i * n + j])));
        }
    }
    logf("selftest chol: max |L L^T - A| / max|A| = %.3e\n", worst / scale);
    std::vector<float> l2 = l;
    gptq::tri_inv_lower(l.data(), n, l2.data(), threads);
    // check L * L^-1 == I
    worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int64_t k = j; k <= i; ++k) {
                s += static_cast<double>(l[i * n + k]) * l2[k * n + j];
            }
            worst = std::max(worst, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
    }
    logf("selftest inverse: max |L L^-1 - I| = %.3e\n", worst);
    // the block importance must be diag(inv(Hinv_SS)) = diag(inv(inv(H_SS))), i.e. diag(H_SS).
    // Hinv = M^T M, so inv(Hinv_SS) = (M_SS M_SS^T)^-1 and its diagonal is what
    // block_importance returns. Check against a naive Gauss-Jordan of that block.
    const int64_t nb = 32;
    std::vector<double> w;
    std::vector<float> imp;
    gptq::block_factors(l2.data(), n, 0, nb, w, imp);
    std::vector<double> hss(static_cast<size_t>(nb) * static_cast<size_t>(nb), 0.0);
    for (int64_t i = 0; i < nb; ++i) {
        for (int64_t j = 0; j < nb; ++j) {
            double s = 0.0;
            for (int64_t k = 0; k < nb; ++k) {
                s += static_cast<double>(l2[i * n + k]) * l2[j * n + k];
            }
            hss[i * nb + j] = s;   // Hinv_SS = M_SS M_SS^T
        }
    }
    // invert hss with Gauss-Jordan
    std::vector<double> aug(static_cast<size_t>(nb) * static_cast<size_t>(2 * nb), 0.0);
    for (int64_t i = 0; i < nb; ++i) {
        for (int64_t j = 0; j < nb; ++j) {
            aug[i * 2 * nb + j] = hss[i * nb + j];
        }
        aug[i * 2 * nb + nb + i] = 1.0;
    }
    for (int64_t c = 0; c < nb; ++c) {
        const double d = aug[c * 2 * nb + c];
        for (int64_t j = 0; j < 2 * nb; ++j) {
            aug[c * 2 * nb + j] /= d;
        }
        for (int64_t r = 0; r < nb; ++r) {
            if (r == c) {
                continue;
            }
            const double f = aug[r * 2 * nb + c];
            for (int64_t j = 0; j < 2 * nb; ++j) {
                aug[r * 2 * nb + j] -= f * aug[c * 2 * nb + j];
            }
        }
    }
    double mean = 0.0;
    for (int64_t i = 0; i < nb; ++i) {
        mean += aug[i * 2 * nb + nb + i];
    }
    mean /= static_cast<double>(nb);
    double worst_imp = 0.0, ref_imp = 0.0;
    for (int64_t i = 0; i < nb; ++i) {
        const double want = aug[i * 2 * nb + nb + i] / mean;
        const double got = static_cast<double>(imp[static_cast<size_t>(i)]);
        worst_imp = std::max(worst_imp, std::fabs(got - want));
        ref_imp = std::max(ref_imp, want);
    }
    logf("selftest block importance: max |imp - diag(inv(inv(H_SS)))| / max = %.3e\n", worst_imp / ref_imp);
    logf("selftest done\n");
}

/// Assemble a layer pack from --exp-pack-frag's one-file-per-(kind, expert) directory. The
/// fragments are the exact bytes the experiment measured, so a pass split across several PC
/// turns (each with its own --exp-start) produces the same file as one uninterrupted run.
void merge_layer_pack(const Options & opt, const Hyp & h) {
    const ggml_type type = parse_type(opt.exp_type);
    const int nexp = h.n_expert;
    const int layer = opt.exp_layer;
    const char * kinds[3] = {"gate", "up", "down"};
    const char * leaves[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    const int64_t n_in[3] = {h.n_embd, h.n_embd, h.n_ff_exp};
    const int64_t n_out[3] = {h.n_ff_exp, h.n_ff_exp, h.n_embd};
    GgufWriter w;
    w.alignment = 32;
    w.open(opt.exp_pack);
    w.add_str("general.architecture", "minimax-m3");
    w.add_str("general.name", "MiniMax-M3.1-Flash layer " + std::to_string(layer) + " " + opt.exp_method + " " +
                                   type_name(type));
    w.add_str("general.baseline", "sparsa minimax gptq-iq experiment");
    w.add_u32("general.alignment", 32);
    w.add_u32("minimax-m3.block_count", static_cast<uint32_t>(h.n_layer));
    w.add_u32("minimax-m3.expert_count", static_cast<uint32_t>(nexp));
    w.add_str("minimax.quant.method", opt.exp_method);
    w.add_f32("minimax.quant.gptq_prior", static_cast<float>(opt.gptq_prior));
    w.add_f32("minimax.quant.gptq_damp", static_cast<float>(opt.gptq_damp));
    w.add_f32("minimax.quant.holdout", static_cast<float>(opt.exp_holdout));
    for (int k = 0; k < 3; ++k) {
        TensorDesc t;
        t.name = blk_name(layer, leaves[k]);
        t.n_dims = 3;
        t.ne[0] = n_in[k];
        t.ne[1] = n_out[k];
        t.ne[2] = nexp;
        t.type = static_cast<uint32_t>(type);
        t.nbytes = row_size(type, n_in[k]) * static_cast<uint64_t>(n_out[k]) * static_cast<uint64_t>(nexp);
        w.add_tensor(std::move(t));
    }
    w.write_header();
    std::vector<uint8_t> buf;
    for (int k = 0; k < 3; ++k) {
        const size_t want = row_size(type, n_in[k]) * static_cast<size_t>(n_out[k]);
        for (int e = 0; e < nexp; ++e) {
            char fb[96];
            std::snprintf(fb, sizeof(fb), "%s-%03d.bin", kinds[k], e);
            const fs::path fp = opt.pack_merge / fb;
            FILE * f = std::fopen(fp.string().c_str(), "rb");
            if (!f) {
                die("merge: missing " + fp.string());
            }
            buf.resize(want);
            const size_t got = std::fread(buf.data(), 1, want, f);
            std::fclose(f);
            if (got != want) {
                die("merge: " + fp.string() + " holds " + std::to_string(got) + " bytes, expected " +
                    std::to_string(want));
            }
            w.write_payload(buf.data(), got);
        }
        w.end_tensor();
    }
    w.close();
    logf("merged layer pack %s (%d experts, %s)\n", opt.exp_pack.string().c_str(), nexp, type_name(type));
}

void experiment(const Catalog & cat, const Hyp & h, const Recipe & r, const Options & opt,
                const Imp * imp, int threads) {
    const ExpMethod method = parse_exp_method(opt.exp_method);
    if (opt.exp_layer < 0 || opt.exp_layer >= h.n_layer) {
        die("--exp-layer is outside 0.." + std::to_string(h.n_layer - 1));
    }
    if (!h.is_moe(opt.exp_layer)) {
        die("layer " + std::to_string(opt.exp_layer) + " has no routed experts");
    }
    const ggml_type type = opt.exp_type.empty() ? r.experts : parse_type(opt.exp_type);
    const int nexp = opt.exp_experts > 0 ? std::min(opt.exp_experts, h.n_expert) : h.n_expert;
    const int estart = std::max(0, std::min(opt.exp_start, nexp));
    const std::string lb = layer_base(opt.exp_layer);
    std::vector<const float *> xrows;
    std::vector<float> xdata;
    if (!opt.exp_x.empty()) {
        int64_t nx = 0, ncol = 0;
        xdata = npy_f32(opt.exp_x, nx, ncol);
        if (ncol != h.n_embd) {
            die("--exp-x has " + std::to_string(ncol) + " columns, expected " + std::to_string(h.n_embd));
        }
        for (int64_t t = 0; t < nx; ++t) {
            xrows.push_back(xdata.data() + t * ncol);
        }
        logf("experiment activations: %lld rows x %lld from %s\n", static_cast<long long>(nx),
             static_cast<long long>(ncol), opt.exp_x.string().c_str());
    }
    if (method != ExpMethod::Old && xrows.empty()) {
        die("--exp-method " + opt.exp_method + " needs --exp-x and --exp-dumps");
    }
    // The split is on tokens, not per expert, so the layer-wide prior cannot see a held-out token
    // through some other expert's rows: every `1/holdout`-th row of --exp-x is held out.
    std::vector<char> held_token(xrows.size(), 0);
    {
        std::vector<int64_t> all(xrows.size());
        for (size_t i = 0; i < all.size(); ++i) {
            all[i] = static_cast<int64_t>(i);
        }
        std::vector<int64_t> fit, held;
        split_holdout(all, opt.exp_holdout, fit, held);
        for (int64_t t : held) {
            held_token[static_cast<size_t>(t)] = 1;
        }
    }
    if (imp && imp->test_only) {
        logf("WARNING: experiment uses the TEST-ONLY uniform imatrix\n");
    }
    logf("experiment method %s, type %s, layer %d, %d experts, holdout %.2f, prior %.0f, damp %.3f\n",
         opt.exp_method.c_str(), type_name(type), opt.exp_layer, nexp, opt.exp_holdout, opt.gptq_prior,
         opt.gptq_damp);
    // The layer-wide input Hessian, the prior of a rarely routed expert: mean over every row.
    std::vector<float> hbar;
    if (method == ExpMethod::Gptq && !xrows.empty() && opt.gptq_prior > 0.0) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<const float *> prior_rows;
        prior_rows.reserve(xrows.size());
        for (size_t i = 0; i < xrows.size(); ++i) {
            if (!held_token[i]) {
                prior_rows.push_back(xrows[i]);
            }
        }
        if (prior_rows.empty()) {
            prior_rows = xrows;
        }
        hbar.assign(static_cast<size_t>(h.n_embd) * static_cast<size_t>(h.n_embd), 0.f);
        add_outer(hbar.data(), h.n_embd, prior_rows, threads);
        const float inv = 1.f / static_cast<float>(prior_rows.size());
        for (float & v : hbar) {
            v *= inv;
        }
        logf("experiment layer Hessian (prior) built in %.1f s\n",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const char * kinds[3] = {"gate", "up", "down"};
    const char * leaves[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    const char * wkeys[3] = {"w1", "w3", "w2"};
    const bool is_down[3] = {false, false, true};
    std::vector<KindResult> results;
    std::vector<std::vector<uint8_t>> pack_bytes[3];
    // One expert's gate+up+down costs a 6144x6144 Cholesky and two triangular solves (about 4e10
    // flops) before the quantizer runs, so the experts are the unit of parallelism in GPTQ mode:
    // `workers` experts at a time, each with threads/workers of its own.
    const int workers = method == ExpMethod::Gptq ? std::max(1, std::min<int>(8, threads / 6)) : 1;
    const int wthreads = std::max(1, threads / workers);
    if (workers > 1) {
        logf("experiment: %d experts in parallel, %d threads each\n", workers, wthreads);
    }
    struct ExpStat {
        double plain_num = 0, plain_den = 0, plain_mean = 0;
        double wnum = 0, wden = 0, wmean = 0;
        double onum = 0, oden = 0;
        int64_t otok = 0, oexp = 0;
        double qsec = 0;
    };
    for (int k = 0; k < 3; ++k) {
        KindResult res;
        res.name = kinds[k];
        res.tensor = blk_name(opt.exp_layer, leaves[k]);
        const int64_t n_in = k == 2 ? h.n_ff_exp : h.n_embd;
        const int64_t n_out = k == 2 ? h.n_embd : h.n_ff_exp;
        const size_t row_q = row_size(type, n_in);
        const ggml_type_traits * traits = is_ikq(type) ? nullptr : ggml_get_type_traits(type);
        if (!is_ikq(type) && (!traits || !traits->to_float)) {
            die("no dequantizer for " + std::string(type_name(type)));
        }
        std::vector<ExpStat> st(static_cast<size_t>(nexp));
        if (!opt.exp_pack.empty()) {
            pack_bytes[k].resize(static_cast<size_t>(nexp));
        }
        auto one_expert = [&](int e) {
            const std::string base = lb + ".block_sparse_moe.experts." + std::to_string(e) + "." + wkeys[k];
            Weight w = resolve_weight(cat, base, false);
            if (w.store == Weight::Store::Missing) {
                die("experiment: missing " + base);
            }
            bind_shape(w, n_in, n_out, base);
            // this expert's rows: gate/up from the routed token list, down from the dumped hidden states
            std::vector<const float *> fit_rows, held_rows;
            std::vector<float> hidden;   // the down projection's inputs (kept alive while rows point into it)
            if (!is_down[k]) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "expert_%03d_token_idx.npy", e);
                const fs::path tp = opt.exp_dumps / buf;
                if (xrows.empty() || !fs::is_regular_file(tp)) {
                    die(std::string("experiment: --exp-x and ") + tp.string() + " are both needed for " + kinds[k]);
                }
                int64_t nt = 0;
                const std::vector<int64_t> idx = npy_i64(tp, nt);
                for (int64_t t : idx) {
                    if (t < 0 || t >= static_cast<int64_t>(xrows.size())) {
                        die("token index out of range in " + tp.string());
                    }
                    if (held_token[static_cast<size_t>(t)]) {
                        held_rows.push_back(xrows[static_cast<size_t>(t)]);
                    } else {
                        fit_rows.push_back(xrows[static_cast<size_t>(t)]);
                    }
                }
            } else {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "expert_%03d_hidden.npy", e);
                const fs::path hp = opt.exp_dumps / buf;
                if (fs::is_regular_file(hp)) {
                    int64_t nh = 0, ncol = 0;
                    hidden = npy_f32(hp, nh, ncol);
                    if (ncol != n_in) {
                        die(hp.string() + " has " + std::to_string(ncol) + " columns, expected " +
                            std::to_string(n_in));
                    }
                    std::vector<int64_t> all(static_cast<size_t>(nh));
                    for (int64_t t = 0; t < nh; ++t) {
                        all[static_cast<size_t>(t)] = t;
                    }
                    std::vector<int64_t> fit, held;
                    split_holdout(all, opt.exp_holdout, fit, held);
                    for (int64_t t : fit) {
                        fit_rows.push_back(hidden.data() + t * ncol);
                    }
                    for (int64_t t : held) {
                        held_rows.push_back(hidden.data() + t * ncol);
                    }
                }
            }
            const float * im = nullptr;          // the importance the quantizer is given
            const float * im_metric = nullptr;   // the yardstick the weighted error uses
            std::vector<float> im_diag, hess, uniform_row;
            const float * im_old = imp ? imp->at(res.tensor, n_in, h.n_expert, e) : nullptr;
            if (method == ExpMethod::Old) {
                im = im_old;
                im_metric = im_old;
            } else {
                diag_importance(fit_rows, n_in, im_diag);
                im = im_diag.data();
                im_metric = im_old ? im_old : im_diag.data();
            }
            if (!im) {
                if (type_needs_imatrix(type)) {
                    die(std::string("experiment: ") + type_name(type) + " needs an imatrix");
                }
                uniform_row.assign(static_cast<size_t>(n_in), 1.f);
                im = uniform_row.data();
                im_metric = im;
            }
            std::vector<float> f32(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
            std::vector<float> dq(f32.size());
            std::vector<uint8_t> q(row_q * static_cast<size_t>(n_out));
            dequant_rows(w, 0, n_out, n_in, f32.data(), 0.f, wthreads);
            const auto tq0 = std::chrono::steady_clock::now();
            if (method == ExpMethod::Gptq) {
                hess.assign(static_cast<size_t>(n_in) * static_cast<size_t>(n_in), 0.f);
                add_outer(hess.data(), n_in, fit_rows, wthreads);
                if (!is_down[k] && opt.gptq_prior > 0.0) {
                    add_scaled(hess.data(), hbar.data(), n_in, opt.gptq_prior, wthreads);
                } else if (fit_rows.empty() && !hbar.empty() && n_in == h.n_embd) {
                    add_scaled(hess.data(), hbar.data(), n_in, 1.0, wthreads);
                }
                double sec = 0.0;
                gptq::round_rows(type, f32.data(), n_out, n_in, hess.data(), q.data(), opt.gptq_damp,
                                 wthreads, &sec);
                logf("gptq %-4s expert %3d: %6lld fit rows, %5lld held, %.1f s\n", kinds[k], e,
                     static_cast<long long>(fit_rows.size()), static_cast<long long>(held_rows.size()), sec);
            } else {
                quantize_rows(type, f32.data(), q.data(), n_out, n_in, im, workers > 1 ? wthreads : threads);
            }
            st[static_cast<size_t>(e)].qsec +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tq0).count();
            for (int64_t row = 0; row < n_out; ++row) {
                gptq::dequant_row(type, q.data() + static_cast<size_t>(row) * row_q,
                                  dq.data() + row * n_in, n_in);
            }
            double pnum = 0.0, pden = 0.0, wnum = 0.0, wden = 0.0, wrow = 0.0;
            int64_t wrow_n = 0;
            for (int64_t row = 0; row < n_out; ++row) {
                const float * wr = f32.data() + row * n_in;
                const float * qr = dq.data() + row * n_in;
                double rn = 0.0, rd = 0.0, rwn = 0.0, rwd = 0.0;
                for (int64_t i = 0; i < n_in; ++i) {
                    const double wv = wr[i];
                    const double dd = wv - qr[i];
                    const double iw = im_metric[i];
                    rn += dd * dd;
                    rd += wv * wv;
                    rwn += iw * dd * dd;
                    rwd += iw * wv * wv;
                }
                pnum += rn;
                pden += rd;
                wnum += rwn;
                wden += rwd;
                if (rwd > 0.0) {
                    wrow += std::sqrt(rwn / rwd);
                    ++wrow_n;
                }
            }
            ExpStat & s = st[static_cast<size_t>(e)];
            s.plain_num += pnum;
            s.plain_den += pden;
            s.plain_mean += pden > 0.0 ? std::sqrt(pnum / pden) : 0.0;
            s.wnum += wnum;
            s.wden += wden;
            s.wmean += wrow_n > 0 ? wrow / static_cast<double>(wrow_n) : 0.0;
            if (!held_rows.empty()) {
                matvec_error(f32.data(), dq.data(), n_out, n_in, held_rows, wthreads, s.onum, s.oden);
                s.otok += static_cast<int64_t>(held_rows.size());
                ++s.oexp;
            }
            if (!opt.exp_pack.empty()) {
                pack_bytes[k][static_cast<size_t>(e)].assign(q.begin(), q.end());
            }
            if (!opt.exp_pack_frag.empty()) {
                fs::create_directories(opt.exp_pack_frag);
                char fb[96];
                std::snprintf(fb, sizeof(fb), "%s-%03d.bin", kinds[k], e);
                const fs::path fp = opt.exp_pack_frag / fb;
                std::ofstream fo(fp, std::ios::binary);
                if (!fo) {
                    die("cannot write " + fp.string());
                }
                fo.write(reinterpret_cast<const char *>(q.data()), static_cast<std::streamsize>(q.size()));
                fo.close();
                if (!fo) {
                    die("short write " + fp.string());
                }
            }
        };
        const int64_t span = nexp - estart;
        if (workers <= 1) {
            for (int e = estart; e < nexp; ++e) {
                one_expert(e);
            }
        } else {
            gptq::parallel_for(span, workers, [&](int64_t a, int64_t b) {
                for (int64_t i = a; i < b; ++i) {
                    one_expert(estart + static_cast<int>(i));
                }
            });
        }
        for (int e = estart; e < nexp; ++e) {
            const ExpStat & s = st[static_cast<size_t>(e)];
            res.plain_num += s.plain_num;
            res.plain_den += s.plain_den;
            res.plain_mean += s.plain_mean;
            res.weighted_num += s.wnum;
            res.weighted_den += s.wden;
            res.weighted_mean += s.wmean;
            res.out_num += s.onum;
            res.out_den += s.oden;
            res.out_tokens += s.otok;
            res.out_experts += s.oexp;
            res.quant_sec += s.qsec;
            ++res.experts;
        }
        results.push_back(std::move(res));
    }

    if (!opt.exp_pack.empty()) {
        const int64_t n_in[3] = {h.n_embd, h.n_embd, h.n_ff_exp};
        const int64_t n_out[3] = {h.n_ff_exp, h.n_ff_exp, h.n_embd};
        GgufWriter w;
        w.alignment = 32;
        w.open(opt.exp_pack);
        w.add_str("general.architecture", "minimax-m3");
        w.add_str("general.name", "MiniMax-M3.1-Flash layer " + std::to_string(opt.exp_layer) + " " +
                                       opt.exp_method + " " + type_name(type));
        w.add_str("general.baseline", "sparsa minimax gptq-iq experiment");
        w.add_u32("general.alignment", 32);
        w.add_u32("minimax-m3.block_count", static_cast<uint32_t>(h.n_layer));
        w.add_u32("minimax-m3.expert_count", static_cast<uint32_t>(h.n_expert));
        w.add_str("minimax.quant.method", opt.exp_method);
        w.add_f32("minimax.quant.gptq_prior", static_cast<float>(opt.gptq_prior));
        w.add_f32("minimax.quant.gptq_damp", static_cast<float>(opt.gptq_damp));
        w.add_f32("minimax.quant.holdout", static_cast<float>(opt.exp_holdout));
        for (int k = 0; k < 3; ++k) {
            TensorDesc t;
            t.name = blk_name(opt.exp_layer, leaves[k]);
            t.n_dims = 3;
            t.ne[0] = n_in[k];
            t.ne[1] = n_out[k];
            t.ne[2] = nexp;
            t.type = static_cast<uint32_t>(type);
            t.nbytes = row_size(type, n_in[k]) * static_cast<uint64_t>(n_out[k]) * static_cast<uint64_t>(nexp);
            w.add_tensor(std::move(t));
        }
        w.write_header();
        for (int k = 0; k < 3; ++k) {
            for (const std::vector<uint8_t> & b : pack_bytes[k]) {
                w.write_payload(b.data(), b.size());
            }
            w.end_tensor();
        }
        w.close();
        logf("wrote layer pack %s (%d experts, %s)\n", opt.exp_pack.string().c_str(), nexp, type_name(type));
    }

    char line[512];
    for (const KindResult & res : results) {
        const double plain = res.plain_den > 0 ? std::sqrt(res.plain_num / res.plain_den) : 0.0;
        const double wglobal = res.weighted_den > 0 ? std::sqrt(res.weighted_num / res.weighted_den) : 0.0;
        const double outc = res.out_den > 0 ? std::sqrt(res.out_num / res.out_den) : 0.0;
        std::snprintf(line, sizeof(line),
                      "exp %-4s %-8s experts %d  plain %.6f  plain_mean %.6f  weighted %.6f  weighted_global %.6f  output %.6f  tokens %lld  outEX %d  outNum %.6g  outDen %.6g  quant %.1fs\n",
                      res.name.c_str(), type_name(type), static_cast<int>(res.experts), plain,
                      res.plain_mean / std::max<int64_t>(1, res.experts),
                      res.weighted_mean / std::max<int64_t>(1, res.experts), wglobal, outc,
                      static_cast<long long>(res.out_tokens), static_cast<int>(res.out_experts),
                      res.out_num, res.out_den, res.quant_sec);
        logf("%s", line);
    }
    if (!opt.exp_out.empty()) {
        if (!opt.exp_out.parent_path().empty()) {
            fs::create_directories(opt.exp_out.parent_path());
        }
        std::ofstream o(opt.exp_out);
        if (!o) {
            die("cannot write " + opt.exp_out.string());
        }
        auto js = [](const std::string & s) {
            std::string r;
            for (char c : s) {
                if (c == '\\' || c == '"') {
                    r.push_back('\\');
                }
                r.push_back(c);
            }
            return r;
        };
        char num[2048];
        o << "{\n";
        o << "  \"layer\": " << opt.exp_layer << ",\n";
        o << "  \"type\": \"" << type_name(type) << "\",\n";
        o << "  \"experts\": " << nexp << ",\n";
        o << "  \"imatrix\": \"" << js(opt.imatrix.string()) << "\",\n";
        o << "  \"imatrix_test_only\": " << (imp && imp->test_only ? "true" : "false") << ",\n";
        o << "  \"imatrix_min_count\": " << opt.imatrix_min_count << ",\n";
        o << "  \"imatrix_blend\": " << opt.imatrix_blend << ",\n";
        o << "  \"exp_x\": \"" << js(opt.exp_x.string()) << "\",\n";
        o << "  \"exp_dumps\": \"" << js(opt.exp_dumps.string()) << "\",\n";
        o << "  \"kinds\": {\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const KindResult & res = results[i];
            const double plain = res.plain_den > 0 ? std::sqrt(res.plain_num / res.plain_den) : 0.0;
            const double wglobal = res.weighted_den > 0 ? std::sqrt(res.weighted_num / res.weighted_den) : 0.0;
            const double outc = res.out_den > 0 ? std::sqrt(res.out_num / res.out_den) : 0.0;
            o << "    \"" << res.name << "\": {";
            std::snprintf(num, sizeof(num),
                          "\"tensor\": \"%s\", \"experts\": %d, \"plain_rel\": %.9g, \"plain_rel_mean\": %.9g, "
                          "\"weighted_rel\": %.9g, \"weighted_rel_global\": %.9g, \"output_rel\": %.9g, "
                          "\"output_tokens\": %lld, \"output_experts\": %d, \"quant_seconds\": %.3f",
                          res.tensor.c_str(), static_cast<int>(res.experts), plain,
                          res.plain_mean / std::max<int64_t>(1, res.experts),
                          res.weighted_mean / std::max<int64_t>(1, res.experts), wglobal, outc,
                          static_cast<long long>(res.out_tokens), static_cast<int>(res.out_experts),
                          res.quant_sec);
            o << num << "}";
            o << (i + 1 < results.size() ? ",\n" : "\n");
        }
        o << "  }\n}\n";
        o.close();
        logf("wrote experiment report %s\n", opt.exp_out.string().c_str());
    }
}

// ---------------------------------------------------------------------------

fs::path find_python() {
    if (const char * e = std::getenv("PYTHON")) {
        if (fs::is_regular_file(e)) {
            return e;
        }
    }
#ifdef _WIN32
    const fs::path cand = "C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python311\\python.exe";
    if (fs::is_regular_file(cand)) {
        return cand;
    }
    return "python";
#else
    return "python3";
#endif
}

fs::path find_llama(const Options & opt) {
    if (!opt.llama_cpp.empty()) {
        return opt.llama_cpp;
    }
    if (const char * e = std::getenv("STRATA_LLAMA_CPP_DIR")) {
        return e;
    }
    return {};
}

int run_mmproj(const Options & opt) {
    const fs::path s13 = opt.model / "model-00013-of-00048.safetensors";
    const fs::path s48 = opt.model / "model-00048-of-00048.safetensors";
    std::string missing;
    if (!fs::is_regular_file(s13)) {
        missing += " model-00013-of-00048.safetensors";
    }
    if (!fs::is_regular_file(s48)) {
        missing += " model-00048-of-00048.safetensors";
    }
    if (!missing.empty()) {
        logf("SKIP mmproj:%s\n", missing.c_str());
        logf("Vision tensors are BF16. The tower is shard 48 and part of the projector is shard 13. No mmproj file was invented.\n");
        return 0;
    }
    fs::path script = opt.exe_dir / "mmproj.py";
    if (!fs::is_regular_file(script)) {
        script = fs::path("tools") / "minimax" / "quant" / "mmproj.py";
    }
    const fs::path llama = find_llama(opt);
    if (llama.empty() || !fs::is_regular_file(llama / "convert_hf_to_gguf.py")) {
        die("--mmproj needs --llama-cpp (or STRATA_LLAMA_CPP_DIR) pointing at a tree with convert_hf_to_gguf.py");
    }
    const fs::path py = find_python();
    logf("mmproj via %s\n", script.string().c_str());
#ifdef _WIN32
    auto quote = [](const std::wstring & s) {
        std::wstring o = L"\"";
        for (wchar_t c : s) {
            if (c == L'"') {
                o += L"\\\"";
            } else {
                o.push_back(c);
            }
        }
        o += L'"';
        return o;
    };
    const std::wstring cmd = quote(py.wstring()) + L" " + quote(script.wstring()) + L" " +
                             quote(opt.model.wstring()) + L" " + quote(llama.wstring()) + L" " +
                             quote(opt.mmproj_out.wstring());
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        die("CreateProcessW failed for mmproj");
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (code == 2) {
        logf("SKIP mmproj (converter reported missing vision shards)\n");
        return 0;
    }
    return code == 0 ? 0 : 1;
#else
    pid_t pid = fork();
    if (pid < 0) {
        die("fork failed");
    }
    if (pid == 0) {
        execlp(py.c_str(), py.c_str(), script.c_str(), opt.model.c_str(), llama.c_str(), opt.mmproj_out.c_str(),
               (char *) nullptr);
        _exit(127);
    }
    int st = 1;
    waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 2) {
        logf("SKIP mmproj (converter reported missing vision shards)\n");
        return 0;
    }
    return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : 1;
#endif
}

bool need_recipe(const Options & opt) {
    return opt.dry_run || opt.bench || !opt.out.empty() || !opt.write_imatrix.empty() || opt.exp_layer >= 0;
}

}  // namespace

int run(const Options & opt) {
    try {
        if (opt.gptq_selftest > 0) {
            const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::thread::hardware_concurrency());
            experiment_selftest(opt.gptq_selftest, threads);
            return 0;
        }
        const Hyp h = load_hyp(opt.model);
        logf("MiniMax-M3.1 text  layers %d  embd %d  dense_ff %d  expert_ff %d  experts %d  dense_prefix %d  indexer_first %d\n",
             h.n_layer, h.n_embd, h.n_ff, h.n_ff_exp, h.n_expert, h.n_dense, h.idx_first);
        const int threads = threads_of(opt);
        logf("threads %d\n", threads);

        Recipe recipe;
        Profile prof;
        std::string hot_mode = opt.hot_mode.empty() ? "record" : opt.hot_mode;
        std::vector<int> sel;
        std::vector<Item> items;
        int split_tensors = opt.split_tensors;
        if (need_recipe(opt)) {
            recipe = load_recipe(opt);
            prof = load_profile(opt);
            if (!opt.hot_mode_set && !prof.mode.empty()) {
                hot_mode = prof.mode;
            }
            if (hot_mode != "record" && hot_mode != "promote-layer") {
                die("--hot-expert-mode must be record or promote-layer");
            }
            if (opt.emit_lo >= 0 && opt.keep_indices) {
                die("--emit filters a remapped plan; do not combine it with --layers A-B");
            }
            if (opt.emit_lo >= 0 && opt.emit_hi < opt.emit_lo) {
                die("--emit range is empty");
            }
            if (opt.no_globals && opt.include_globals) {
                die("--include-globals and --no-globals conflict");
            }
            if (opt.split_part >= 0 || opt.split_count > 0 || opt.split_tensors >= 0) {
                if (opt.split_count < 2) {
                    die("--split-count must be at least 2");
                }
                if (opt.split_part < 0 || opt.split_part >= opt.split_count) {
                    die("--split-part must be in 0 .. split-count-1");
                }
            }
            sel = select_layers(opt, h);
            if (opt.emit_lo >= 0 && opt.emit_hi >= static_cast<int>(sel.size())) {
                die("--emit is outside the selected layers");
            }
            const std::vector<Item> planned = build_plan(h, recipe, sel, hot_mode, prof, opt.keep_indices);
            if (opt.split_count >= 2 && split_tensors < 0) {
                if (opt.keep_indices) {
                    std::vector<int> all;
                    all.reserve(static_cast<size_t>(h.n_layer));
                    for (int i = 0; i < h.n_layer; ++i) {
                        all.push_back(i);
                    }
                    split_tensors = static_cast<int>(build_plan(h, recipe, all, hot_mode, prof, true).size());
                } else {
                    split_tensors = static_cast<int>(planned.size());
                }
            }
            items = filter_items(planned, opt);
            if (!opt.out.empty() && items.empty()) {
                die("this part has no tensors");
            }
            logf("recipe %s  experts %s  down %s  file_type %u  imatrix %s  layers %zu  tensors %zu\n",
                 recipe.name.c_str(), type_name(recipe.experts), type_name(recipe.ffn_down),
                 recipe.file_type, recipe.require_imatrix ? "required" : "optional", sel.size(), items.size());
            if (opt.split_count >= 2) {
                logf("split part %d/%d  globals %s  split.tensors.count %d\n", opt.split_part, opt.split_count,
                     want_globals(opt) ? "yes" : "no", split_tensors);
            }
        }
        if (opt.dry_run) {
            print_sizes("full model from config (promote-layer not applied)", project_full(h, recipe));
            if (!items.empty()) {
                print_sizes("selected layers", sizes_of_plan(items));
            }
        }
        if (!opt.write_imatrix.empty()) {
            if (!opt.write_imatrix.parent_path().empty()) {
                fs::create_directories(opt.write_imatrix.parent_path());
            }
            write_test_imatrix(opt.write_imatrix, items);
        }

        const bool need_weights = opt.nvfp4_check || !opt.dump_name.empty() || opt.bench || !opt.out.empty() ||
                                  opt.exp_layer >= 0;
        Catalog cat;
        if (need_weights) {
            std::vector<std::string> needles;
            const bool subset = !opt.layers.empty() || opt.max_layers >= 0;
            if (subset) {
                std::unordered_set<int> layers;
                for (const Item & it : items) {
                    if (it.src >= 0) {
                        layers.insert(it.src);
                    }
                }
                for (int s : layers) {
                    needles.push_back("language_model.model.layers." + std::to_string(s) + ".");
                }
                if (want_globals(opt) && !(opt.bench && opt.out.empty())) {
                    needles.push_back("language_model.model.embed_tokens");
                    needles.push_back("language_model.lm_head");
                    needles.push_back("language_model.model.norm.weight");
                }
            }
            cat = load_catalog(opt.model, needles);
        }
        Imp loaded;
        const Imp * imp = nullptr;
        if (!opt.imatrix.empty()) {
            loaded = load_imatrix(opt.imatrix, opt.imatrix_min_count, opt.imatrix_blend);
            imp = &loaded;
        }
        if (opt.nvfp4_check) {
            nvfp4_check(cat, h);
        }
        if (!opt.dump_name.empty()) {
            if (opt.dump_out.empty()) {
                die("--dump-dequant needs an output path");
            }
            dump_dequant(cat, opt, threads);
        }
        if (opt.bench) {
            bench(cat, h, recipe, sel, imp, opt, threads);
        }
        if (!opt.pack_merge.empty()) {
            merge_layer_pack(opt, h);
        }
        if (opt.exp_layer >= 0) {
            experiment(cat, h, recipe, opt, imp, threads);
        }
        if (!opt.out.empty()) {
            Options wopt = opt;
            wopt.split_tensors = split_tensors;
            write_model(wopt, h, recipe, prof, hot_mode, sel, items, cat, imp, threads);
        }
        int rc = 0;
        if (!opt.mmproj_out.empty()) {
            rc = run_mmproj(opt);
        }
        if (!opt.dry_run && opt.out.empty() && opt.write_imatrix.empty() && opt.dump_name.empty() && !opt.bench &&
            !opt.nvfp4_check && opt.mmproj_out.empty() && opt.exp_layer < 0) {
            die("nothing to do: pass --out, --dry-run, --bench, --nvfp4-check, --dump-dequant, --write-test-imatrix, --exp-layer, or --mmproj");
        }
        return rc;
    } catch (const std::exception & e) {
        logf("error: %s\n", e.what());
        return 1;
    }
}

}  // namespace mmq
