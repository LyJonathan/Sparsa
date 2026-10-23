// src/minimax/engine.cpp - MiniMax-M3.1-Flash forward pass.
//
// Token-by-token, prefill and decode share one path (P2b; a batched prefill can replace it later without changing
// the math). Dense GEMVs are fp32. Quantized routed experts are ggml-cpu vec_dot (activation quantized with the
// type's vec_dot_type) plus swigluoai. KV is fp16 on the host only; nothing is mirrored to the device until a GPU
// attention kernel exists. See include/strata/minimax/forward.hpp.
#include "strata/minimax/forward.hpp"

#include "cuda_ops.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/platform/direct_file.hpp"

namespace strata::minimax {
namespace {

constexpr int kF32 = 0;
constexpr int kF16 = 1;
constexpr int kBF16 = 30;
constexpr int kTile = 128;
constexpr int32_t kNotResident = -1;

struct DevPtr {
    void* p = nullptr;
    DevPtr() = default;
    explicit DevPtr(void* q) : p(q) {}
    DevPtr(const DevPtr&) = delete;
    DevPtr& operator=(const DevPtr&) = delete;
    DevPtr(DevPtr&& o) noexcept : p(o.p) { o.p = nullptr; }
    DevPtr& operator=(DevPtr&& o) noexcept {
        if (p) cu::dealloc(p);
        p = o.p;
        o.p = nullptr;
        return *this;
    }
    ~DevPtr() {
        if (p) cu::dealloc(p);
    }
};

struct Tensor {
    std::string name;
    int type = 0;
    int64_t ne0 = 1, ne1 = 1, ne2 = 1;
    const uint8_t* host = nullptr;
    size_t bytes = 0;
    size_t row_bytes = 0;
    size_t shard = 0;
    uint64_t file_offset = 0;   ///< byte offset in its GGUF shard, for the SSD tier's unbuffered reads
    DevPtr dev;
    bool dev_f32 = false;
    bool gpu = false;
    const ggml_type_traits_cpu* cpu = nullptr;
    const ggml_type_traits* base = nullptr;
};

struct Layer {
    bool moe = false;
    bool indexer = false;
    bool want_gpu = false;
    std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, iq_norm, ik_norm, bias;
    Tensor wq, wk, wv, wo;
    Tensor gate, up, down;          // dense MLP, or the shared expert on a MoE layer
    bool have_shared = false;
    Tensor router;
    Tensor egate, eup, edown;
    Tensor iq, ik;

    // P3: bytes of one expert's three role slices, in [gate, up, down] order, and the per-layer tiers.
    size_t expert_gu = 0;           // one gate matrix (up is the same)
    size_t expert_down = 0;
    size_t expert_bytes = 0;
    // MM-NVFP4: per-expert global scales (a DIVISOR) from blk.N.ffn_*_exps.scale. Empty when the
    // layer's expert tensors carry no side scale. Applied exactly once per projection.
    std::vector<float> egate_scale, eup_scale, edown_scale;
    struct VramSlot {
        DevPtr mem;
        int32_t expert = -1;
    };
    std::vector<VramSlot> vram;
    std::vector<int32_t> vram_of;   // n_expert -> slot or -1
    uint64_t ram_off = 0;           // byte offset of slot 0 in the pinned arena
    int ram_n = 0;
    std::vector<int32_t> ram_of;
    std::vector<int32_t> ram_expert_of;   // slot -> expert
    std::vector<uint64_t> ram_tick;
};

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

// ---- MM-NVFP4 parity probes: a minimal .npy codec (little-endian fp32, C order, no pickle).
bool read_npy_f32(const std::string& path, std::vector<float>& out, int64_t& rows, int64_t& cols) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char magic[8] = {};
    unsigned char hlen_le[2] = {};
    if (std::fread(magic, 1, 8, f) != 8 || std::fread(hlen_le, 1, 2, f) != 2) {
        std::fclose(f);
        return false;
    }
    const unsigned hlen = (unsigned)hlen_le[0] | ((unsigned)hlen_le[1] << 8);
    std::string header(hlen, '\0');
    if (hlen == 0 || std::fread(&header[0], 1, hlen, f) != hlen) {
        std::fclose(f);
        return false;
    }
    if (header.find("<f4") == std::string::npos || header.find("'fortran_order': True") != std::string::npos) {
        std::fclose(f);
        return false;
    }
    const size_t lp = header.find('(');
    const size_t rp = header.find(')', lp);
    int64_t a = 0, b = 1;
    if (lp == std::string::npos || rp == std::string::npos) {
        std::fclose(f);
        return false;
    }
    const std::string shape = header.substr(lp + 1, rp - lp - 1);
    if (std::sscanf(shape.c_str(), "%lld , %lld", &a, &b) != 2 && std::sscanf(shape.c_str(), "%lld", &a) != 1) {
        std::fclose(f);
        return false;
    }
    if (a < 1 || b < 1) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t)a * (size_t)b);
    const size_t got = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    if (got != out.size()) return false;
    rows = a;
    cols = b;
    return true;
}

void write_npy_f32(const std::string& path, const float* data, int64_t rows, int64_t cols) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                         std::to_string(cols) + "), }";
    const int pad = (int)((64 - ((10 + header.size() + 1) % 64)) % 64);
    header.append((size_t)pad, ' ');
    header.push_back('\n');
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) fail("cannot create " + path);
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    std::fwrite(magic, 1, 8, f);
    const uint16_t hlen = (uint16_t)header.size();
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(header.data(), 1, header.size(), f);
    const size_t n = (size_t)rows * (size_t)cols;
    if (std::fwrite(data, sizeof(float), n, f) != n) {
        std::fclose(f);
        fail("short npy write " + path);
    }
    std::fclose(f);
}

int64_t bigger(int64_t a, int64_t b) { return a > b ? a : b; }

// One pool for the whole engine. parallel_for used to spawn and join on every GEMV row-split.
struct Pool {
    int nthreads = 1;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::function<void(int, int)> fn;
    int rows = 0;
    int parts = 1;
    int epoch = 0;
    int pending = 0;
    bool stop = false;

    Pool() = default;
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    void start(int threads) {
        if (!workers.empty()) return;
        if (threads < 1) threads = 1;
        nthreads = threads;
        for (int p = 1; p < nthreads; ++p) workers.emplace_back([this, p] { worker(p); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            ++epoch;
        }
        cv_work.notify_all();
        for (auto& t : workers)
            if (t.joinable()) t.join();
    }

    void worker(int p) {
        int seen = 0;
        for (;;) {
            std::function<void(int, int)> job;
            int a = 0, b = 0;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [&] { return stop || epoch != seen; });
                if (stop) return;
                seen = epoch;
                job = fn;
                a = (int)((int64_t)rows * p / parts);
                b = (int)((int64_t)rows * (p + 1) / parts);
            }
            if (a < b && job) job(a, b);
            std::lock_guard<std::mutex> lk(mu);
            if (--pending == 0) cv_done.notify_one();
        }
    }

    void run(int nrows, const std::function<void(int, int)>& job) {
        if (nrows <= 0) return;
        if (nthreads <= 1 || nrows < nthreads * 4) {
            job(0, nrows);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            fn = job;
            rows = nrows;
            parts = nthreads;
            pending = nthreads - 1;
            ++epoch;
        }
        cv_work.notify_all();
        const int b0 = (int)((int64_t)nrows / nthreads);
        if (b0 > 0) job(0, b0);
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return pending == 0; });
        fn = nullptr;
    }
};

bool mul_size(size_t a, size_t b, size_t& o) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    o = a * b;
    return true;
}

bool add_size(size_t a, size_t b, size_t& o) {
    if (a > std::numeric_limits<size_t>::max() - b) return false;
    o = a + b;
    return true;
}

float sigmoid(float x) {
    // fp64 exp, then back to fp32: stable for the large negative gates swigluoai does not clamp.
    return (float)(1.0 / (1.0 + std::exp(-(double)x)));
}

float swigluoai(float gate, float up, float alpha, float limit) {
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    return (up + 1.f) * gate * sigmoid(alpha * gate);
}

float dot_f32(const float* w, const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)w[i] * (double)x[i];
    return (float)s;
}

float dot_f16w(const uint16_t* w, const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)ggml_fp16_to_fp32(w[i]) * (double)x[i];
    return (float)s;
}

float dot_bf16w(const uint16_t* w, const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        ggml_bf16_t b;
        b.bits = w[i];
        s += (double)ggml_bf16_to_fp32(b) * (double)x[i];
    }
    return (float)s;
}

float dot_f16_cache(const uint16_t* a, const float* b, int n) { return dot_f16w(a, b, n); }

bool quant_gpu_ok(const Tensor& t) {
    if (t.type == kF32 || t.type == kF16 || t.type == kBF16) return false;
    const bool iq = t.type == 16 || t.type == 17 || t.type == 18 || t.type == 21 || t.type == 22 || t.type == 29;
    if (!iq && !strata::kernels::dequant_bf16_supported(t.type)) return false;
    int be = 0, bb = 0;
    if (!block_geometry((uint32_t)t.type, be, bb)) return false;
    return t.ne0 % be == 0;
}

size_t gpu_bytes(const Tensor& t) {
    const size_t n = (size_t)t.ne0 * (size_t)t.ne1 * (size_t)t.ne2;
    if (t.type == kF32) return t.bytes;
    if (t.type == kF16 || t.type == kBF16) return n * sizeof(float);
    if (quant_gpu_ok(t)) return t.bytes;
    return std::numeric_limits<size_t>::max();
}

void decode_f32(const Tensor& t, const uint8_t* src, float* dst, int64_t n) {
    // ggml leaves F32's to_float null; the bytes already are the floats.
    if (t.type == kF32) {
        std::memcpy(dst, src, (size_t)n * sizeof(float));
        return;
    }
    if (!t.base || !t.base->to_float) fail(t.name + ": ggml has no to_float for type " + std::to_string(t.type));
    t.base->to_float(src, dst, n);
}

const uint8_t* row_ptr(const Tensor& t, int64_t e, int64_t r) {
    return t.host + (size_t)(e * t.ne1 + r) * t.row_bytes;
}

void rms(const float* x, const float* w, float* y, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.f / std::sqrt(ss / (float)n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

void rms_heads(float* x, int n_head, int dim, const float* w, float eps) {
    for (int h = 0; h < n_head; ++h) rms(x + (size_t)h * dim, w, x + (size_t)h * dim, dim, eps);
}

void rope(float* x, int n_head, int dim, int n_rot, int pos, double theta) {
    const int half = n_rot / 2;
    for (int h = 0; h < n_head; ++h) {
        float* v = x + (size_t)h * dim;
        for (int i = 0; i < half; ++i) {
            const double freq = (double)pos / std::pow(theta, (2.0 * i) / (double)n_rot);
            const float c = (float)std::cos(freq);
            const float s = (float)std::sin(freq);
            const float a = v[i];
            const float b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

void softmax_inplace(float* s, int n) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i)
        if (s[i] > m) m = s[i];
    if (!(m > -std::numeric_limits<float>::infinity())) {
        for (int i = 0; i < n; ++i) s[i] = 0.f;
        return;
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        if (s[i] == -std::numeric_limits<float>::infinity()) s[i] = 0.f;
        else s[i] = std::exp(s[i] - m);
        sum += s[i];
    }
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int i = 0; i < n; ++i) s[i] *= inv;
}

bool finite_row(const float* x, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}

bool raw_expert_type(int t) { return t == kF32 || t == kF16 || t == kBF16; }
bool iq_expert_type(int t) { return t == 8 || strata::kernels::iq_supported(t); }

// Available physical memory, for the automatic pinned-tier budget.
size_t host_available_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) return (size_t)m.ullAvailPhys;
    return 0;
#else
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t v = 0;
    std::string unit;
    while (f >> key >> v >> unit) {
        if (key == "MemAvailable:") return v * 1024;
    }
    return 0;
#endif
}

// ---- STRP, byte for byte the format tools/make_profile.py and src/core/expert_cache.cpp read.  MiniMax has one
// profile row per MoE layer (57 x 128 for M3.1), not per model layer: dense layers have no routed experts.
bool read_strp(const std::string& path, int64_t n_moe, int64_t n_expert,
               std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "minimax profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "minimax profile: " + path + " is too short";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "minimax profile: " + path + " does not start with STRP";
        return false;
    }
    if (hdr[0] != 1 || (int64_t)hdr[1] != n_moe || (int64_t)hdr[2] != n_expert) {
        std::fclose(f);
        err = "minimax profile: " + path + " is version " + std::to_string(hdr[0]) + ", " +
              std::to_string(hdr[1]) + "x" + std::to_string(hdr[2]) + "; this model needs 1, " +
              std::to_string(n_moe) + "x" + std::to_string(n_expert) + " (MoE layers only)";
        return false;
    }
    const uint32_t n = hdr[4];
    if (n > hdr[3] || n > (uint32_t)(n_moe * n_expert)) {
        std::fclose(f);
        err = "minimax profile: the header's ranked count is inconsistent";
        return false;
    }
    std::vector<uint16_t> raw((size_t)n * 2);
    if (n && std::fread(raw.data(), 2, raw.size(), f) != raw.size()) {
        std::fclose(f);
        err = "minimax profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    ranked.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t l = raw[(size_t)i * 2], e = raw[(size_t)i * 2 + 1];
        if (l < 0 || l >= n_moe || e < 0 || e >= n_expert) {
            err = "minimax profile: a ranked pair is out of range";
            ranked.clear();
            return false;
        }
        ranked[(size_t)i] = {l, e};
    }
    return true;
}

bool write_strp(const std::string& path, int64_t n_moe, int64_t n_expert,
                const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_moe <= 0 || n_expert <= 0 || n_moe > 65535 || n_expert > 65535) {
        err = "minimax profile: the model's layout does not fit STRP";
        return false;
    }
    std::vector<int32_t> table((size_t)(n_moe * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_moe || e < 0 || e >= n_expert) {
            err = "minimax profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t)(l * n_expert + e)] = (int32_t)r;
        pairs.push_back((uint16_t)l);
        pairs.push_back((uint16_t)e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t)n_moe, (uint32_t)n_expert, (uint32_t)ranked.size(),
                             (uint32_t)ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "minimax profile: cannot create " + tmp;
        return false;
    }
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "minimax profile: cannot write " + path;
        return false;
    }
    return true;
}

}  // namespace

struct Engine::Impl {
    EngineOptions opt;
    GgufModel model;
    Config cfg;
    bool norm_add1 = false;
    int dense_gpu = 0;
    bool output_gpu = false;
    int threads = 1;
    int64_t ctx = 0;
    float eps = 1e-6f;
    float alpha = 1.702f;
    float limit = 7.f;
    float scale_routed = 2.f;
    bool renorm = true;

    Tensor embd, output;
    std::vector<float> out_norm;
    std::vector<Layer> layers;
    std::unordered_set<int32_t> eos;

    std::vector<uint16_t> k_host, v_host, ik_host;
    DevPtr d_x, d_y, d_scratch;
    int64_t max_cols = 1, max_rows = 1;

    std::vector<float> x, h, q, k, v, proj, ff_g, ff_u, ff_h, ff_y, iq, ik, logits, last_logits;
    std::vector<int32_t> held;
    int64_t n_past = 0;
    bool have_logits = false;
    std::mt19937_64 rng{0};
    Pool pool;

    // ---- P3 expert tiers
    bool tiers_on = false;
    bool tiers_gpu = false;         // a VRAM tier exists (CUDA is up even when dense layers are CPU)
    int vram_per_layer = 0;
    int ram_per_layer = 0;
    int moe_layers = 0;
    size_t max_expert_bytes = 0;
    std::vector<std::pair<int32_t, int32_t>> profile;   // (moe layer, expert), hottest first
    std::unique_ptr<strata::core::PinnedArena> ram_arena;
    uint64_t ram_bytes = 0;
    std::vector<std::unique_ptr<strata::platform::DirectFile>> ssd_files;
    uint8_t* ssd_stage = nullptr;
    size_t ssd_stage_bytes = 0;
    void* ssd_tmp = nullptr;
    size_t ssd_tmp_bytes = 0;
    void* side_stream = nullptr;

    // device scratch for the grouped routed-expert pass
    DevPtr d_ex_ptr;                // top * 3 pointers
    DevPtr d_ex_xq, d_ex_hq, d_ex_h, d_ex_out;
    DevPtr d_ex_sg, d_ex_su, d_ex_sd;   // MM-NVFP4: per-selected-expert global scales (f32), device

    // ---- MM-NVFP4 parity probes. Off unless --minimax-dump-dir / --minimax-seed-hidden is given.
    bool dump_on = false;
    std::string dump_dir;
    std::vector<char> dump_sel;     // [n_layers]
    std::vector<int> dump_src;      // [n_layers] source layer id (strata.source_layers) for the file name
    std::unordered_map<int, std::vector<float>> dump_rows;   // layer -> rows so far, n_embd each
    std::vector<float> seed_h;
    int64_t seed_rows = 0, seed_cols = 0;

    std::vector<float> usage;       // decayed routing counts, [n_layers * n_expert]
    std::vector<double> route_counts;   // undecayed counts for --minimax-profile-save
    uint64_t rebalance_tick = 0;
    uint64_t lru_tick = 0;
    uint64_t swaps = 0;

    struct TokenStats {
        int64_t vram = 0, ram = 0, ssd = 0, ssd_bytes = 0;
    };
    TokenStats tok_stats;
    TokenStats total_stats;
    int64_t tokens_seen = 0;
    int64_t profile_saves = 0;

    explicit Impl(const std::string& path, const EngineOptions& o) : opt(o), model(GgufModel::open(path)) {
        cfg = from_gguf(model.meta());
        const std::string bad = check_structure(cfg);
        if (!bad.empty()) fail(bad);
        if (cfg.n_head % cfg.idx_heads != 0)
            fail("attention.head_count is not a multiple of the indexer head count");
        if (cfg.idx_first_layer < cfg.n_layers && cfg.n_rot > cfg.idx_dim)
            fail("rope.dimension_count is wider than the indexer head");
        eps = (float)cfg.rms_eps;
        alpha = (float)cfg.swiglu_alpha;
        limit = (float)cfg.swiglu_limit;
        scale_routed = (float)cfg.routed_scale;
        renorm = cfg.expert_weights_norm;
        ctx = cfg.context;
        if (opt.max_context > 0 && opt.max_context < ctx) ctx = opt.max_context;
        if (ctx < 1) fail("context length is 0");

        ggml_cpu_init();
        load_tensors();
        decide_norm();
        build_layers();
        read_eos();
        init_probes();

        threads = opt.cpu_threads;
        if (threads <= 0) threads = (int)std::thread::hardware_concurrency();
        if (threads <= 0) threads = 4;
        if (threads > 24) threads = 24;
        pool.start(threads);

        // N=0 is the all-CPU dense path: no device, no CUDA runtime calls. Automatic placement (-1), any
        // positive layer count, and the P3 expert tiers still need a device to size VRAM.
        const bool cpu_only = opt.dense_layers_gpu == 0 && !opt.tiers;
        if (!cpu_only) cu::init();
        alloc_cache();
        // The row/col limits are also the CPU gemv guard. Device buffers exist only when a layer can be uploaded.
        size_workspace();
        if (!cpu_only) alloc_workspace();
        place_weights();
        if (opt.tiers) {
            init_tiers();
        } else if (!opt.profile_save_path.empty() || !opt.routing_counts_path.empty()) {
            for (const Layer& L : layers)
                if (L.moe) ++moe_layers;
            route_counts.assign((size_t)cfg.n_layers * (size_t)cfg.n_expert, 0.0);
        }

        const int64_t qlen = cfg.n_head * cfg.head_dim;
        const int64_t klen = cfg.n_head_kv * cfg.head_dim;
        const int64_t ff = bigger(cfg.n_ff_dense, cfg.n_ff_exp * (cfg.n_shared_experts > 0 ? cfg.n_shared_experts : 1));
        x.assign((size_t)cfg.n_embd, 0.f);
        h.assign((size_t)cfg.n_embd, 0.f);
        q.assign((size_t)qlen, 0.f);
        k.assign((size_t)klen, 0.f);
        v.assign((size_t)klen, 0.f);
        proj.assign((size_t)cfg.n_embd, 0.f);
        ff_g.assign((size_t)ff, 0.f);
        ff_u.assign((size_t)ff, 0.f);
        ff_h.assign((size_t)ff, 0.f);
        ff_y.assign((size_t)cfg.n_embd, 0.f);
        iq.assign((size_t)(cfg.idx_heads * cfg.idx_dim), 0.f);
        ik.assign((size_t)cfg.idx_dim, 0.f);
        logits.assign((size_t)cfg.n_vocab, 0.f);
        last_logits.assign((size_t)cfg.n_vocab, 0.f);
    }

    ~Impl() {
        write_own_profile();
        flush_dumps();
        if (side_stream) cu::stream_destroy(side_stream);
        if (ssd_stage) strata::platform::DirectFile::free_aligned(ssd_stage);
        if (ssd_tmp) strata::platform::DirectFile::free_aligned(ssd_tmp);
    }

    bool try_must(const std::string& name, Tensor& t) {
        size_t shard = 0;
        const TensorInfo* info = model.find(name, &shard);
        if (!info) return false;
        t.name = name;
        t.type = (int)info->type;
        t.ne0 = (int64_t)info->shape[0];
        t.ne1 = info->shape.size() > 1 ? (int64_t)info->shape[1] : 1;
        t.ne2 = info->shape.size() > 2 ? (int64_t)info->shape[2] : 1;
        if (info->shape.size() > 3) fail(name + " has more than 3 dimensions");
        t.host = model.shard(shard).tensor_data(*info);
        t.shard = shard;
        t.file_offset = model.shard(shard).data_start() + info->offset;
        t.bytes = tensor_payload_bytes(*info);
        t.row_bytes = ggml_row_size((ggml_type)t.type, t.ne0);
        if (t.row_bytes == 0 || t.row_bytes * (size_t)t.ne1 * (size_t)t.ne2 != t.bytes)
            fail(name + ": row size does not match the GGUF payload (type " + std::to_string(t.type) + ")");
        t.base = ggml_get_type_traits((ggml_type)t.type);
        t.cpu = ggml_get_type_traits_cpu((ggml_type)t.type);
        if (!t.base) fail(name + ": ggml has no type traits for type " + std::to_string(t.type));
        if (t.type != kF32 && !t.base->to_float)
            fail(name + ": no ggml to_float for type " + std::to_string(t.type));
        ggml_quantize_init((ggml_type)t.type);
        if (t.cpu && t.cpu->vec_dot_type != (ggml_type)t.type) ggml_quantize_init(t.cpu->vec_dot_type);
        return true;
    }

    Tensor must(const std::string& name) {
        Tensor t;
        if (!try_must(name, t)) fail("missing tensor " + name);
        return t;
    }

    // blk.N.ffn_*_exps.scale: f32 [n_expert], the MM-NVFP4 per-expert global scale (a divisor).
    bool load_scale_vec(const std::string& name, std::vector<float>& out, int64_t n) {
        Tensor t;
        if (!try_must(name, t)) return false;
        if (t.type != kF32 || t.ne0 != n || t.ne1 != 1 || t.ne2 != 1)
            fail(name + ": expected f32 [" + std::to_string(n) + "] (got type " + std::to_string(t.type) + ", " +
                 std::to_string(t.ne0) + "x" + std::to_string(t.ne1) + "x" + std::to_string(t.ne2) + ")");
        out.resize((size_t)n);
        decode_f32(t, t.host, out.data(), n);
        for (float v : out)
            if (!(v > 0.f) || !std::isfinite(v))
                fail(name + ": a global scale is not a positive finite float");
        // The tensor stores the DIVISOR (spec A.3). The kernels multiply, so keep the reciprocal here:
        // one division per expert at load, never per token, and the scale is applied exactly once.
        for (float& v : out) v = 1.0f / v;
        return true;
    }

    std::vector<float> norm_vec(const Tensor& t) {
        if (t.ne0 < 1 || t.ne1 != 1 || t.ne2 != 1) fail(t.name + ": expected a 1-D norm");
        std::vector<float> w((size_t)t.ne0);
        decode_f32(t, t.host, w.data(), t.ne0);
        if (norm_add1)
            for (float& v : w) v += 1.f;
        return w;
    }

    void load_tensors() {
        embd = must("token_embd.weight");
        output = must("output.weight");
        if (embd.ne0 != cfg.n_embd || embd.ne1 != cfg.n_vocab) fail("token_embd.weight has an unexpected shape");
        if (output.ne0 != cfg.n_embd || output.ne1 != cfg.n_vocab) fail("output.weight has an unexpected shape");
    }

    void decide_norm() {
        const Tensor on = must("output_norm.weight");
        const Tensor an = must("blk.0.attn_norm.weight");
        std::vector<float> a((size_t)on.ne0), b((size_t)an.ne0);
        decode_f32(on, on.host, a.data(), on.ne0);
        decode_f32(an, an.host, b.data(), an.ne0);
        double sum = 0.0;
        for (float v : a) sum += v;
        for (float v : b) sum += v;
        const double mean = sum / (double)(a.size() + b.size());
        const MetaValue* key = model.meta().get("minimax-m3.norm.add1");
        if (key) {
            if (key->type != MetaType::BOOL)
                fail("minimax-m3.norm.add1 is present but is not a bool");
            norm_add1 = key->u != 0;
            return;
        }
        // llama.cpp's converter bakes +1 into every *.norm.weight (spec F.2). Values then sit near 1.
        // A mean near 0 is the raw Gemma parameter. Refuse that on a real model; the synthetic zero file
        // (development only) is used as stored.
        if (mean < 0.5) {
            if (!opt.allow_nonstandard)
                fail("norm weights look un-baked (mean " + std::to_string(mean) +
                     "). A llama.cpp minimax-m3 GGUF already has +1 in the norm tensors. "
                     "Set minimax-m3.norm.add1 true to add 1 at runtime, or reconvert the file.");
            std::fprintf(stderr,
                         "strata-minimax: norm weights have mean %.4f; using them as stored (no +1) because "
                         "--minimax-allow-nonstandard is set. A real GGUF bakes +1 (values near 1). Set "
                         "minimax-m3.norm.add1 true to add 1 at runtime.\n",
                         mean);
            norm_add1 = false;
            return;
        }
        norm_add1 = false;
    }

    void build_layers() {
        layers.resize((size_t)cfg.n_layers);
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            Layer& L = layers[(size_t)i];
            const std::string b = "blk." + std::to_string(i) + ".";
            L.moe = cfg.is_moe(i);
            L.indexer = cfg.has_indexer(i);
            L.attn_norm = norm_vec(must(b + "attn_norm.weight"));
            L.ffn_norm = norm_vec(must(b + "ffn_norm.weight"));
            L.q_norm = norm_vec(must(b + "attn_q_norm.weight"));
            L.k_norm = norm_vec(must(b + "attn_k_norm.weight"));
            L.wq = must(b + "attn_q.weight");
            L.wk = must(b + "attn_k.weight");
            L.wv = must(b + "attn_v.weight");
            L.wo = must(b + "attn_output.weight");
            if (L.wq.ne0 != cfg.n_embd || L.wq.ne1 != cfg.n_head * cfg.head_dim) fail(L.wq.name + ": unexpected shape");
            if (L.wk.ne0 != cfg.n_embd || L.wk.ne1 != cfg.n_head_kv * cfg.head_dim) fail(L.wk.name + ": unexpected shape");
            if (L.wv.ne0 != cfg.n_embd || L.wv.ne1 != L.wk.ne1) fail(L.wv.name + ": unexpected shape");
            if (L.wo.ne0 != cfg.n_head * cfg.head_dim || L.wo.ne1 != cfg.n_embd) fail(L.wo.name + ": unexpected shape");
            if ((int)L.q_norm.size() != cfg.head_dim || (int)L.k_norm.size() != cfg.head_dim)
                fail(b + "q/k norm width is not head_dim");
            if (!L.moe) {
                L.gate = must(b + "ffn_gate.weight");
                L.up = must(b + "ffn_up.weight");
                L.down = must(b + "ffn_down.weight");
                if (L.gate.ne0 != cfg.n_embd || L.gate.ne1 != cfg.n_ff_dense) fail(L.gate.name + ": unexpected shape");
                if (L.up.ne0 != L.gate.ne0 || L.up.ne1 != L.gate.ne1) fail(L.up.name + ": unexpected shape");
                if (L.down.ne0 != cfg.n_ff_dense || L.down.ne1 != cfg.n_embd) fail(L.down.name + ": unexpected shape");
            } else {
                L.router = must(b + "ffn_gate_inp.weight");
                Tensor bias = must(b + "exp_probs_b.bias");
                if (L.router.ne0 != cfg.n_embd || L.router.ne1 != cfg.n_expert) fail(L.router.name + ": unexpected shape");
                if (bias.ne0 != cfg.n_expert) fail(bias.name + ": unexpected shape");
                L.bias.resize((size_t)cfg.n_expert);
                decode_f32(bias, bias.host, L.bias.data(), cfg.n_expert);
                L.egate = must(b + "ffn_gate_exps.weight");
                L.eup = must(b + "ffn_up_exps.weight");
                L.edown = must(b + "ffn_down_exps.weight");
                if (L.egate.ne0 != cfg.n_embd || L.egate.ne1 != cfg.n_ff_exp || L.egate.ne2 != cfg.n_expert)
                    fail(L.egate.name + ": unexpected shape");
                if (L.eup.ne0 != L.egate.ne0 || L.eup.ne1 != L.egate.ne1 || L.eup.ne2 != L.egate.ne2)
                    fail(L.eup.name + ": unexpected shape");
                if (L.edown.ne0 != cfg.n_ff_exp || L.edown.ne1 != cfg.n_embd || L.edown.ne2 != cfg.n_expert)
                    fail(L.edown.name + ": unexpected shape");
                if (L.egate.type != L.eup.type)
                    fail(b + "gate and up experts have different ggml types");
                // MM-NVFP4: per-expert global scale side tensors. Present only for the exact NVFP4 layout.
                const bool sg = load_scale_vec(b + "ffn_gate_exps.scale", L.egate_scale, cfg.n_expert);
                const bool su = load_scale_vec(b + "ffn_up_exps.scale", L.eup_scale, cfg.n_expert);
                const bool sd = load_scale_vec(b + "ffn_down_exps.scale", L.edown_scale, cfg.n_expert);
                if (sg != su) fail(b + " has only one of ffn_gate_exps.scale / ffn_up_exps.scale");
                if ((L.egate.type == 40 || L.eup.type == 40) && !sg)
                    fail(b + " stores gate/up experts as NVFP4 but has no blk.N.ffn_{gate,up}_exps.scale "
                            "(a per-expert global scale is a divisor, not foldable into the E4M3 byte)");
                if (L.edown.type == 40 && !sd)
                    fail(b + " stores down experts as NVFP4 but has no blk.N.ffn_down_exps.scale");
                if (sg && L.egate.type != 40)
                    fail(b + " has ffn_gate_exps.scale but the expert type is " + std::to_string(L.egate.type) +
                                ", not NVFP4 (40)");
                if (sd && L.edown.type != 40)
                    fail(b + " has ffn_down_exps.scale but the expert type is " + std::to_string(L.edown.type) +
                                ", not NVFP4 (40)");
                if (L.egate.ne2 > 0) L.expert_gu = L.egate.bytes / (size_t)L.egate.ne2;
                if (L.eup.ne2 > 0 && L.eup.bytes / (size_t)L.eup.ne2 != L.expert_gu)
                    fail(b + "gate and up experts have different byte sizes");
                if (L.edown.ne2 > 0) L.expert_down = L.edown.bytes / (size_t)L.edown.ne2;
                L.expert_bytes = L.expert_gu * 2 + L.expert_down;
                if (cfg.n_shared_experts > 0) {
                    L.have_shared = true;
                    L.gate = must(b + "ffn_gate_shexp.weight");
                    L.up = must(b + "ffn_up_shexp.weight");
                    L.down = must(b + "ffn_down_shexp.weight");
                    const int64_t sh = cfg.n_ff_exp * cfg.n_shared_experts;
                    if (L.gate.ne0 != cfg.n_embd || L.gate.ne1 != sh) fail(L.gate.name + ": unexpected shape");
                    if (L.up.ne0 != L.gate.ne0 || L.up.ne1 != L.gate.ne1) fail(L.up.name + ": unexpected shape");
                    if (L.down.ne0 != sh || L.down.ne1 != cfg.n_embd) fail(L.down.name + ": unexpected shape");
                }
            }
            if (L.indexer) {
                L.iq = must(b + "indexer.q_proj.weight");
                L.ik = must(b + "indexer.k_proj.weight");
                L.iq_norm = norm_vec(must(b + "indexer.q_norm.weight"));
                L.ik_norm = norm_vec(must(b + "indexer.k_norm.weight"));
                if (L.iq.ne0 != cfg.n_embd || L.iq.ne1 != cfg.idx_heads * cfg.idx_dim) fail(L.iq.name + ": unexpected shape");
                if (L.ik.ne0 != cfg.n_embd || L.ik.ne1 != cfg.idx_dim) fail(L.ik.name + ": unexpected shape");
                if ((int)L.iq_norm.size() != cfg.idx_dim || (int)L.ik_norm.size() != cfg.idx_dim)
                    fail(b + "indexer norm width is not indexer key_length");
            }
        }
        out_norm = norm_vec(must("output_norm.weight"));
    }

    // ---- MM-NVFP4 parity probes. Both are opt-in and change nothing in a normal run.
    void init_probes() {
        if (!opt.seed_hidden.empty()) {
            if (!read_npy_f32(opt.seed_hidden, seed_h, seed_rows, seed_cols))
                fail("--minimax-seed-hidden: cannot read " + opt.seed_hidden + " as an .npy fp32 C-order array");
            if (seed_cols != cfg.n_embd)
                fail("--minimax-seed-hidden: last dimension is " + std::to_string(seed_cols) +
                     ", the model needs " + std::to_string(cfg.n_embd));
            if (seed_rows < 1) fail("--minimax-seed-hidden: the array is empty");
        }
        if (opt.dump_dir.empty()) return;
        dump_on = true;
        dump_dir = opt.dump_dir;
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        std::vector<int> src_of((size_t)cfg.n_layers);
        for (int64_t i = 0; i < cfg.n_layers; ++i) src_of[(size_t)i] = (int)i;
        if (const MetaValue* sl = model.meta().get("strata.source_layers")) {
            if (sl->type == MetaType::ARRAY && (int64_t)sl->items.size() == cfg.n_layers) {
                for (int64_t i = 0; i < cfg.n_layers; ++i) {
                    if (sl->items[(size_t)i].is_num()) src_of[(size_t)i] = (int)sl->items[(size_t)i].u;
                }
            }
        }
        dump_src = src_of;
        dump_sel.assign((size_t)cfg.n_layers, 0);
        const std::string want = opt.dump_layers.empty() ? "all" : opt.dump_layers;
        if (want == "all") {
            std::fill(dump_sel.begin(), dump_sel.end(), 1);
        } else {
            std::stringstream ss(want);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) continue;
                const int id = std::atoi(item.c_str());
                bool hit = false;
                for (int64_t i = 0; i < cfg.n_layers; ++i)
                    if (src_of[(size_t)i] == id) {
                        dump_sel[(size_t)i] = 1;
                        hit = true;
                    }
                if (!hit) std::fprintf(stderr, "strata-minimax: --minimax-dump-layers %d is not in this file\n", id);
            }
        }
        for (int64_t i = 0; i < cfg.n_layers; ++i)
            if (dump_sel[(size_t)i]) dump_rows[(int)i] = {};
    }

    void flush_dumps() {
        if (!dump_on || dump_rows.empty()) return;
        for (auto& kv : dump_rows) {
            if (kv.second.empty()) continue;
            const int src = kv.first < (int)dump_src.size() ? dump_src[(size_t)kv.first] : kv.first;
            const std::string path = dump_dir + "/L" + std::to_string(src) + ".npy";
            write_npy_f32(path, kv.second.data(), (int64_t)(kv.second.size() / (size_t)cfg.n_embd), cfg.n_embd);
            std::fprintf(stderr, "INFO minimax-dump %s rows=%lld cols=%lld\n", path.c_str(),
                         (long long)(kv.second.size() / (size_t)cfg.n_embd), (long long)cfg.n_embd);
        }
        dump_rows.clear();
        dump_on = false;
    }

    void read_eos() {        if (const MetaValue* e = model.meta().get("tokenizer.ggml.eos_token_id"))
            if (e->is_num()) eos.insert((int32_t)e->u);
        if (const MetaValue* a = model.meta().get("tokenizer.ggml.eos_token_ids"))
            if (a->type == MetaType::ARRAY)
                for (const MetaValue& it : a->items)
                    if (it.is_num()) eos.insert((int32_t)it.u);
    }

    void alloc_cache() {
        // Host fp16 only. K and V are one buffer each; the indexer K is the third. A failed alloc names the
        // exact byte count so a 1M-context GGUF does not die as a bare bad_alloc.
        size_t layers_ctx = 0, kp = 0, ip = 0, elems = 0, bytes = 0;
        const bool fits = mul_size((size_t)cfg.n_layers, (size_t)ctx, layers_ctx) &&
                          mul_size(layers_ctx, (size_t)cfg.n_head_kv, kp) &&
                          mul_size(kp, (size_t)cfg.head_dim, kp) &&
                          mul_size(layers_ctx, (size_t)cfg.idx_dim, ip) &&
                          add_size(kp, kp, elems) && add_size(elems, ip, elems) &&
                          mul_size(elems, sizeof(uint16_t), bytes);
        if (!fits) {
            fail("failed to allocate the host fp16 KV cache for context " + std::to_string(ctx) +
                 ": the K, V, and indexer cache does not fit in the address space. Lower --max-context.");
        }
#if defined(_WIN32)
        // Refuse before touching pages. A commit larger than the process can back would otherwise zero
        // tens of gigabytes and grow the pagefile before bad_alloc.
        MEMORYSTATUSEX mem{};
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem) && mem.ullAvailPageFile < (DWORDLONG)bytes) {
            fail("failed to allocate the host fp16 KV cache for context " + std::to_string(ctx) + ": " +
                 std::to_string(bytes) + " bytes are required. Lower --max-context. available commit is " +
                 std::to_string((unsigned long long)mem.ullAvailPageFile) + " bytes.");
        }
#endif
        try {
            k_host.assign(kp, 0);
            v_host.assign(kp, 0);
            ik_host.assign(ip, 0);
        } catch (const std::exception& e) {
            fail("failed to allocate the host fp16 KV cache for context " + std::to_string(ctx) + ": " +
                 std::to_string(bytes) + " bytes are required. Lower --max-context. " + e.what());
        }
    }

    void size_workspace() {
        max_cols = cfg.n_embd;
        max_cols = bigger(max_cols, cfg.n_ff_dense);
        max_cols = bigger(max_cols, cfg.n_ff_exp * (cfg.n_shared_experts > 0 ? cfg.n_shared_experts : 1));
        max_cols = bigger(max_cols, cfg.n_ff_exp);
        max_cols = bigger(max_cols, cfg.n_head * cfg.head_dim);
        max_rows = cfg.n_vocab;
        max_rows = bigger(max_rows, cfg.n_head * cfg.head_dim);
        max_rows = bigger(max_rows, cfg.n_head_kv * cfg.head_dim);
        max_rows = bigger(max_rows, cfg.n_ff_dense);
        max_rows = bigger(max_rows, cfg.n_ff_exp * (cfg.n_shared_experts > 0 ? cfg.n_shared_experts : 1));
        max_rows = bigger(max_rows, cfg.n_expert);
        max_rows = bigger(max_rows, cfg.idx_heads * cfg.idx_dim);
    }

    void alloc_workspace() {
        d_x = DevPtr(cu::alloc((size_t)max_cols * sizeof(float)));
        d_y = DevPtr(cu::alloc((size_t)max_rows * sizeof(float)));
        d_scratch = DevPtr(cu::alloc((size_t)kTile * (size_t)max_cols * sizeof(float)));
    }

    void upload(Tensor& t) {
        const size_t n = (size_t)t.ne0 * (size_t)t.ne1 * (size_t)t.ne2;
        if (t.type == kF32) {
            t.dev = DevPtr(cu::alloc(t.bytes));
            cu::copy_h2d(t.dev.p, t.host, t.bytes);
            t.dev_f32 = true;
        } else if (t.type == kF16 || t.type == kBF16) {
            std::vector<float> tmp(n);
            decode_f32(t, t.host, tmp.data(), (int64_t)n);
            t.dev = DevPtr(cu::alloc(n * sizeof(float)));
            cu::copy_h2d(t.dev.p, tmp.data(), n * sizeof(float));
            t.dev_f32 = true;
        } else {
            t.dev = DevPtr(cu::alloc(t.bytes));
            cu::copy_h2d(t.dev.p, t.host, t.bytes);
            t.dev_f32 = false;
        }
        t.gpu = true;
    }

    size_t layer_cost(const Layer& L) const {
        size_t c = 0;
        auto add = [&](const Tensor& t) {
            if (t.host == nullptr) return;
            const size_t b = gpu_bytes(t);
            if (b != std::numeric_limits<size_t>::max()) c += b;
        };
        add(L.wq);
        add(L.wk);
        add(L.wv);
        add(L.wo);
        add(L.gate);
        add(L.up);
        add(L.down);
        add(L.router);
        add(L.iq);
        add(L.ik);
        return c;
    }

    void place_weights() {
        if (opt.dense_layers_gpu == 0) {
            dense_gpu = 0;
            output_gpu = false;
            return;
        }
        std::vector<size_t> cost((size_t)cfg.n_layers);
        for (int64_t i = 0; i < cfg.n_layers; ++i) cost[(size_t)i] = layer_cost(layers[(size_t)i]);
        const size_t out_b = gpu_bytes(output);
        const cu::MemInfo mi = cu::mem();
        const size_t reserve = (size_t)(opt.vram_reserve_mib > 0 ? opt.vram_reserve_mib : 0) << 20;
        const size_t budget = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
        const bool automatic = opt.dense_layers_gpu < 0;
        int N = automatic ? 0 : opt.dense_layers_gpu;
        if (N > (int)cfg.n_layers) N = (int)cfg.n_layers;
        if (automatic) {
            size_t used = 0;
            while (N < (int)cfg.n_layers && used + cost[(size_t)N] <= budget) {
                used += cost[(size_t)N];
                ++N;
            }
        } else if (N > 0) {
            size_t used = 0;
            for (int i = 0; i < N; ++i) used += cost[(size_t)i];
            if (used > budget) {
                fail("the first " + std::to_string(N) + " dense layers need " +
                     std::to_string(used / (1024 * 1024)) + " MiB of GPU weight storage but only " +
                     std::to_string(budget / (1024 * 1024)) + " MiB is free after KV and a " +
                     std::to_string(opt.vram_reserve_mib) + " MiB reserve. Lower --minimax-dense-layers-gpu.");
            }
        }
        dense_gpu = N;
        for (int i = 0; i < N; ++i) {
            Layer& L = layers[(size_t)i];
            L.want_gpu = true;
            auto up = [&](Tensor& t) {
                if (t.host && gpu_bytes(t) != std::numeric_limits<size_t>::max()) upload(t);
            };
            up(L.wq);
            up(L.wk);
            up(L.wv);
            up(L.wo);
            up(L.gate);
            up(L.up);
            up(L.down);
            up(L.router);
            up(L.iq);
            up(L.ik);
        }
        if (N > 0 && out_b != std::numeric_limits<size_t>::max()) {
            const cu::MemInfo now = cu::mem();
            if (now.free_bytes > out_b + ((size_t)256 << 20)) {
                upload(output);
                output_gpu = true;
            }
        }
    }

    // ===================================================================================== P3: the expert tiers

    static size_t q8_1_bytes(int64_t n) { return (size_t)(n / 32) * 36; }

    void init_tiers() {
        for (const Layer& L : layers)
            if (L.moe) {
                ++moe_layers;
                if (L.expert_bytes > max_expert_bytes) max_expert_bytes = L.expert_bytes;
            }
        if (moe_layers == 0 || max_expert_bytes == 0) {
            tiers_on = false;
            return;
        }
        tiers_on = true;
        const int64_t n_moe = moe_layers;
        if (!opt.profile_path.empty()) {
            std::string err;
            if (!read_strp(opt.profile_path, n_moe, cfg.n_expert, profile, err)) fail(err);
        }

        // ---- VRAM slots.  Auto sizing is free VRAM after dense weights, KV and workspace, less the reserve.
        const size_t reserve = (size_t)(opt.vram_reserve_mib > 0 ? opt.vram_reserve_mib : 0) << 20;
        size_t vram_budget = 0;
        if (tiers_gpu_possible()) {
            const cu::MemInfo mi = cu::mem();
            vram_budget = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
        }
        int q = opt.vram_slots_per_layer;
        if (q < 0) {
            q = vram_budget / ((size_t)moe_layers * max_expert_bytes);
            if (q > cfg.n_expert) q = (int)cfg.n_expert;
        }
        if (q < 0) q = 0;
        if (q > cfg.n_expert) q = (int)cfg.n_expert;
        if (q > 0) {
            if (!tiers_gpu_possible())
                fail("--minimax-vram-slots-per-layer needs a CUDA device (dense_layers_gpu=0 still starts CUDA "
                     "when the expert tiers are on)");
            size_t need = 0;
            for (const Layer& L : layers)
                if (L.moe) need += (size_t)q * L.expert_bytes;
            if (need > vram_budget) {
                fail("the VRAM expert tier asked for " + std::to_string(q) + " slots per layer (" +
                     std::to_string(need / (1024 * 1024)) + " MiB) but only " +
                     std::to_string(vram_budget / (1024 * 1024)) + " MiB is free after dense/KV/workspace and the " +
                     std::to_string(opt.vram_reserve_mib) + " MiB reserve. Lower --minimax-vram-slots-per-layer "
                     "or --vram-reserve-mib.");
            }
            for (Layer& L : layers) {
                if (!L.moe) continue;
                L.vram.reserve((size_t)q);
                for (int s = 0; s < q; ++s) {
                    Layer::VramSlot slot;
                    slot.mem = DevPtr(cu::alloc(L.expert_bytes));
                    L.vram.push_back(std::move(slot));
                }
            }
            const int top = (int)cfg.n_expert_used;
            d_ex_ptr = DevPtr(cu::alloc((size_t)top * 3 * sizeof(void*)));
            d_ex_xq = DevPtr(cu::alloc(q8_1_bytes(cfg.n_embd)));
            d_ex_hq = DevPtr(cu::alloc((size_t)top * q8_1_bytes(cfg.n_ff_exp)));
            d_ex_h = DevPtr(cu::alloc((size_t)top * (size_t)cfg.n_ff_exp * sizeof(float)));
            d_ex_out = DevPtr(cu::alloc((size_t)top * (size_t)cfg.n_embd * sizeof(float)));
            side_stream = cu::stream_create();
        }
        vram_per_layer = q;
        tiers_gpu = q > 0;
        for (Layer& L : layers)
            if (L.moe) L.vram_of.assign((size_t)cfg.n_expert, kNotResident);

        // ---- pinned RAM tier, auto = available RAM less a 2 GiB reserve, or --memory-limit-mib.
        size_t ram_budget = 0;
        if (opt.memory_limit_mib > 0) {
            ram_budget = (size_t)opt.memory_limit_mib << 20;
        } else {
            const size_t avail = host_available_bytes();
            const size_t keep = (size_t)2048 << 20;
            ram_budget = avail > keep ? avail - keep : 0;
        }
        int r = opt.ram_experts_per_layer;
        if (r < 0) {
            r = ram_budget / ((size_t)moe_layers * max_expert_bytes);
            if (r > cfg.n_expert) r = (int)cfg.n_expert;
        }
        if (r < 0) r = 0;
        if (r > cfg.n_expert) r = (int)cfg.n_expert;
        ram_per_layer = r;
        for (Layer& L : layers)
            if (L.moe) L.ram_of.assign((size_t)cfg.n_expert, kNotResident);
        if (r > 0) {
            uint64_t off = 0;
            for (Layer& L : layers) {
                if (!L.moe) continue;
                L.ram_off = off;
                L.ram_n = r;
                off += (uint64_t)r * L.expert_bytes;
            }
            ram_bytes = off;
            ram_arena = std::make_unique<strata::core::PinnedArena>(ram_bytes);
            if (!ram_arena->valid()) {
                fail("the pinned RAM expert tier could not reserve " + std::to_string(ram_bytes / (1024 * 1024)) +
                     " MiB (" + ram_arena->note + "). Lower --memory-limit-mib or "
                     "--minimax-ram-experts-per-layer.");
            }
            for (Layer& L : layers) {
                if (!L.moe) continue;
                L.ram_of.assign((size_t)cfg.n_expert, kNotResident);
                L.ram_expert_of.assign((size_t)r, kNotResident);
                L.ram_tick.assign((size_t)r, 0);
            }
        }

        // ---- static fill from the profile: each layer's hottest q go to VRAM, the next r to RAM.
        if (!profile.empty()) {
            std::vector<int> rank((size_t)(n_moe * cfg.n_expert), -1);
            for (size_t i = 0; i < profile.size(); ++i) {
                const auto [pl, pe] = profile[i];
                rank[(size_t)(pl * cfg.n_expert + pe)] = (int)i;
            }
            for (int64_t li = 0; li < cfg.n_layers; ++li) {
                Layer& L = layers[(size_t)li];
                if (!L.moe) continue;
                const int ml = (int)(li - cfg.leading_dense);
                std::vector<std::pair<int, int>> order;
                for (int e = 0; e < (int)cfg.n_expert; ++e) {
                    const int rr = rank[(size_t)(ml * cfg.n_expert + e)];
                    if (rr >= 0) order.push_back({rr, e});
                }
                std::sort(order.begin(), order.end());
                int at = 0;
                for (int s = 0; s < vram_per_layer && at < (int)order.size(); ++s, ++at)
                    vram_fill(L, s, order[(size_t)at].second);
                for (int s = 0; s < ram_per_layer && at < (int)order.size(); ++s, ++at)
                    ram_fill(L, s, order[(size_t)at].second);
            }
        }

        // ---- SSD: unbuffered reads straight from the GGUF shards.  DirectFile owns the aligned OVERLAPPED /
        // O_DIRECT I/O; this keeps the offsets and the aligned superset rule.
        ssd_files.resize(model.size());
        for (size_t s = 0; s < model.size(); ++s) {
            ssd_files[s] = std::make_unique<strata::platform::DirectFile>();
            std::string err;
            if (!ssd_files[s]->open(model.shard(s).path(), err))
                fail("minimax SSD tier: " + err);
        }
        // One staging slot per routed entry: several selected experts of one layer may be SSD misses, and their
        // byte slices have to stay alive together until the combine.
        ssd_stage_bytes = (size_t)cfg.n_expert_used * max_expert_bytes;
        ssd_stage = (uint8_t*)strata::platform::DirectFile::alloc_aligned(ssd_stage_bytes);
        if (!ssd_stage) fail("minimax SSD tier: cannot allocate the 4 KiB-aligned staging buffer");
        ssd_tmp = strata::platform::DirectFile::alloc_aligned(ssd_stage_bytes + 4096);
        if (!ssd_tmp) fail("minimax SSD tier: cannot allocate the 4 KiB-aligned read buffer");
        ssd_tmp_bytes = ssd_stage_bytes + 4096;

        const size_t cells = (size_t)cfg.n_layers * (size_t)cfg.n_expert;
        if (opt.adaptive) usage.assign(cells, 0.f);
        if (!opt.profile_save_path.empty() || !opt.routing_counts_path.empty()) route_counts.assign(cells, 0.0);
    }

    bool tiers_gpu_possible() const { return opt.dense_layers_gpu != 0 || opt.tiers; }

    void vram_fill(Layer& L, int slot, int expert) {
        if (slot < 0 || slot >= (int)L.vram.size()) fail("VRAM slot out of range");
        Layer::VramSlot& s = L.vram[(size_t)slot];
        if (s.expert >= 0) L.vram_of[(size_t)s.expert] = kNotResident;
        uint8_t* p = (uint8_t*)s.mem.p;
        cu::copy_h2d(p, L.egate.host + (size_t)expert * L.expert_gu, L.expert_gu);
        cu::copy_h2d(p + L.expert_gu, L.eup.host + (size_t)expert * L.expert_gu, L.expert_gu);
        cu::copy_h2d(p + 2 * L.expert_gu, L.edown.host + (size_t)expert * L.expert_down, L.expert_down);
        s.expert = expert;
        L.vram_of[(size_t)expert] = slot;
    }

    void vram_swap(Layer& L, int slot, int expert) {
        Layer::VramSlot& s = L.vram[(size_t)slot];
        uint8_t* p = (uint8_t*)s.mem.p;
        cu::copy_h2d_async(p, L.egate.host + (size_t)expert * L.expert_gu, L.expert_gu, side_stream);
        cu::copy_h2d_async(p + L.expert_gu, L.eup.host + (size_t)expert * L.expert_gu, L.expert_gu, side_stream);
        cu::copy_h2d_async(p + 2 * L.expert_gu, L.edown.host + (size_t)expert * L.expert_down, L.expert_down,
                           side_stream);
        if (s.expert >= 0) L.vram_of[(size_t)s.expert] = kNotResident;
        s.expert = expert;
        L.vram_of[(size_t)expert] = slot;
    }

    uint8_t* ram_slot_ptr(Layer& L, int slot) const {
        return ram_arena->data() + L.ram_off + (uint64_t)slot * L.expert_bytes;
    }

    void ram_fill(Layer& L, int slot, int expert) {
        if (!ram_arena || !ram_arena->valid()) fail("RAM tier is not allocated");
        uint8_t* p = ram_slot_ptr(L, slot);
        std::memcpy(p, L.egate.host + (size_t)expert * L.expert_gu, L.expert_gu);
        std::memcpy(p + L.expert_gu, L.eup.host + (size_t)expert * L.expert_gu, L.expert_gu);
        std::memcpy(p + 2 * L.expert_gu, L.edown.host + (size_t)expert * L.expert_down, L.expert_down);
        for (int32_t& v : L.ram_of)
            if (v == slot) v = kNotResident;
        L.ram_of[(size_t)expert] = slot;
        L.ram_expert_of[(size_t)slot] = expert;
        L.ram_tick[(size_t)slot] = ++lru_tick;
    }

    // Copy an expert that was just read from the SSD into the least-recently-used RAM slot.  The bytes are already
    // in `src`; a later SSD miss for this expert then costs a RAM read instead.
    void ram_admit(Layer& L, int expert, const uint8_t* src) {
        if (!opt.adaptive || !ram_arena || L.ram_n <= 0) return;
        int victim = 0;
        for (int s = 1; s < L.ram_n; ++s)
            if (L.ram_tick[(size_t)s] < L.ram_tick[(size_t)victim]) victim = s;
        uint8_t* p = ram_slot_ptr(L, victim);
        std::memcpy(p, src, L.expert_bytes);
        for (int32_t& v : L.ram_of)
            if (v == victim) v = kNotResident;
        L.ram_of[(size_t)expert] = victim;
        L.ram_expert_of[(size_t)victim] = expert;
        L.ram_tick[(size_t)victim] = ++lru_tick;
    }

    void ssd_read_role(const Tensor& t, int64_t expert, uint8_t* dst) {
        const size_t len = t.bytes / (size_t)t.ne2;
        const uint64_t off = t.file_offset + (uint64_t)expert * (uint64_t)len;
        constexpr uint64_t kAlign = 4096;
        const uint64_t aoff = off & ~(kAlign - 1);
        const uint64_t aend = (off + len + kAlign - 1) & ~(kAlign - 1);
        const size_t alen = (size_t)(aend - aoff);
        strata::platform::DirectFile& file = *ssd_files[t.shard];
        std::string err;
        if (!file.submit(aoff, ssd_tmp, (uint32_t)alen, 1, err)) fail("minimax SSD read: " + err);
        strata::platform::Completion c{};
        while (file.wait(&c, 1, -1) == 0) {
        }
        if (!c.ok) fail("minimax SSD read: I/O failed at " + std::to_string(off));
        if ((uint64_t)c.bytes < (off - aoff) + len)
            fail("minimax SSD read: short read at " + std::to_string(off));
        std::memcpy(dst, (const uint8_t*)ssd_tmp + (size_t)(off - aoff), len);
        if (std::getenv("STRATA_MINIMAX_VERIFY_SSD") != nullptr) {
            const uint8_t* want = t.host + (size_t)expert * len;
            size_t bad = 0;
            while (bad < len && dst[bad] == want[bad]) ++bad;
            if (bad != len)
                std::fprintf(stderr, "SSD MISMATCH %s expert %lld at byte %zu (file offset %llu, aligned %llu)\n",
                             t.name.c_str(), (long long)expert, bad, (unsigned long long)off,
                             (unsigned long long)aoff);
        }
    }

    void ssd_read_expert(const Layer& L, int64_t expert, uint8_t* stage) {
        ssd_read_role(L.egate, expert, stage);
        ssd_read_role(L.eup, expert, stage + L.expert_gu);
        ssd_read_role(L.edown, expert, stage + 2 * L.expert_gu);
        ++tok_stats.ssd;
        tok_stats.ssd_bytes += (int64_t)L.expert_bytes;
    }

    void gpu_experts(const Layer& L, const std::vector<int>& exps, const float* in, std::vector<float>& out) {
        const int k = (int)exps.size();
        if (k == 0) return;
        std::vector<const void*> gp((size_t)k), up((size_t)k), dp((size_t)k);
        for (int i = 0; i < k; ++i) {
            const int e = exps[(size_t)i];
            const int slot = L.vram_of[(size_t)e];
            if (slot < 0) fail("internal: a GPU expert has no VRAM slot");
            const uint8_t* base = (const uint8_t*)L.vram[(size_t)slot].mem.p;
            gp[(size_t)i] = base;
            up[(size_t)i] = base + L.expert_gu;
            dp[(size_t)i] = base + 2 * L.expert_gu;
        }
        const int top = (int)cfg.n_expert_used;
        void** ptr = (void**)d_ex_ptr.p;
        cu::copy_h2d(ptr, gp.data(), (size_t)k * sizeof(void*));
        cu::copy_h2d(ptr + top, up.data(), (size_t)k * sizeof(void*));
        cu::copy_h2d(ptr + 2 * top, dp.data(), (size_t)k * sizeof(void*));
        const int n_embd = (int)cfg.n_embd, n_ff = (int)cfg.n_ff_exp;
        const int gt = L.egate.type, dt = L.edown.type;
        const bool iq = iq_expert_type(gt) && iq_expert_type(dt);
        const bool raw = raw_expert_type(gt) && raw_expert_type(dt);
        // MM-NVFP4: per-selected-expert divisors, applied once inside the kernel (gate/up before swigluoai).
        const bool nv = !L.egate_scale.empty() || !L.edown_scale.empty();
        const float* sg = nullptr;
        const float* su = nullptr;
        const float* sd = nullptr;
        if (nv) {
            if ((int)L.egate_scale.size() != (int)cfg.n_expert) fail("internal: expert scale vector is short");
            std::vector<float> a((size_t)k), b((size_t)k), c((size_t)k);
            for (int i = 0; i < k; ++i) {
                const int e = exps[(size_t)i];
                a[(size_t)i] = L.egate_scale[(size_t)e];
                b[(size_t)i] = L.eup_scale[(size_t)e];
                c[(size_t)i] = L.edown_scale[(size_t)e];
            }
            const size_t bytes = (size_t)cfg.n_expert_used * sizeof(float);
            if (!d_ex_sg.p) d_ex_sg = DevPtr(cu::alloc(bytes));
            if (!d_ex_su.p) d_ex_su = DevPtr(cu::alloc(bytes));
            if (!d_ex_sd.p) d_ex_sd = DevPtr(cu::alloc(bytes));
            cu::copy_h2d(d_ex_sg.p, a.data(), (size_t)k * sizeof(float));
            cu::copy_h2d(d_ex_su.p, b.data(), (size_t)k * sizeof(float));
            cu::copy_h2d(d_ex_sd.p, c.data(), (size_t)k * sizeof(float));
            sg = (const float*)d_ex_sg.p;
            su = (const float*)d_ex_su.p;
            sd = (const float*)d_ex_sd.p;
        }
        if (iq) {
            if (cfg.n_embd % 32 != 0 || cfg.n_ff_exp % 32 != 0)
                fail("the GPU IQ expert path needs rounded 32-wide activations");
            cu::copy_h2d(d_x.p, in, (size_t)n_embd * sizeof(float));
            strata::kernels::quantize_q8_1_rows((const float*)d_x.p, 1, n_embd, d_ex_xq.p, nullptr);
            strata::kernels::minimax_moe_gu(gt, (const void* const*)ptr, (const void* const*)(ptr + top), k,
                                            n_embd, n_ff, L.egate.row_bytes, d_ex_xq.p, (float*)d_ex_h.p, alpha, limit,
                                            nullptr, sg, su);
            strata::kernels::quantize_q8_1_rows((const float*)d_ex_h.p, k, n_ff, d_ex_hq.p, nullptr);
            strata::kernels::minimax_moe_down(dt, (const void* const*)(ptr + 2 * top), k, n_embd, n_ff,
                                              L.edown.row_bytes, d_ex_hq.p, (float*)d_ex_out.p, nullptr, sd);
            cu::sync();
        } else if (raw) {
            if (gt != dt)
                fail("the GPU F32/F16/BF16 expert path needs the same type for gate/up and down");
            cu::copy_h2d(d_x.p, in, (size_t)n_embd * sizeof(float));
            cu::moe_expert_f16(gt, (const void* const*)ptr, (const void* const*)(ptr + top),
                               (const void* const*)(ptr + 2 * top), k, n_embd, n_ff, L.egate.row_bytes,
                               L.edown.row_bytes, (const float*)d_x.p, (float*)d_ex_h.p, (float*)d_ex_out.p, alpha,
                               limit);
        } else {
            fail("no GPU expert kernel for gate/up ggml type " + std::to_string(gt) + " and down type " +
                 std::to_string(dt) + " (supported: F32/F16/BF16 and the IQ/Q8_0 types)");
        }
        out.resize((size_t)k * (size_t)n_embd);
        cu::copy_d2h(out.data(), d_ex_out.p, out.size() * sizeof(float));
    }

    void moe_tiered(const Layer& L, const float* in, float* yout) {
        // The router and the combine are the CPU path's, unchanged.  Only where each selected expert's bytes come
        // from (VRAM / pinned RAM / SSD) and who computes it change.
        const int nexp = (int)cfg.n_expert;
        const int top = (int)cfg.n_expert_used;
        std::vector<float> logits_r((size_t)nexp), prob((size_t)nexp), score((size_t)nexp);
        gemv(L.router, in, logits_r.data());
        for (int i = 0; i < nexp; ++i) {
            prob[(size_t)i] = sigmoid(logits_r[(size_t)i]);
            score[(size_t)i] = prob[(size_t)i] + L.bias[(size_t)i];
        }
        std::vector<int> order(nexp);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
            if (score[(size_t)a] != score[(size_t)b]) return score[(size_t)a] > score[(size_t)b];
            return a < b;
        });
        std::vector<float> w((size_t)top);
        float sum = 0.f;
        for (int j = 0; j < top; ++j) {
            w[(size_t)j] = prob[(size_t)order[(size_t)j]];
            sum += w[(size_t)j];
        }
        if (renorm && sum > 0.f)
            for (float& v : w) v /= sum;

        const int64_t li = (int64_t)(&L - layers.data());
        const size_t base_idx = (size_t)li * (size_t)cfg.n_expert;
        if (!usage.empty()) {
            for (int j = 0; j < top; ++j) usage[base_idx + (size_t)order[(size_t)j]] += 1.f;
        }
        if (!route_counts.empty()) {
            for (int j = 0; j < top; ++j) route_counts[base_idx + (size_t)order[(size_t)j]] += 1.0;
        }

        std::vector<int> gpu_e;
        struct CpuExpert {
            int e;
            const uint8_t* gate;
            const uint8_t* up;
            const uint8_t* down;
        };
        std::vector<CpuExpert> cpu;
        std::vector<std::pair<int, const uint8_t*>> ssd_admits;
        gpu_e.reserve((size_t)top);
        cpu.reserve((size_t)top);
        int ssd_n = 0;
        for (int j = 0; j < top; ++j) {
            const int e = order[(size_t)j];
            // Compulsory admission when no profile filled the slots: the first distinct experts a layer routes take
            // the free slots.  A profile-filled tier has no free slot and leaves this a pure lookup.
            if (!profile.empty() || L.vram_of[(size_t)e] >= 0) {
                // lookup only
            } else {
                for (int s = 0; s < (int)L.vram.size(); ++s) {
                    if (L.vram[(size_t)s].expert < 0) {
                        vram_fill(const_cast<Layer&>(L), s, e);
                        break;
                    }
                }
                if (L.vram_of[(size_t)e] < 0 && L.ram_n > 0 && L.ram_of[(size_t)e] < 0) {
                    for (int s = 0; s < L.ram_n; ++s)
                        if (L.ram_expert_of[(size_t)s] < 0) {
                            ram_fill(const_cast<Layer&>(L), s, e);
                            break;
                        }
                }
            }
            const int vs = L.vram_of[(size_t)e];
            if (vs >= 0) {
                gpu_e.push_back(e);
                ++tok_stats.vram;
                continue;
            }
            const int rs = L.ram_of[(size_t)e];
            if (rs >= 0) {
                const uint8_t* p = ram_slot_ptr(const_cast<Layer&>(L), rs);
                cpu.push_back({e, p, p + L.expert_gu, p + 2 * L.expert_gu});
                ++tok_stats.ram;
                continue;
            }
            uint8_t* stage = ssd_stage + (size_t)ssd_n++ * max_expert_bytes;
            ssd_read_expert(L, e, stage);
            cpu.push_back({e, stage, stage + L.expert_gu, stage + 2 * L.expert_gu});
            // Deferred: admitting (and evicting) now could overwrite a RAM slot another entry of this same
            // token is still going to read.
            ssd_admits.push_back({e, stage});
        }

        std::vector<float> gpu_out;
        if (!gpu_e.empty()) gpu_experts(L, gpu_e, in, gpu_out);

        std::fill(yout, yout + cfg.n_embd, 0.f);
        std::vector<float> ye((size_t)cfg.n_embd);
        std::vector<const CpuExpert*> cpu_of((size_t)cfg.n_expert, nullptr);
        for (const CpuExpert& c : cpu) cpu_of[(size_t)c.e] = &c;
        size_t gi = 0;
        for (int j = 0; j < top; ++j) {
            const int e = order[(size_t)j];
            if (L.vram_of[(size_t)e] >= 0) {
                std::memcpy(ye.data(), gpu_out.data() + gi * (size_t)cfg.n_embd,
                            (size_t)cfg.n_embd * sizeof(float));
                ++gi;
            } else {
                const CpuExpert* c = cpu_of[(size_t)e];
                if (!c) fail("internal: routed expert has no source");
                const float* sg = L.egate_scale.empty() ? nullptr : &L.egate_scale[(size_t)e];
                const float* su = L.eup_scale.empty() ? nullptr : &L.eup_scale[(size_t)e];
                const float* sd = L.edown_scale.empty() ? nullptr : &L.edown_scale[(size_t)e];
                mlp_ptr(L.egate, c->gate, L.eup, c->up, L.edown, c->down, in, ye.data(), (int)cfg.n_ff_exp, sg, su, sd);
            }
            const float WJ = w[(size_t)j];
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += WJ * ye[(size_t)i];
        }
        const float s = scale_routed;
        for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] *= s;
        if (L.have_shared) {
            mlp(L.gate, L.up, L.down, 0, in, ye.data(), (int)L.gate.ne1);
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += ye[(size_t)i];
        }
        for (const auto& a : ssd_admits) ram_admit(const_cast<Layer&>(L), a.first, a.second);
        if (opt.stats_per_layer) {
            std::fprintf(stderr, "INFO minimax-stats layer=%lld vram=%lld ram=%lld ssd=%lld ssd_bytes=%lld\n",
                         (long long)li, (long long)tok_stats.vram, (long long)tok_stats.ram,
                         (long long)tok_stats.ssd, (long long)tok_stats.ssd_bytes);
        }
    }

    void rebalance() {
        if (!tiers_gpu || !opt.adaptive || usage.empty()) return;
        bool any = false;
        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            if (!L.moe || L.vram.empty()) continue;
            const size_t b = (size_t)li * (size_t)cfg.n_expert;
            int best = -1, worst = -1;
            float best_u = -1.f, worst_u = std::numeric_limits<float>::max();
            for (int e = 0; e < (int)cfg.n_expert; ++e) {
                if (L.vram_of[(size_t)e] < 0) {
                    if (usage[b + (size_t)e] > best_u) {
                        best_u = usage[b + (size_t)e];
                        best = e;
                    }
                } else if (usage[b + (size_t)e] < worst_u) {
                    worst_u = usage[b + (size_t)e];
                    worst = e;
                }
            }
            if (best < 0 || worst < 0) continue;
            if (best_u > worst_u * 1.10f + 0.5f) {
                vram_swap(L, L.vram_of[(size_t)worst], best);
                ++swaps;
                any = true;
            }
        }
        if (any) cu::stream_sync(side_stream);
    }

    void decay_usage() {
        if (!opt.adaptive || usage.empty()) return;
        for (float& v : usage) v *= 0.98f;
    }

    void write_own_profile() {
        if (opt.profile_save_path.empty() && opt.routing_counts_path.empty()) return;
        if (route_counts.empty()) return;
        const int64_t n_moe = moe_layers;
        std::vector<std::pair<int32_t, int32_t>> ranked;
        ranked.reserve((size_t)(n_moe * cfg.n_expert));
        for (int64_t ml = 0; ml < n_moe; ++ml)
            for (int64_t e = 0; e < cfg.n_expert; ++e) ranked.push_back({(int32_t)ml, (int32_t)e});
        const int64_t model_l0 = cfg.leading_dense;
        auto count_of = [&](int32_t ml, int32_t e) {
            return route_counts[(size_t)((model_l0 + ml) * cfg.n_expert + e)];
        };
        std::stable_sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
            const double ca = count_of(a.first, a.second), cb = count_of(b.first, b.second);
            if (ca != cb) return ca > cb;
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        std::string err;
        if (!opt.profile_save_path.empty() && !write_strp(opt.profile_save_path, n_moe, cfg.n_expert, ranked, err))
            std::fprintf(stderr, "strata-minimax: %s\n", err.c_str());
        if (!opt.routing_counts_path.empty()) {
            std::FILE* f = std::fopen(opt.routing_counts_path.c_str(), "wb");
            if (!f) {
                std::fprintf(stderr, "strata-minimax: cannot write %s\n", opt.routing_counts_path.c_str());
            } else {
                for (int64_t ml = 0; ml < n_moe; ++ml)
                    for (int64_t e = 0; e < cfg.n_expert; ++e)
                        std::fprintf(f, "%lld %lld %.0f\n", (long long)ml, (long long)e, count_of((int32_t)ml, (int32_t)e));
                std::fclose(f);
            }
        }
        ++profile_saves;
    }

    void gemv(const Tensor& t, const float* xin, float* yout) {
        if (t.ne2 != 1) fail(t.name + ": internal gemv is for a matrix, not a stacked expert");
        const int rows = (int)t.ne1;
        const int cols = (int)t.ne0;
        if ((int64_t)cols > max_cols || (int64_t)rows > max_rows) fail(t.name + ": gemv exceeds the workspace");
        if (t.gpu) {
            cu::copy_h2d(d_x.p, xin, (size_t)cols * sizeof(float));
            float* dy = (float*)d_y.p;
            if (t.dev_f32) {
                cu::gemv((const float*)t.dev.p, (const float*)d_x.p, dy, rows, cols);
            } else {
                for (int r = 0; r < rows; r += kTile) {
                    const int n = rows - r < kTile ? rows - r : kTile;
                    cu::dequant_f32(t.type, t.dev.p, r, n, cols, (float*)d_scratch.p);
                    cu::gemv((const float*)d_scratch.p, (const float*)d_x.p, dy + r, n, cols);
                }
            }
            cu::copy_d2h(yout, dy, (size_t)rows * sizeof(float));
            return;
        }
        if (t.type == kF32 || t.type == kF16 || t.type == kBF16) {
            pool.run(rows, [&](int a, int b) {
                for (int r = a; r < b; ++r) {
                    const uint8_t* row = t.host + (size_t)r * t.row_bytes;
                    if (t.type == kF32) yout[r] = dot_f32((const float*)row, xin, cols);
                    else if (t.type == kF16) yout[r] = dot_f16w((const uint16_t*)row, xin, cols);
                    else yout[r] = dot_bf16w((const uint16_t*)row, xin, cols);
                }
            });
            return;
        }
        if (!t.cpu || !t.cpu->vec_dot) fail(t.name + ": ggml-cpu has no vec_dot for type " + std::to_string(t.type));
        const ggml_type_traits_cpu* act = ggml_get_type_traits_cpu(t.cpu->vec_dot_type);
        if (!act || !act->from_float) fail(t.name + ": ggml-cpu has no from_float for the activation type");
        const size_t act_bytes = ggml_row_size(t.cpu->vec_dot_type, cols);
        std::vector<uint8_t> aq(act_bytes);
        act->from_float(xin, aq.data(), cols);
        const ggml_vec_dot_t dot = t.cpu->vec_dot;
        pool.run(rows, [&](int a, int b) {
            for (int r = a; r < b; ++r) {
                float s = 0.f;
                dot(cols, &s, 0, t.host + (size_t)r * t.row_bytes, 0, aq.data(), 0, 1);
                yout[r] = s;
            }
        });
    }

    void dot_rows_base(const Tensor& t, const uint8_t* base, const float* xin, float* yout, int rows_n, int cols) {
        // One expert's rows start at `base`: yout[r] = row(r) . xin.  `base` may be the GGUF mapping, the pinned
        // RAM arena, or an SSD staging buffer; only t's type, row size and row count are used.
        if (t.type == kF32 || t.type == kF16 || t.type == kBF16) {
            pool.run(rows_n, [&](int a, int b) {
                for (int r = a; r < b; ++r) {
                    const uint8_t* row = base + (size_t)r * t.row_bytes;
                    if (t.type == kF32) yout[r] = dot_f32((const float*)row, xin, cols);
                    else if (t.type == kF16) yout[r] = dot_f16w((const uint16_t*)row, xin, cols);
                    else yout[r] = dot_bf16w((const uint16_t*)row, xin, cols);
                }
            });
            return;
        }
        if (!t.cpu || !t.cpu->vec_dot) fail(t.name + ": ggml-cpu has no vec_dot for type " + std::to_string(t.type));
        const ggml_type_traits_cpu* act = ggml_get_type_traits_cpu(t.cpu->vec_dot_type);
        if (!act || !act->from_float) fail(t.name + ": no from_float for the expert activation");
        std::vector<uint8_t> aq(ggml_row_size(t.cpu->vec_dot_type, cols));
        act->from_float(xin, aq.data(), cols);
        const ggml_vec_dot_t dot = t.cpu->vec_dot;
        pool.run(rows_n, [&](int a, int b) {
            for (int r = a; r < b; ++r) {
                float s = 0.f;
                dot(cols, &s, 0, base + (size_t)r * t.row_bytes, 0, aq.data(), 0, 1);
                yout[r] = s;
            }
        });
    }

    void dot_rows(const Tensor& t, int64_t expert, const float* xin, float* yout, int rows_n, int cols) {
        dot_rows_base(t, row_ptr(t, expert, 0), xin, yout, rows_n, cols);
    }

    // A plain matrix that was uploaded uses the device GEMV. Stacked experts (ne2 > 1) stay on ggml-cpu,
    // including when the caller passes expert 0, so swigluoai never goes through the Qwen SiLU pool.
    void apply(const Tensor& t, int64_t expert, const float* xin, float* yout, int rows_n, int cols) {
        if (expert == 0 && t.ne2 == 1 && t.gpu) {
            gemv(t, xin, yout);
            return;
        }
        dot_rows(t, expert, xin, yout, rows_n, cols);
    }

    void mlp(const Tensor& gate, const Tensor& up, const Tensor& down, int64_t expert, const float* in, float* yout,
             int ff, const float* sg = nullptr, const float* su = nullptr, const float* sd = nullptr) {
        if ((int)ff_g.size() < ff) fail("MLP wider than the workspace");
        apply(gate, expert, in, ff_g.data(), ff, (int)gate.ne0);
        if (sg)
            for (int i = 0; i < ff; ++i) ff_g[(size_t)i] *= *sg;
        apply(up, expert, in, ff_u.data(), ff, (int)up.ne0);
        if (su)
            for (int i = 0; i < ff; ++i) ff_u[(size_t)i] *= *su;
        for (int i = 0; i < ff; ++i) ff_h[i] = swigluoai(ff_g[i], ff_u[i], alpha, limit);
        apply(down, expert, ff_h.data(), yout, (int)down.ne1, ff);
        if (sd)
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] *= *sd;
    }

    // The same MLP from three explicit byte slices (a RAM arena slot or an SSD staging buffer).
    void mlp_ptr(const Tensor& gate, const uint8_t* gb, const Tensor& up, const uint8_t* ub, const Tensor& down,
                 const uint8_t* db, const float* in, float* yout, int ff, const float* sg = nullptr,
                 const float* su = nullptr, const float* sd = nullptr) {
        if ((int)ff_g.size() < ff) fail("MLP wider than the workspace");
        dot_rows_base(gate, gb, in, ff_g.data(), ff, (int)gate.ne0);
        if (sg)
            for (int i = 0; i < ff; ++i) ff_g[(size_t)i] *= *sg;
        dot_rows_base(up, ub, in, ff_u.data(), ff, (int)up.ne0);
        if (su)
            for (int i = 0; i < ff; ++i) ff_u[(size_t)i] *= *su;
        for (int i = 0; i < ff; ++i) ff_h[i] = swigluoai(ff_g[i], ff_u[i], alpha, limit);
        dot_rows_base(down, db, ff_h.data(), yout, (int)down.ne1, ff);
        if (sd)
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] *= *sd;
    }

    void moe(const Layer& L, const float* in, float* yout) {
        if (tiers_on) {
            moe_tiered(L, in, yout);
            return;
        }
        const int nexp = (int)cfg.n_expert;
        const int top = (int)cfg.n_expert_used;
        std::vector<float> logits_r((size_t)nexp), prob((size_t)nexp), score((size_t)nexp);
        gemv(L.router, in, logits_r.data());
        for (int i = 0; i < nexp; ++i) {
            prob[(size_t)i] = sigmoid(logits_r[(size_t)i]);
            score[(size_t)i] = prob[(size_t)i] + L.bias[(size_t)i];
        }
        std::vector<int> order(nexp);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
            if (score[(size_t)a] != score[(size_t)b]) return score[(size_t)a] > score[(size_t)b];
            return a < b;
        });
        std::vector<float> w((size_t)top);
        float sum = 0.f;
        for (int j = 0; j < top; ++j) {
            w[(size_t)j] = prob[(size_t)order[(size_t)j]];
            sum += w[(size_t)j];
        }
        if (renorm && sum > 0.f)
            for (float& v : w) v /= sum;
        if (!route_counts.empty()) {
            const size_t b = (size_t)(&L - layers.data()) * (size_t)cfg.n_expert;
            for (int j = 0; j < top; ++j) route_counts[b + (size_t)order[(size_t)j]] += 1.0;
        }
        std::fill(yout, yout + cfg.n_embd, 0.f);
        std::vector<float> ye((size_t)cfg.n_embd);
        const int ff = (int)cfg.n_ff_exp;
        for (int j = 0; j < top; ++j) {
            const int e = order[(size_t)j];
            const float* sg = L.egate_scale.empty() ? nullptr : &L.egate_scale[(size_t)e];
            const float* su = L.eup_scale.empty() ? nullptr : &L.eup_scale[(size_t)e];
            const float* sd = L.edown_scale.empty() ? nullptr : &L.edown_scale[(size_t)e];
            mlp(L.egate, L.eup, L.edown, e, in, ye.data(), ff, sg, su, sd);
            const float WJ = w[(size_t)j];
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += WJ * ye[(size_t)i];
        }
        const float s = scale_routed;
        for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] *= s;
        if (L.have_shared) {
            mlp(L.gate, L.up, L.down, 0, in, ye.data(), (int)L.gate.ne1);
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += ye[(size_t)i];
        }
    }

    const uint16_t* k_at(int layer, int64_t pos, int kv) const {
        const size_t stride = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        return k_host.data() + ((size_t)layer * (size_t)ctx + (size_t)pos) * stride + (size_t)kv * (size_t)cfg.head_dim;
    }
    const uint16_t* v_at(int layer, int64_t pos, int kv) const {
        const size_t stride = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        return v_host.data() + ((size_t)layer * (size_t)ctx + (size_t)pos) * stride + (size_t)kv * (size_t)cfg.head_dim;
    }
    const uint16_t* ik_at(int layer, int64_t pos) const {
        return ik_host.data() + ((size_t)layer * (size_t)ctx + (size_t)pos) * (size_t)cfg.idx_dim;
    }

    void store_f16(std::vector<uint16_t>& host, size_t off, const float* src, int n) {
        for (int i = 0; i < n; ++i) host[off + (size_t)i] = ggml_fp32_to_fp16(src[i]);
    }

    void attend(int layer, const Layer& L, int64_t pos, float* att) {
        const int H = (int)cfg.n_head;
        const int HKV = (int)cfg.n_head_kv;
        const int D = (int)cfg.head_dim;
        const int64_t Tk = pos + 1;
        const int q_per_kv = H / HKV;
        const float scale = 1.f / std::sqrt((float)D);
        std::vector<char> sel;
        int nb = 0;
        if (L.indexer) {
            const int HI = (int)cfg.idx_heads;
            const int BLK = (int)cfg.idx_block;
            nb = (int)((Tk + BLK - 1) / BLK);
            std::vector<float> bs((size_t)HI * nb, -std::numeric_limits<float>::infinity());
            for (int h = 0; h < HI; ++h) {
                const float* qh = iq.data() + (size_t)h * cfg.idx_dim;
                for (int b = 0; b < nb; ++b) {
                    const int64_t begin = (int64_t)b * BLK;
                    const int64_t end = begin + BLK < Tk ? begin + BLK : Tk;
                    float m = -std::numeric_limits<float>::infinity();
                    for (int64_t t = begin; t < end; ++t) {
                        if (t > pos) break;
                        const float sc = dot_f16_cache(ik_at(layer, t), qh, (int)cfg.idx_dim);
                        if (sc > m) m = sc;
                    }
                    bs[(size_t)h * nb + b] = m;
                }
                const int qb = (int)(pos / BLK);
                for (int l = 0; l < (int)cfg.idx_local; ++l) {
                    int bi = qb - l;
                    if (bi < 0) bi = 0;
                    if (bi < nb) bs[(size_t)h * nb + bi] = std::numeric_limits<float>::infinity();
                }
            }
            const int kk = (int)cfg.idx_top_k < nb ? (int)cfg.idx_top_k : nb;
            sel.assign((size_t)HI * nb, 0);
            for (int h = 0; h < HI; ++h) {
                std::vector<int> order(nb);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int a, int b) {
                    const float sa = bs[(size_t)h * nb + a];
                    const float sb = bs[(size_t)h * nb + b];
                    if (sa != sb) return sa > sb;
                    return a < b;
                });
                for (int i = 0; i < kk; ++i) {
                    const int b = order[(size_t)i];
                    if (bs[(size_t)h * nb + b] == -std::numeric_limits<float>::infinity()) continue;
                    sel[(size_t)h * nb + b] = 1;
                }
            }
        }
        const int q_per_idx = L.indexer ? H / (int)cfg.idx_heads : H;
        // One slot list per indexer head: the selected blocks, cut at `pos` inside the current block.
        // Heads that share an indexer head reuse that list. 16 blocks of 128 is 2048 slots; the vector
        // grows if a file selects more, so the mask is not truncated.
        const int HI = L.indexer ? (int)cfg.idx_heads : 1;
        const int BLK = (int)cfg.idx_block;
        std::vector<int32_t> slots;
        std::vector<int> slot_off((size_t)HI + 1, 0);
        if (L.indexer) {
            const int64_t cap = (int64_t)cfg.idx_top_k * BLK;
            slots.reserve((size_t)HI * (size_t)(cap < 2048 ? cap : 2048));
            for (int ih = 0; ih < HI; ++ih) {
                slot_off[(size_t)ih] = (int)slots.size();
                for (int b = 0; b < nb; ++b) {
                    if (!sel[(size_t)ih * nb + b]) continue;
                    const int64_t begin = (int64_t)b * BLK;
                    int64_t end = begin + BLK;
                    if (end > Tk) end = Tk;
                    for (int64_t t = begin; t < end; ++t) {
                        if (t > pos) break;
                        slots.push_back((int32_t)t);
                    }
                }
            }
            slot_off[(size_t)HI] = (int)slots.size();
        }
        for (int qh = 0; qh < H; ++qh) {
            const int kv = qh / q_per_kv;
            const float* qrow = q.data() + (size_t)qh * D;
            float* out = att + (size_t)qh * D;
            for (int d = 0; d < D; ++d) out[d] = 0.f;
            if (!L.indexer) {
                std::vector<float> prob((size_t)Tk);
                for (int64_t t = 0; t < Tk; ++t) {
                    if (t > pos) prob[(size_t)t] = -std::numeric_limits<float>::infinity();
                    else prob[(size_t)t] = dot_f16_cache(k_at(layer, t, kv), qrow, D) * scale;
                }
                softmax_inplace(prob.data(), (int)Tk);
                for (int64_t t = 0; t < Tk; ++t) {
                    const float p = prob[(size_t)t];
                    if (p == 0.f) continue;
                    const uint16_t* vv = v_at(layer, t, kv);
                    for (int d = 0; d < D; ++d) out[d] += p * ggml_fp16_to_fp32(vv[d]);
                }
                continue;
            }
            const int ih = qh / q_per_idx;
            const int a = slot_off[(size_t)ih];
            const int b = slot_off[(size_t)ih + 1];
            const int nslot = b - a;
            if (nslot <= 0) continue;
            std::vector<float> prob((size_t)nslot);
            for (int i = 0; i < nslot; ++i)
                prob[(size_t)i] = dot_f16_cache(k_at(layer, slots[(size_t)(a + i)], kv), qrow, D) * scale;
            softmax_inplace(prob.data(), nslot);
            for (int i = 0; i < nslot; ++i) {
                const float p = prob[(size_t)i];
                if (p == 0.f) continue;
                const uint16_t* vv = v_at(layer, slots[(size_t)(a + i)], kv);
                for (int d = 0; d < D; ++d) out[d] += p * ggml_fp16_to_fp32(vv[d]);
            }
        }
    }

    void forward_token(int32_t token, float* logit_out) {
        if (token < 0 || (int64_t)token >= cfg.n_vocab) fail("token id out of range");
        if (n_past >= ctx) fail("context is full (" + std::to_string(ctx) + ")");
        const int64_t pos = n_past;
        tok_stats = TokenStats{};
        decay_usage();
        if (!seed_h.empty()) {
            if (pos >= seed_rows)
                fail("--minimax-seed-hidden has only " + std::to_string(seed_rows) + " rows");
            std::memcpy(x.data(), seed_h.data() + (size_t)pos * (size_t)cfg.n_embd,
                        (size_t)cfg.n_embd * sizeof(float));
        } else {
            decode_f32(embd, embd.host + (size_t)token * embd.row_bytes, x.data(), cfg.n_embd);
        }

        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            rms(x.data(), L.attn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            gemv(L.wq, h.data(), q.data());
            gemv(L.wk, h.data(), k.data());
            gemv(L.wv, h.data(), v.data());
            rms_heads(q.data(), (int)cfg.n_head, (int)cfg.head_dim, L.q_norm.data(), eps);
            rms_heads(k.data(), (int)cfg.n_head_kv, (int)cfg.head_dim, L.k_norm.data(), eps);
            rope(q.data(), (int)cfg.n_head, (int)cfg.head_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
            rope(k.data(), (int)cfg.n_head_kv, (int)cfg.head_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
            const size_t koff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
            store_f16(k_host, koff, k.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            store_f16(v_host, koff, v.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            if (L.indexer) {
                gemv(L.iq, h.data(), iq.data());
                gemv(L.ik, h.data(), ik.data());
                rms_heads(iq.data(), (int)cfg.idx_heads, (int)cfg.idx_dim, L.iq_norm.data(), eps);
                rms(ik.data(), L.ik_norm.data(), ik.data(), (int)cfg.idx_dim, eps);
                rope(iq.data(), (int)cfg.idx_heads, (int)cfg.idx_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
                rope(ik.data(), 1, (int)cfg.idx_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
                const size_t ioff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.idx_dim;
                store_f16(ik_host, ioff, ik.data(), (int)cfg.idx_dim);
            }
            std::vector<float> att((size_t)cfg.n_head * (size_t)cfg.head_dim);
            attend((int)li, L, pos, att.data());
            gemv(L.wo, att.data(), proj.data());
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];

            rms(x.data(), L.ffn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            if (!L.moe) mlp(L.gate, L.up, L.down, 0, h.data(), proj.data(), (int)cfg.n_ff_dense);
            else moe(L, h.data(), proj.data());
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];
            if (dump_on && dump_sel[(size_t)li]) {
                auto& rows = dump_rows[(int)li];
                rows.insert(rows.end(), x.begin(), x.begin() + (size_t)cfg.n_embd);
            }
        }
        rms(x.data(), out_norm.data(), h.data(), (int)cfg.n_embd, eps);
        gemv(output, h.data(), logit_out);
        ++n_past;
        if (tiers_on) {
            total_stats.vram += tok_stats.vram;
            total_stats.ram += tok_stats.ram;
            total_stats.ssd += tok_stats.ssd;
            total_stats.ssd_bytes += tok_stats.ssd_bytes;
            ++tokens_seen;
            if (opt.adaptive && (++rebalance_tick % 16) == 0) rebalance();
            if (opt.stats)
                std::fprintf(stderr,
                             "INFO minimax-stats token=%lld vram=%lld ram=%lld ssd=%lld ssd_bytes=%lld swaps=%llu\n",
                             (long long)tokens_seen, (long long)tok_stats.vram, (long long)tok_stats.ram,
                             (long long)tok_stats.ssd, (long long)tok_stats.ssd_bytes,
                             (unsigned long long)swaps);
        }
    }

    void reset() {
        n_past = 0;
        held.clear();
        have_logits = false;
    }

    int sample(std::vector<float>& row, const SampleParams& sp, const std::vector<int32_t>& hist) {
        const int n = (int)cfg.n_vocab;
        if (sp.penalty_repeat != 1.f || sp.penalty_freq != 0.f || sp.penalty_present != 0.f) {
            int window = sp.penalty_last_n;
            if (window < 0 || window > (int)hist.size()) window = (int)hist.size();
            std::unordered_map<int32_t, int> cnt;
            for (int i = (int)hist.size() - window; i < (int)hist.size(); ++i) cnt[hist[(size_t)i]]++;
            for (const auto& kv : cnt) {
                const int id = (int)kv.first;
                if (id < 0 || id >= n) continue;
                if (row[(size_t)id] < 0.f) row[(size_t)id] *= sp.penalty_repeat;
                else if (sp.penalty_repeat != 0.f) row[(size_t)id] /= sp.penalty_repeat;
                row[(size_t)id] -= (float)kv.second * sp.penalty_freq + sp.penalty_present;
            }
        }
        if (!(sp.temperature > 0.f)) {
            int best = 0;
            for (int i = 1; i < n; ++i)
                if (row[(size_t)i] > row[(size_t)best]) best = i;
            return best;
        }
        const float inv_t = 1.f / sp.temperature;
        for (int i = 0; i < n; ++i) row[(size_t)i] *= inv_t;
        if (sp.top_k > 0 && sp.top_k < n) {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::nth_element(order.begin(), order.begin() + sp.top_k, order.end(), [&](int a, int b) {
                if (row[(size_t)a] != row[(size_t)b]) return row[(size_t)a] > row[(size_t)b];
                return a < b;
            });
            // nth_element's first top_k entries are the kept ids, unsorted. The cutoff is their minimum.
            float cut = row[(size_t)order[0]];
            for (int i = 1; i < sp.top_k; ++i)
                if (row[(size_t)order[(size_t)i]] < cut) cut = row[(size_t)order[(size_t)i]];
            for (int i = 0; i < n; ++i)
                if (row[(size_t)i] < cut) row[(size_t)i] = -std::numeric_limits<float>::infinity();
        }
        softmax_inplace(row.data(), n);
        if (sp.min_p > 0.f) {
            float peak = 0.f;
            for (int i = 0; i < n; ++i)
                if (row[(size_t)i] > peak) peak = row[(size_t)i];
            const float floor = sp.min_p * peak;
            float sum = 0.f;
            for (int i = 0; i < n; ++i) {
                if (row[(size_t)i] < floor) row[(size_t)i] = 0.f;
                sum += row[(size_t)i];
            }
            if (sum > 0.f)
                for (int i = 0; i < n; ++i) row[(size_t)i] /= sum;
        }
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (row[(size_t)a] != row[(size_t)b]) return row[(size_t)a] > row[(size_t)b];
            return a < b;
        });
        if (sp.top_p > 0.f && sp.top_p < 1.f) {
            float c = 0.f;
            int keep = 0;
            for (; keep < n; ++keep) {
                c += row[(size_t)order[(size_t)keep]];
                if (c >= sp.top_p) {
                    ++keep;
                    break;
                }
            }
            if (keep < 1) keep = 1;
            float sum = 0.f;
            for (int i = 0; i < n; ++i) {
                bool kept = false;
                for (int j = 0; j < keep; ++j)
                    if (order[(size_t)j] == i) {
                        kept = true;
                        break;
                    }
                if (!kept) row[(size_t)i] = 0.f;
                sum += row[(size_t)i];
            }
            if (sum > 0.f)
                for (int i = 0; i < n; ++i) row[(size_t)i] /= sum;
        }
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double c = 0.0;
        for (int i = 0; i < n; ++i) {
            c += (double)row[(size_t)i];
            if (r <= c) return i;
        }
        for (int i = n - 1; i >= 0; --i)
            if (row[(size_t)i] > 0.f) return i;
        return 0;
    }
};

Engine::Engine(const std::string& path, const EngineOptions& opt) : impl_(new Impl(path, opt)) {}
Engine::~Engine() { delete impl_; }
const Config& Engine::config() const { return impl_->cfg; }
int64_t Engine::context() const { return impl_->ctx; }
int Engine::dense_layers_gpu() const { return impl_->dense_gpu; }
bool Engine::norm_add1() const { return impl_->norm_add1; }

std::string Engine::describe() const {
    std::ostringstream o;
    o << "prefill=token dense_gpu_layers=" << impl_->dense_gpu << " output_gpu=" << (impl_->output_gpu ? 1 : 0)
      << " experts=";
    if (!impl_->tiers_on) {
        o << "cpu";
    } else {
        o << "tiered vram_slots_per_layer=" << impl_->vram_per_layer << " ram_experts_per_layer="
          << impl_->ram_per_layer << " profile=" << (impl_->profile.empty() ? "none" : "loaded")
          << " adaptive=" << (impl_->opt.adaptive ? 1 : 0);
    }
    o << " kv=f16 norm_add1=" << (impl_->norm_add1 ? 1 : 0) << " threads=" << impl_->threads
      << " ctx=" << impl_->ctx;
    return o.str();
}

std::string Engine::stats() const {
    if (!impl_->tiers_on) return std::string();
    std::ostringstream o;
    o << "tiers=on vram_slots_per_layer=" << impl_->vram_per_layer << " ram_experts_per_layer="
      << impl_->ram_per_layer << " tokens=" << impl_->tokens_seen << " vram_hits=" << impl_->total_stats.vram
      << " ram_hits=" << impl_->total_stats.ram << " ssd_reads=" << impl_->total_stats.ssd
      << " ssd_bytes=" << impl_->total_stats.ssd_bytes << " swaps=" << impl_->swaps;
    return o.str();
}

void Engine::save_profile() { impl_->write_own_profile(); }

void Engine::teacher_forced(const std::vector<int32_t>& tokens, int last_k, std::vector<float>& logits, int& rows) {
    impl_->reset();
    const int vocab = (int)impl_->cfg.n_vocab;
    const int n = (int)tokens.size();
    if (n < 1) fail("teacher-forced sequence is empty");
    if ((int64_t)n > impl_->ctx) fail("teacher-forced sequence is longer than the context");
    std::vector<float> all((size_t)n * vocab);
    for (int i = 0; i < n; ++i) {
        impl_->forward_token(tokens[(size_t)i], all.data() + (size_t)i * vocab);
        if (!finite_row(all.data() + (size_t)i * vocab, vocab))
            fail("non-finite logits at position " + std::to_string(i));
    }
    int keep = last_k <= 0 || last_k > n ? n : last_k;
    const int start = n - keep;
    logits.assign(all.begin() + (size_t)start * vocab, all.end());
    rows = keep;
    impl_->flush_dumps();
}

GenerateResult Engine::generate(const std::vector<int32_t>& prompt, int max_new, const SampleParams& sp,
                                std::atomic<bool>& stop, const std::function<void(int32_t)>& on_token,
                                const std::function<void(int, int, double)>& on_pp) {
    GenerateResult gr;
    if (prompt.empty()) fail("prompt is empty");
    if ((int64_t)prompt.size() > impl_->ctx) fail("prompt is longer than the context");
    if (max_new < 0) max_new = 0;
    if (sp.have_seed) impl_->rng.seed(sp.seed);
    else {
        std::random_device rd;
        impl_->rng.seed(((uint64_t)rd() << 32) ^ rd());
    }

    // n_past is the cache. The last sampled token sits in `held` but is not forwarded until the next
    // request needs its logits, so held.size() is one past the cache after a normal stop. Reuse only the
    // forwarded prefix. Logits always come from the last prompt token: an exact cache hit re-forwards it.
    int64_t cached = impl_->n_past;
    bool extend = cached > 0 && (int64_t)prompt.size() >= cached && (int64_t)impl_->held.size() >= cached;
    if (extend) {
        for (int64_t i = 0; i < cached; ++i) {
            if (impl_->held[(size_t)i] != prompt[(size_t)i]) {
                extend = false;
                break;
            }
        }
    }
    if (!extend) {
        impl_->reset();
        cached = 0;
    }
    gr.reused = (int)cached;
    gr.prompt = (int)prompt.size();
    const bool reforward = extend && (int64_t)prompt.size() == cached;
    const int64_t begin = reforward ? cached - 1 : cached;
    impl_->held.assign(prompt.begin(), prompt.begin() + (size_t)begin);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto ms_since = [&](clock::time_point a) {
        return std::chrono::duration<double, std::milli>(clock::now() - a).count();
    };
    const int chunk = impl_->opt.chunk > 0 ? impl_->opt.chunk : 16;
    for (int i = (int)begin; i < (int)prompt.size(); ++i) {
        if (stop.load(std::memory_order_relaxed)) {
            gr.finish = "cancel";
            gr.prompt_ms = ms_since(t0);
            return gr;
        }
        if (reforward && i == (int)begin) impl_->n_past = begin;
        impl_->forward_token(prompt[(size_t)i], impl_->last_logits.data());
        impl_->have_logits = true;
        impl_->held.push_back(prompt[(size_t)i]);
        const int done = i + 1;
        if (on_pp && (done == (int)prompt.size() || done % chunk == 0))
            on_pp(done, (int)prompt.size(), ms_since(t0));
        if (!finite_row(impl_->last_logits.data(), (int)impl_->cfg.n_vocab)) {
            gr.finish = "nonfinite";
            gr.prompt_ms = ms_since(t0);
            return gr;
        }
    }
    gr.prompt_ms = ms_since(t0);
    if (!impl_->have_logits) {
        gr.finish = "length";
        return gr;
    }

    const auto t1 = clock::now();
    std::vector<int32_t> hist = impl_->held;
    for (int n = 0; n < max_new; ++n) {
        if (stop.load(std::memory_order_relaxed)) {
            gr.finish = "cancel";
            gr.generated = n;
            gr.decode_ms = ms_since(t1);
            return gr;
        }
        std::vector<float> row = impl_->last_logits;
        if (!finite_row(row.data(), (int)row.size())) {
            gr.finish = "nonfinite";
            gr.generated = n;
            gr.decode_ms = ms_since(t1);
            return gr;
        }
        const int id = impl_->sample(row, sp, hist);
        if (on_token) on_token(id);
        hist.push_back(id);
        impl_->held.push_back(id);
        gr.generated = n + 1;
        if (impl_->eos.count(id)) {
            gr.finish = "stop";
            gr.decode_ms = ms_since(t1);
            return gr;
        }
        if (n + 1 >= max_new) break;
        if (impl_->n_past >= impl_->ctx) {
            gr.finish = "length";
            gr.decode_ms = ms_since(t1);
            return gr;
        }
        impl_->forward_token(id, impl_->last_logits.data());
        if (!finite_row(impl_->last_logits.data(), (int)impl_->cfg.n_vocab)) {
            gr.finish = "nonfinite";
            gr.generated = n + 1;
            gr.decode_ms = ms_since(t1);
            return gr;
        }
    }
    gr.finish = "length";
    gr.decode_ms = ms_since(t1);
    return gr;
}

void write_npy_f32(const std::string& path, const float* data, int rows, int cols) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                         std::to_string(cols) + "), }";
    // v1.0: magic(6) + version(2) + header_len(2) + header is a multiple of 16.
    const size_t prefix = 10;
    size_t pad = 16 - ((prefix + header.size() + 1) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');
    if (header.size() > 65535) fail("npy header does not fit in v1");
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("cannot write " + path);
    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    const unsigned char ver[2] = {1, 0};
    const uint16_t hlen = (uint16_t)header.size();
    f.write(magic, 6);
    f.write((const char*)ver, 2);
    f.write((const char*)&hlen, 2);
    f.write(header.data(), (std::streamsize)header.size());
    f.write((const char*)data, (std::streamsize)((size_t)rows * (size_t)cols * sizeof(float)));
    if (!f) fail("short write to " + path);
}

std::vector<int32_t> read_token_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("cannot read " + path);
    std::vector<int32_t> ids;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        for (char& c : line)
            if (c == ',') c = ' ';
        std::istringstream in(line);
        std::string tok;
        while (in >> tok) {
            char* end = nullptr;
            const long v = std::strtol(tok.c_str(), &end, 10);
            if (end == tok.c_str() || *end != '\0') fail("not a token id: " + tok);
            ids.push_back((int32_t)v);
        }
    }
    if (ids.empty()) fail(path + " has no token ids");
    return ids;
}

}  // namespace strata::minimax
