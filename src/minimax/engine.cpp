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
#include <map>
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

    // P4: this layer's owning device (an index into Impl::devs) and the KV it keeps there.
    int dev = 0;
    DevPtr d_k, d_v, d_ik;          // when the KV lives in VRAM
};

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

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

// ---- parity probes.  A minimal .npy codec: little-endian fp32 / int64, C order, no pickle.
bool read_npy_f32(const std::string& path, std::vector<float>& out, int64_t& rows, int64_t& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6] = {0};
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    unsigned char ver[2] = {0, 0};
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char b[2] = {0, 0};
        f.read(reinterpret_cast<char*>(b), 2);
        hlen = (uint32_t)b[0] | ((uint32_t)b[1] << 8);
    } else {
        unsigned char b[4] = {0, 0, 0, 0};
        f.read(reinterpret_cast<char*>(b), 4);
        hlen = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    }
    std::string hdr(hlen, '\0');
    f.read(&hdr[0], (std::streamsize)hlen);
    if (!f) return false;
    if (hdr.find("'<f4'") == std::string::npos) return false;
    if (hdr.find("'fortran_order': True") != std::string::npos) return false;
    const auto lp = hdr.find('(');
    const auto rp = hdr.find(')');
    if (lp == std::string::npos || rp == std::string::npos || rp < lp) return false;
    std::vector<int64_t> dims;
    std::string inner = hdr.substr(lp + 1, rp - lp - 1);
    for (size_t i = 0; i < inner.size();) {
        if (inner[i] < '0' || inner[i] > '9') { ++i; continue; }
        size_t j = i;
        while (j < inner.size() && inner[j] >= '0' && inner[j] <= '9') ++j;
        dims.push_back(std::stoll(inner.substr(i, j - i)));
        i = j;
    }
    if (dims.empty()) return false;
    int64_t total = 1;
    for (int64_t d : dims) total *= d;
    rows = 1;
    for (size_t i = 0; i + 1 < dims.size(); ++i) rows *= dims[i];
    cols = dims.back();
    out.resize((size_t)total);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)total * 4);
    return (bool)f;
}

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

    // ---- parity probes (P4).  Off unless --minimax-dump-dir / --minimax-seed-hidden is given.
    bool dump_on = false;
    std::string dump_dir;
    std::vector<char> dump_sel;       // [n_layers] 1 when this layer's activations are collected
    std::vector<int32_t> src_layer;   // [n_layers] source layer id (strata.source_layers when the file is remapped)
    struct DumpBuf {
        int cols = 0;
        std::vector<float> f;
        std::vector<int64_t> i;
    };
    std::map<std::string, DumpBuf> dumps;
    std::vector<float> seed_h;
    int64_t seed_rows = 0, seed_cols = 0;
    int64_t dump_seq = 0;            // sequence length of the current eval, for the block-score padding
    bool dumping(int64_t li) const { return dump_on && dump_sel[(size_t)li] != 0; }
    void dbuf_put(const std::string& key, const float* p, int n) {
        DumpBuf& b = dumps[key];
        if (b.cols == 0) b.cols = n;
        b.f.insert(b.f.end(), p, p + n);
    }
    void dbuf_put_i(const std::string& key, const int64_t* p, int n) {
        DumpBuf& b = dumps[key];
        if (b.cols == 0) b.cols = n;
        b.i.insert(b.i.end(), p, p + n);
    }

    std::vector<uint16_t> k_host_v, v_host_v, ik_host_v;   // host-mode storage (std::vector)
    uint16_t* k_host = nullptr;      // the active K/V/indexer-K storage: pinned when CUDA is up
    uint16_t* v_host = nullptr;
    uint16_t* ik_host = nullptr;
    void* kv_arena = nullptr;        // cudaHostAlloc'd bytes behind k_host (null in host mode)
    size_t kv_arena_bytes = 0;
    DevPtr d_x, d_y, d_scratch;
    int64_t max_cols = 1, max_rows = 1;

    // ---- P4: devices, layer ownership, KV placement
    std::vector<int> devs;           // CUDA ordinals in use, in order
    std::vector<int> layer_dev;      // [n_layers] index into devs
    int cur = 0;                     // index into devs of the current CUDA device
    int kv_mode = 0;                 // 0 host (CPU attention), 1 VRAM, 2 pinned RAM + indexer keys in VRAM
    int64_t kv_row_bytes = 0;        // one token's K (or V) bytes: n_head_kv * head_dim * 2
    int64_t ik_row_bytes = 0;
    int32_t max_slot = 0;            // selected positions one KV head can have
    struct Work {
        DevPtr x, y, scratch, ex_ptr, ex_xq, ex_hq, ex_h, ex_out;
        DevPtr ex_sg, ex_su, ex_sd;     // MM-NVFP4: per-selected-expert divisors, one per device
        DevPtr qd, bs, slots, counts, kstage, vstage, att;
        uint16_t* kstage_h = nullptr;   // pinned staging for the RAM tier
        uint16_t* vstage_h = nullptr;
        void* hop_h = nullptr;          // pinned staging for the hidden state that crosses devices
        size_t hop_bytes = 0;
        void* side = nullptr;
    };
    std::vector<Work> work;          // one entry per device in `devs`
    Work& W() { return work[(size_t)cur]; }
    void use_layer_device(int64_t li) {
        const int d = layer_dev[(size_t)li];
        if (d != cur) {
            cur = d;
            cu::set_device(devs[(size_t)d]);
        }
    }
    void use_device_index(int d) {
        if (d != cur) {
            cur = d;
            cu::set_device(devs[(size_t)d]);
        }
    }

    std::vector<float> x, h, q, k, v, proj, ff_g, ff_u, ff_h, ff_y, iq, ik, logits, last_logits;
    std::vector<uint16_t> kv_tmp;
    std::vector<int32_t> held;
    int64_t n_past = 0;
    int64_t cur_pos = 0;
    bool have_logits = false;
    std::mt19937_64 rng{0};
    Pool pool;

    // ---- P5a: batched prompt processing.  `batch_cap` tokens per layer at a time; every buffer below lives on the
    // device that owns the layer being computed (one set per device).  Off unless --minimax-prefill-batch > 1.
    bool batch_on = false;
    int batch_cap = 0;
    bool prefill_gpu_experts = true;
    bool prefill_dense_f32 = true;
    bool prefill_direct = false;   // unbuffered expert reads (--minimax-prefill-read direct)
    struct PfBuf {
        uint16_t* x16 = nullptr;     // [T][embd] fp16 hidden state
        uint16_t* h16 = nullptr;     // [T][embd] fp16 normed input to the projections
        float* y32 = nullptr;        // [T][maxN] fp32 GEMM output (also the attention output)
        float* p32 = nullptr;        // [T][embd] fp32 projection / routed MoE output
        float* sh = nullptr;         // [T][embd] fp32 shared-expert output and the final MoE sum
        uint16_t* w16 = nullptr;     // fp16 dequantization scratch for one weight tile
        float* w32 = nullptr;        // fp32 scratch (the parity-first cuBLAS SGEMM path)
        int64_t w16_elems = 0;
        uint16_t* gx = nullptr;      // gathered group activations [rows][embd] fp16
        float* gx32 = nullptr;       // the same in fp32 (the q8_1 path's source)
        float* gy = nullptr;         // [rows][ff] gate
        float* gu = nullptr;         // [rows][ff] up
        uint16_t* gh = nullptr;      // [rows][ff] hidden (fp16, cuBLAS path)
        float* gh32 = nullptr;       // [rows][ff] hidden (fp32, q8_1 path)
        float* gd = nullptr;         // [rows][embd] down
        void* xq = nullptr;          // q8_1 activations [rows][embd]
        void* hq = nullptr;          // q8_1 hidden [rows][ff]
        void* ring = nullptr;        // expert streaming slot (raw GGUF bytes)
        size_t ring_slot = 0;
        void* pin = nullptr;         // page-locked staging for the ring (same bytes)
        DevPtr d_iq16, d_bs, d_blk, d_nblk, d_rows, d_ent_row, d_ent_w;

        DevPtr d_q16;                // fp16 attention queries for the batched MSA kernel
        int64_t rows_cap = 0;
    };
    std::vector<PfBuf> pf;
    int64_t max_dense_n = 0;         // widest projection (rows)
    int64_t rows_cap = 0;            // rows per expert/dense chunk in the batched path
    int pf_max_blk = 0;              // top_k blocks per query (device shared-memory bound)

    struct PfEntry {
        int tid;
        int expert;
        float w;
        int row;                     // row in the gathered group buffers
    };
    std::vector<PfEntry> pf_entries;
    std::vector<int> pf_tok_exps;    // [T*top] expert ids
    std::vector<float> pf_tok_w;     // [T*top] combine weights
    std::vector<int32_t> pf_ent_row; // [T*top] row into the layer's routed output
    std::vector<float> pf_router;    // [T*nexp] router logits (host)
    std::vector<float> pf_h;         // [T*embd] host hidden state
    std::vector<float> pf_q, pf_kv;  // [T*qlen] / [T*klen] host
    std::vector<int32_t> pf_blk;     // [T*idx_heads*top_k]
    std::vector<int32_t> pf_nblk;    // [T*idx_heads]
    std::vector<uint16_t> pf_k16, pf_v16;
    std::vector<float> pf_gate_h, pf_up_h, pf_hid_h;
    std::vector<float> pf_moe_h;     // [T][embd] the layer's FFN output on the host
    std::vector<float> pf_dense_h;   // [T][max_dense_n] host intermediate for the CPU dense MLP
    std::vector<float> pf_out_tile;  // [T][lm_head tile] host staging for the output projection

    // ---- P5b: decode.  Exactly one layer ahead, the router of layer l+1 is applied to the residual layer l just
    // produced (that IS layer l+1's input), and its non-resident experts are copied into free VRAM slots on a side
    // stream while layer l's expert math runs.  `prefetch_pending` is the stream's work for the layer above.
    int prefetch_slots = 0;   // --minimax-expert-prefetch (0 = off)
    int prefetch_pending = -1;

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
        const bool cpu_only = opt.dense_layers_gpu == 0 && !opt.tiers && opt.layer_split.empty() &&
                              (opt.kv_space.empty() || opt.kv_space == "host");
        if (!cpu_only) {
            cu::init();
            setup_devices();
            choose_split_and_kv();
        }
        alloc_cache();
        // The row/col limits are also the CPU gemv guard. Device buffers exist only when a layer can be uploaded.
        size_workspace();
        if (!cpu_only) alloc_workspace();
        place_weights();
        if (!cpu_only && kv_mode != 0) alloc_device_kv();
        if (opt.tiers) {
            init_tiers();
        } else if (!opt.profile_save_path.empty() || !opt.routing_counts_path.empty()) {
            for (const Layer& L : layers)
                if (L.moe) ++moe_layers;
            route_counts.assign((size_t)cfg.n_layers * (size_t)cfg.n_expert, 0.0);
        }
        init_prefill();

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
        kv_tmp.assign((size_t)bigger(klen, cfg.idx_dim) * 2, 0);
        logits.assign((size_t)cfg.n_vocab, 0.f);
        last_logits.assign((size_t)cfg.n_vocab, 0.f);
    }

    ~Impl() {
        write_own_profile();
        flush_dumps();
        for (Work& w : work) {
            if (w.side) cu::stream_destroy(w.side);
            w.side = nullptr;
            if (w.kstage_h) cu::host_free(w.kstage_h);
            if (w.vstage_h) cu::host_free(w.vstage_h);
            w.kstage_h = w.vstage_h = nullptr;
        }
        if (kv_arena) cu::host_free(kv_arena);
        for (PfBuf& p : pf) {
            if (p.ring) cu::dealloc(p.ring);
            p.ring = nullptr;
            if (p.pin) cu::host_free(p.pin);
            p.pin = nullptr;
        }
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

    void read_eos() {
        if (const MetaValue* e = model.meta().get("tokenizer.ggml.eos_token_id"))
            if (e->is_num()) eos.insert((int32_t)e->u);
        if (const MetaValue* a = model.meta().get("tokenizer.ggml.eos_token_ids"))
            if (a->type == MetaType::ARRAY)
                for (const MetaValue& it : a->items)
                    if (it.is_num()) eos.insert((int32_t)it.u);
    }

    // ---- parity probes: which layers to collect, the source layer ids, and the optional input seed.
    void init_probes() {
        src_layer.resize((size_t)cfg.n_layers);
        for (int64_t i = 0; i < cfg.n_layers; ++i) src_layer[(size_t)i] = (int32_t)i;
        if (const MetaValue* sl = model.meta().get("strata.source_layers")) {
            if (sl->type == MetaType::ARRAY && (int64_t)sl->items.size() == cfg.n_layers)
                for (int64_t i = 0; i < cfg.n_layers; ++i) src_layer[(size_t)i] = (int32_t)sl->items[(size_t)i].u;
        }
        if (!opt.seed_hidden.empty()) {
            if (!read_npy_f32(opt.seed_hidden, seed_h, seed_rows, seed_cols))
                fail("--minimax-seed-hidden: cannot read " + opt.seed_hidden + " as an .npy fp32 C-order array");
            if (seed_cols != cfg.n_embd)
                fail("--minimax-seed-hidden: last dimension is " + std::to_string(seed_cols) + ", the model needs " +
                     std::to_string(cfg.n_embd));
            if (seed_rows < 1) fail("--minimax-seed-hidden: the array is empty");
        }
        if (opt.dump_dir.empty()) return;
        dump_on = true;
        dump_dir = opt.dump_dir;
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        std::string want = opt.dump_layers.empty() ? "all" : opt.dump_layers;
        dump_sel.assign((size_t)cfg.n_layers, 0);
        if (want == "all") {
            dump_sel.assign((size_t)cfg.n_layers, 1);
        } else {
            std::stringstream ss(want);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (tok.empty()) continue;
                const int id = std::atoi(tok.c_str());
                bool hit = false;
                for (int64_t i = 0; i < cfg.n_layers; ++i)
                    if (src_layer[(size_t)i] == id) { dump_sel[(size_t)i] = 1; hit = true; }
                if (!hit)
                    std::fprintf(stderr, "strata-minimax: --minimax-dump-layers %d is not in this file\n", id);
            }
        }
    }

    void flush_dumps() {
        if (!dump_on || dumps.empty()) return;
        std::filesystem::create_directories(dump_dir);
        for (auto& kv : dumps) {
            const std::string key = kv.first;   // "L40/attn_out"
            const auto slash = key.find('/');
            const std::string sub = dump_dir + "/dump-" + key.substr(0, slash);
            std::filesystem::create_directories(sub);
            const std::string path = sub + "/" + key.substr(slash + 1) + ".npy";
            DumpBuf& b = kv.second;
            if (!b.f.empty()) {
                const int rows = b.cols > 0 ? (int)(b.f.size() / (size_t)b.cols) : 0;
                write_npy_f32(path, b.f.data(), rows, b.cols);
            } else if (!b.i.empty()) {
                const int rows = b.cols > 0 ? (int)(b.i.size() / (size_t)b.cols) : 0;
                std::FILE* f = std::fopen(path.c_str(), "wb");
                if (!f) continue;
                std::string header = "{'descr': '<i8', 'fortran_order': False, 'shape': (" + std::to_string(rows) +
                                     ", " + std::to_string(b.cols) + "), }";
                const size_t total = 10 + header.size() + 1;
                const size_t pad = (64 - (total % 64)) % 64;
                header.append(pad, ' ');
                header += '\n';
                const unsigned char pre[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                               (unsigned char)(header.size() & 0xff),
                                               (unsigned char)((header.size() >> 8) & 0xff)};
                std::fwrite(pre, 1, 10, f);
                std::fwrite(header.data(), 1, header.size(), f);
                std::fwrite(b.i.data(), sizeof(int64_t), b.i.size(), f);
                std::fclose(f);
            }
        }
        dumps.clear();
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
        // P4: in the RAM tier the K/V arrays are page-locked, so the per-step gather reaches full copy-engine
        // bandwidth. The VRAM tier keeps no host copy at all; the host tier (the P2 default) keeps the vectors.
        if (kv_mode == 2 && !work.empty()) {
            size_t kbytes = 0;
            if (!mul_size(kp, sizeof(uint16_t), kbytes)) kbytes = 0;
            kv_arena_bytes = kbytes * 2;
            kv_arena = cu::host_alloc(kv_arena_bytes);
            k_host = (uint16_t*)kv_arena;
            v_host = k_host + kp;
            ik_host = nullptr;
            std::memset(kv_arena, 0, kv_arena_bytes);
            return;
        }
        if (kv_mode == 1 && !work.empty()) {
            k_host = v_host = ik_host = nullptr;
            return;
        }
        try {
            k_host_v.assign(kp, 0);
            v_host_v.assign(kp, 0);
            ik_host_v.assign(ip, 0);
        } catch (const std::exception& e) {
            fail("failed to allocate the host fp16 KV cache for context " + std::to_string(ctx) + ": " +
                 std::to_string(bytes) + " bytes are required. Lower --max-context. " + e.what());
        }
        k_host = k_host_v.data();
        v_host = v_host_v.data();
        ik_host = ik_host_v.data();
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
        for (size_t d = 0; d < devs.size(); ++d) {
            use_device_index((int)d);
            Work& w = work[d];
            w.x = DevPtr(cu::alloc((size_t)max_cols * sizeof(float)));
            w.y = DevPtr(cu::alloc((size_t)max_rows * sizeof(float)));
            w.scratch = DevPtr(cu::alloc((size_t)kTile * (size_t)max_cols * sizeof(float)));
            if (kv_mode != 0) {
                w.qd = DevPtr(cu::alloc((size_t)cfg.n_head * cfg.head_dim * sizeof(float)));
                w.att = DevPtr(cu::alloc((size_t)cfg.n_head * cfg.head_dim * sizeof(float)));
                const size_t nb_max = (size_t)((ctx + cfg.idx_block - 1) / cfg.idx_block);
                w.bs = DevPtr(cu::alloc(nb_max * (size_t)cfg.idx_heads * sizeof(float)));
                w.slots = DevPtr(cu::alloc((size_t)cfg.n_head_kv * (size_t)max_slot * sizeof(int32_t)));
                w.counts = DevPtr(cu::alloc((size_t)cfg.n_head_kv * sizeof(int32_t)));
                w.kstage = DevPtr(cu::alloc((size_t)cfg.n_head_kv * (size_t)max_slot * (size_t)cfg.head_dim * 2));
                w.vstage = DevPtr(cu::alloc((size_t)cfg.n_head_kv * (size_t)max_slot * (size_t)cfg.head_dim * 2));
                if (kv_mode == 2) {
                    const size_t st = (size_t)cfg.n_head_kv * (size_t)max_slot * (size_t)cfg.head_dim * 2;
                    w.kstage_h = (uint16_t*)cu::host_alloc(st);
                    w.vstage_h = (uint16_t*)cu::host_alloc(st);
                }
            }
            w.side = cu::stream_create();
            if (devs.size() > 1) {
                w.hop_bytes = (size_t)max_cols * sizeof(float);
                w.hop_h = cu::host_alloc(w.hop_bytes);
            }
        }
        use_device_index(0);
    }

    // Every activation that crosses to a device goes through page-locked memory when more than one GPU is in
    // play: there is no P2P between these cards, so the hidden state hops through the host, and a pinned source
    // keeps the copy engine at full rate instead of bouncing through a pageable staging buffer.
    void h2d(Work& w, void* dst, const void* src, size_t bytes) {
        if (w.hop_h && bytes <= w.hop_bytes) {
            std::memcpy(w.hop_h, src, bytes);
            cu::copy_h2d(dst, w.hop_h, bytes);
            return;
        }
        cu::copy_h2d(dst, src, bytes);
    }

    // ---- P4: the device list, the split, and where the KV lives.
    void setup_devices() {
        const int n = cu::device_count();
        if (n < 1) fail("CUDA reports no device");
        if (!opt.gpus.empty()) {
            std::stringstream ss(opt.gpus);
            std::string t;
            while (std::getline(ss, t, ',')) {
                if (t.empty()) continue;
                const int d = std::atoi(t.c_str());
                if (d < 0 || d >= n)
                    fail("--gpus " + opt.gpus + ": device " + std::to_string(d) + " does not exist (" +
                         std::to_string(n) + " visible)");
                for (int seen : devs)
                    if (seen == d) fail("--gpus lists device " + std::to_string(d) + " twice");
                devs.push_back(d);
            }
        }
        if (devs.empty()) devs.push_back(0);
        work.clear();
        work.resize(devs.size());
        layer_dev.assign((size_t)cfg.n_layers, 0);
        cur = 0;
        cu::set_device(devs[0]);
    }

    size_t kv_layer_bytes() const {
        // K and V for one layer at this context, plus the indexer keys (over-counted on a file without an indexer
        // on every layer, which is the safe direction).
        return (size_t)(2 * kv_row_bytes) * (size_t)ctx + (size_t)ik_row_bytes * (size_t)ctx;
    }

    void choose_split_and_kv() {
        kv_row_bytes = (int64_t)cfg.n_head_kv * cfg.head_dim * 2;
        ik_row_bytes = (int64_t)cfg.idx_dim * 2;
        max_slot = (int32_t)(cfg.idx_top_k * cfg.idx_block);
        if (max_slot < (int32_t)cfg.idx_block) max_slot = (int32_t)cfg.idx_block;

        const int nd = (int)devs.size();
        const size_t reserve = (size_t)(opt.vram_reserve_mib > 0 ? opt.vram_reserve_mib : 0) << 20;
        std::vector<size_t> budget((size_t)nd), dense((size_t)cfg.n_layers);
        for (int d = 0; d < nd; ++d) {
            const cu::MemInfo mi = cu::mem_on(devs[(size_t)d]);
            budget[(size_t)d] = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
        }
        for (int64_t i = 0; i < cfg.n_layers; ++i) dense[(size_t)i] = layer_cost(layers[(size_t)i]);
        const int64_t dlim = opt.dense_layers_gpu == 0 ? 0
                            : (opt.dense_layers_gpu < 0 ? cfg.n_layers
                                                        : std::min<int64_t>(opt.dense_layers_gpu, cfg.n_layers));
        for (int64_t i = dlim; i < cfg.n_layers; ++i) dense[(size_t)i] = 0;

        // Where does the KV live?  An explicit --minimax-kv wins; otherwise the single-device default keeps the
        // P2 host cache (CPU attention), and a split tries VRAM first and falls back to pinned RAM.
        int mode = 0;
        if (opt.kv_space == "host") mode = 0;
        else if (opt.kv_space == "vram") mode = 1;
        else if (opt.kv_space == "ram") mode = 2;
        else if (opt.kv_space.empty() && nd == 1 && opt.layer_split.empty()) mode = 0;
        else mode = 1;   // auto
        if (mode != 0 && (cfg.n_head_kv != cfg.idx_heads))
            fail("--minimax-kv vram|ram runs MSA on the device, which needs the indexer head count (" +
                 std::to_string(cfg.idx_heads) + ") to equal the KV head count (" + std::to_string(cfg.n_head_kv) +
                 "). This file has " + std::to_string(cfg.n_head_kv) + " KV heads. Use --minimax-kv host.");
        if (mode != 0 && cfg.idx_first_layer > 0) {
            std::fprintf(stderr,
                         "INFO minimax-kv: this file indexes only from layer %lld, so the device attention path "
                         "is off; keeping the KV cache on the host\n",
                         (long long)cfg.idx_first_layer);
            mode = 0;
        }

        // Split points.  `K` = the first layer on the next device.  "auto" fills each device with dense + KV
        // bytes until its own free VRAM (less the reserve) runs out, so unequal cards work.
        std::vector<int> split;   // first layer of devices 1..nd-1
        if (!opt.layer_split.empty() && opt.layer_split != "auto") {
            std::stringstream ss(opt.layer_split);
            std::string t;
            while (std::getline(ss, t, ',')) {
                if (t.empty()) continue;
                split.push_back(std::atoi(t.c_str()));
            }
            if (split.size() + 1 > (size_t)nd)
                fail("--layer-split lists " + std::to_string(split.size() + 1) + " groups but --gpus has " +
                     std::to_string(nd) + " device(s)");
        }

        auto assign = [&](bool with_kv) {
            std::fill(layer_dev.begin(), layer_dev.end(), 0);
            if (nd == 1) return;
            if (!split.empty()) {
                for (int64_t i = 0; i < cfg.n_layers; ++i) {
                    int d = 0;
                    while (d < (int)split.size() && i >= split[(size_t)d]) ++d;
                    layer_dev[(size_t)i] = d;
                }
                return;
            }
            // auto: proportional targets, filled in layer order
            size_t total = 0, sum_b = 0;
            for (int64_t i = 0; i < cfg.n_layers; ++i) total += dense[(size_t)i] + (with_kv ? kv_layer_bytes() : 0);
            for (int d = 0; d < nd; ++d) sum_b += budget[(size_t)d];
            if (sum_b == 0) fail("no free VRAM on the selected devices after the reserve");
            std::vector<size_t> used((size_t)nd, 0), target((size_t)nd, 0);
            for (int d = 0; d < nd; ++d) target[(size_t)d] = (size_t)((double)total * (double)budget[(size_t)d] / (double)sum_b);
            int d = 0;
            for (int64_t i = 0; i < cfg.n_layers && d < nd; ++i) {
                const size_t c = dense[(size_t)i] + (with_kv ? kv_layer_bytes() : 0);
                while (d + 1 < nd && used[(size_t)d] > 0 && used[(size_t)d] + c > target[(size_t)d]) ++d;
                layer_dev[(size_t)i] = d;
                used[(size_t)d] += c;
            }
        };

        auto check_budget = [&](bool with_kv) -> std::string {
            std::vector<size_t> used((size_t)nd, 0), nlay((size_t)nd, 0);
            for (int64_t i = 0; i < cfg.n_layers; ++i) {
                used[(size_t)layer_dev[(size_t)i]] += dense[(size_t)i] + (with_kv ? kv_layer_bytes() : 0);
                ++nlay[(size_t)layer_dev[(size_t)i]];
            }
            for (int d = 0; d < nd; ++d)
                if (used[(size_t)d] > budget[(size_t)d])
                    return "device " + std::to_string(devs[(size_t)d]) + " needs " +
                           std::to_string(used[(size_t)d] >> 20) + " MiB for " + std::to_string(nlay[(size_t)d]) +
                           " layers" + (with_kv ? " with KV" : "") + " but has " +
                           std::to_string(budget[(size_t)d] >> 20) + " MiB free after the " +
                           std::to_string(opt.vram_reserve_mib) + " MiB reserve";
            return "";
        };

        assign(mode != 0);
        std::string bad = check_budget(mode != 0);
        if (!bad.empty() && mode == 1 && opt.kv_space.empty()) {
            std::fprintf(stderr, "INFO minimax-kv: VRAM does not fit (%s); keeping K/V in pinned RAM\n", bad.c_str());
            mode = 2;
            assign(true);
            bad = check_budget(true);
        }
        if (!bad.empty())
            fail(bad + ". Use --layer-split to choose the boundary, --vram-reserve-mib to change the reserve, or "
                 "--minimax-kv ram to move the KV cache to RAM.");
        kv_mode = mode;
        for (int64_t i = 0; i < cfg.n_layers; ++i) layers[(size_t)i].dev = layer_dev[(size_t)i];
    }

    void alloc_device_kv() {
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            Layer& L = layers[(size_t)i];
            use_device_index(layer_dev[(size_t)i]);
            const size_t kbytes = (size_t)kv_row_bytes * (size_t)ctx;
            if (kv_mode == 1) {
                L.d_k = DevPtr(cu::alloc(kbytes));
                L.d_v = DevPtr(cu::alloc(kbytes));
            }
            L.d_ik = DevPtr(cu::alloc((size_t)ik_row_bytes * (size_t)ctx));
        }
        use_device_index(0);
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
        const size_t reserve = (size_t)(opt.vram_reserve_mib > 0 ? opt.vram_reserve_mib : 0) << 20;
        const bool automatic = opt.dense_layers_gpu < 0;
        int N = -1;
        if (devs.size() > 1 || !automatic) {
            // P4 split (or an explicit count): every layer up to the count goes on its own device.  The split
            // was sized in choose_split_and_kv against each card's own free VRAM.
            N = automatic ? (int)cfg.n_layers : std::min<int>((int)cfg.n_layers, opt.dense_layers_gpu);
        } else {
            // Single device, automatic: fill with leading layers until the card is full (P2 behaviour).
            const cu::MemInfo mi = cu::mem_on(devs[0]);
            const size_t budget = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
            size_t used = 0;
            N = 0;
            while (N < (int)cfg.n_layers && used + cost[(size_t)N] <= budget) {
                used += cost[(size_t)N];
                ++N;
            }
        }
        if (!automatic && opt.dense_layers_gpu > 0 && devs.size() == 1) {
            const cu::MemInfo mi = cu::mem_on(devs[0]);
            const size_t budget = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
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
            use_device_index(L.dev);
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
        // lm_head goes on the device that owns the last layer (upstream's "last card" rule), if it fits there.
        if (N > 0 && out_b != std::numeric_limits<size_t>::max()) {
            use_device_index(layer_dev[(size_t)(cfg.n_layers - 1)]);
            const cu::MemInfo now = cu::mem();
            if (now.free_bytes > out_b + ((size_t)256 << 20)) {
                upload(output);
                output_gpu = true;
            }
        }
        use_device_index(0);
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
        // With a layer split each device sizes its own slots from its own free VRAM, so 8+12 GB works as well as
        // 12+12.  `q` is then the smallest per-device count, which is what the counters and the stats report.
        const size_t reserve = (size_t)(opt.vram_reserve_mib > 0 ? opt.vram_reserve_mib : 0) << 20;
        std::vector<size_t> dev_budget(devs.size(), 0);
        std::vector<int64_t> dev_moe(devs.size(), 0);
        for (int64_t i = 0; i < cfg.n_layers; ++i)
            if (layers[(size_t)i].moe) ++dev_moe[(size_t)layer_dev[(size_t)i]];
        if (tiers_gpu_possible()) {
            for (size_t d = 0; d < devs.size(); ++d) {
                const cu::MemInfo mi = cu::mem_on(devs[d]);
                dev_budget[d] = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
            }
        }
        int q = opt.vram_slots_per_layer;
        if (q < 0) {
            q = (int)cfg.n_expert;
            for (size_t d = 0; d < devs.size(); ++d) {
                if (dev_moe[d] <= 0) continue;
                const int qd = (int)(dev_budget[d] / ((size_t)dev_moe[d] * max_expert_bytes));
                if (qd < q) q = qd;
            }
            if (q < 0) q = 0;
        }
        if (q < 0) q = 0;
        if (q > cfg.n_expert) q = (int)cfg.n_expert;
        if (q > 0) {
            if (!tiers_gpu_possible())
                fail("--minimax-vram-slots-per-layer needs a CUDA device (dense_layers_gpu=0 still starts CUDA "
                     "when the expert tiers are on)");
            for (size_t d = 0; d < devs.size(); ++d) {
                size_t need = 0;
                for (int64_t i = 0; i < cfg.n_layers; ++i) {
                    const Layer& L = layers[(size_t)i];
                    if (L.moe && layer_dev[(size_t)i] == (int)d) need += (size_t)q * L.expert_bytes;
                }
                if (need > dev_budget[d]) {
                    fail("the VRAM expert tier asked for " + std::to_string(q) + " slots per layer (" +
                         std::to_string(need / (1024 * 1024)) + " MiB on device " +
                         std::to_string(devs[d]) + ") but only " +
                         std::to_string(dev_budget[d] / (1024 * 1024)) +
                         " MiB is free after dense/KV/workspace and the " + std::to_string(opt.vram_reserve_mib) +
                         " MiB reserve. Lower --minimax-vram-slots-per-layer or --vram-reserve-mib.");
                }
            }
            for (Layer& L : layers) {
                if (!L.moe) continue;
                use_device_index(L.dev);
                L.vram.reserve((size_t)q);
                for (int s = 0; s < q; ++s) {
                    Layer::VramSlot slot;
                    slot.mem = DevPtr(cu::alloc(L.expert_bytes));
                    L.vram.push_back(std::move(slot));
                }
            }
            const int top = (int)cfg.n_expert_used;
            for (size_t d = 0; d < devs.size(); ++d) {
                use_device_index((int)d);
                Work& w = work[d];
                w.ex_ptr = DevPtr(cu::alloc((size_t)top * 3 * sizeof(void*)));
                w.ex_xq = DevPtr(cu::alloc(q8_1_bytes(cfg.n_embd)));
                w.ex_hq = DevPtr(cu::alloc((size_t)top * q8_1_bytes(cfg.n_ff_exp)));
                w.ex_h = DevPtr(cu::alloc((size_t)top * (size_t)cfg.n_ff_exp * sizeof(float)));
                w.ex_out = DevPtr(cu::alloc((size_t)top * (size_t)cfg.n_embd * sizeof(float)));
                if (!w.side) w.side = cu::stream_create();
            }
            use_device_index(0);
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
        use_device_index(L.dev);
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
        use_device_index(L.dev);
        Layer::VramSlot& s = L.vram[(size_t)slot];
        uint8_t* p = (uint8_t*)s.mem.p;
        cu::copy_h2d_async(p, L.egate.host + (size_t)expert * L.expert_gu, L.expert_gu, W().side);
        cu::copy_h2d_async(p + L.expert_gu, L.eup.host + (size_t)expert * L.expert_gu, L.expert_gu, W().side);
        cu::copy_h2d_async(p + 2 * L.expert_gu, L.edown.host + (size_t)expert * L.expert_down, L.expert_down,
                           W().side);
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

    // ---- P5b: the routed experts of one layer, GPU part.  The int8 kernels run on the side stream so the CPU
    // experts of the same token are computed while the GPU works; `finish` joins the stream and copies the rows back.
    bool gpu_experts_begin(const Layer& L, const std::vector<int>& exps, const float* in) {
        const int k = (int)exps.size();
        if (k == 0) return false;
        use_device_index(L.dev);
        Work& w = W();
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
        void** ptr = (void**)w.ex_ptr.p;
        cu::copy_h2d(ptr, gp.data(), (size_t)k * sizeof(void*));
        cu::copy_h2d(ptr + top, up.data(), (size_t)k * sizeof(void*));
        cu::copy_h2d(ptr + 2 * top, dp.data(), (size_t)k * sizeof(void*));
        const int n_embd = (int)cfg.n_embd, n_ff = (int)cfg.n_ff_exp;
        const int gt = L.egate.type, dt = L.edown.type;
        if (!(iq_expert_type(gt) && iq_expert_type(dt))) {
            // F32/F16/BF16 experts: the existing kernel synchronizes itself, so no overlap is offered.
            return false;
        }
        if (cfg.n_embd % 32 != 0 || cfg.n_ff_exp % 32 != 0)
            fail("the GPU IQ expert path needs rounded 32-wide activations");
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
            if (!w.ex_sg.p) w.ex_sg = DevPtr(cu::alloc(bytes));
            if (!w.ex_su.p) w.ex_su = DevPtr(cu::alloc(bytes));
            if (!w.ex_sd.p) w.ex_sd = DevPtr(cu::alloc(bytes));
            cu::copy_h2d(w.ex_sg.p, a.data(), (size_t)k * sizeof(float));
            cu::copy_h2d(w.ex_su.p, b.data(), (size_t)k * sizeof(float));
            cu::copy_h2d(w.ex_sd.p, c.data(), (size_t)k * sizeof(float));
            sg = (const float*)w.ex_sg.p;
            su = (const float*)w.ex_su.p;
            sd = (const float*)w.ex_sd.p;
        }
        if (!w.side) w.side = cu::stream_create();
        void* st = w.side;
        h2d(w, w.x.p, in, (size_t)n_embd * sizeof(float));
        strata::kernels::quantize_q8_1_rows((const float*)w.x.p, 1, n_embd, w.ex_xq.p, st);
        strata::kernels::minimax_moe_gu(gt, (const void* const*)ptr, (const void* const*)(ptr + top), k, n_embd,
                                        n_ff, L.egate.row_bytes, w.ex_xq.p, (float*)w.ex_h.p, alpha, limit, st, sg,
                                        su);
        strata::kernels::quantize_q8_1_rows((const float*)w.ex_h.p, k, n_ff, w.ex_hq.p, st);
        strata::kernels::minimax_moe_down(dt, (const void* const*)(ptr + 2 * top), k, n_embd, n_ff,
                                          L.edown.row_bytes, w.ex_hq.p, (float*)w.ex_out.p, st, sd);
        return true;
    }

    void gpu_experts_finish(int k, std::vector<float>& out) {
        Work& w = W();
        cu::stream_sync(w.side);
        out.resize((size_t)k * (size_t)cfg.n_embd);
        cu::copy_d2h(out.data(), w.ex_out.p, out.size() * sizeof(float));
    }

    // The synchronizing form (raw expert types, or a call site that does not split GPU and CPU work).
    void gpu_experts(const Layer& L, const std::vector<int>& exps, const float* in, std::vector<float>& out) {
        const int k = (int)exps.size();
        if (k == 0) return;
        use_device_index(L.dev);
        if (gpu_experts_begin(L, exps, in)) {
            gpu_experts_finish(k, out);
            return;
        }
        Work& w = W();
        std::vector<const void*> gp((size_t)k), up((size_t)k), dp((size_t)k);
        for (int i = 0; i < k; ++i) {
            const int e = exps[(size_t)i];
            const int slot = L.vram_of[(size_t)e];
            const uint8_t* base = (const uint8_t*)L.vram[(size_t)slot].mem.p;
            gp[(size_t)i] = base;
            up[(size_t)i] = base + L.expert_gu;
            dp[(size_t)i] = base + 2 * L.expert_gu;
        }
        const int top = (int)cfg.n_expert_used;
        void** ptr = (void**)w.ex_ptr.p;
        cu::copy_h2d(ptr, gp.data(), (size_t)k * sizeof(void*));
        cu::copy_h2d(ptr + top, up.data(), (size_t)k * sizeof(void*));
        cu::copy_h2d(ptr + 2 * top, dp.data(), (size_t)k * sizeof(void*));
        const int n_embd = (int)cfg.n_embd, n_ff = (int)cfg.n_ff_exp;
        const int gt = L.egate.type, dt = L.edown.type;
        if (raw_expert_type(gt) && raw_expert_type(dt)) {
            if (gt != dt)
                fail("the GPU F32/F16/BF16 expert path needs the same type for gate/up and down");
            h2d(w, w.x.p, in, (size_t)n_embd * sizeof(float));
            cu::moe_expert_f16(gt, (const void* const*)ptr, (const void* const*)(ptr + top),
                               (const void* const*)(ptr + 2 * top), k, n_embd, n_ff, L.egate.row_bytes,
                               L.edown.row_bytes, (const float*)w.x.p, (float*)w.ex_h.p, (float*)w.ex_out.p, alpha,
                               limit);
        } else {
            fail("no GPU expert kernel for gate/up ggml type " + std::to_string(gt) + " and down type " +
                 std::to_string(dt) + " (supported: F32/F16/BF16 and the IQ/Q8_0 types)");
        }
        out.resize((size_t)k * (size_t)n_embd);
        cu::copy_d2h(out.data(), w.ex_out.p, out.size() * sizeof(float));
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

        // P5b: start the GPU experts on the side stream, compute the CPU ones while it runs, then join and combine.
        std::vector<float> gpu_out;
        bool gpu_async = false;
        if (!gpu_e.empty()) {
            use_device_index(L.dev);
            gpu_async = gpu_experts_begin(L, gpu_e, in);
        }
        std::vector<const CpuExpert*> cpu_of((size_t)cfg.n_expert, nullptr);
        for (const CpuExpert& c : cpu) cpu_of[(size_t)c.e] = &c;
        std::vector<std::vector<float>> cpu_ye((size_t)top);
        for (int j = 0; j < top; ++j) {
            const int e = order[(size_t)j];
            if (L.vram_of[(size_t)e] >= 0) continue;
            const CpuExpert* c = cpu_of[(size_t)e];
            if (!c) fail("internal: routed expert has no source");
            cpu_ye[(size_t)j].resize((size_t)cfg.n_embd);
            const float* cg = L.egate_scale.empty() ? nullptr : &L.egate_scale[(size_t)e];
            const float* cu = L.eup_scale.empty() ? nullptr : &L.eup_scale[(size_t)e];
            const float* cd = L.edown_scale.empty() ? nullptr : &L.edown_scale[(size_t)e];
            mlp_ptr(L.egate, c->gate, L.eup, c->up, L.edown, c->down, in, cpu_ye[(size_t)j].data(),
                    (int)cfg.n_ff_exp, cg, cu, cd);
        }
        if (!gpu_e.empty()) {
            if (gpu_async) gpu_experts_finish((int)gpu_e.size(), gpu_out);
            else gpu_experts(L, gpu_e, in, gpu_out);
        }

        std::fill(yout, yout + cfg.n_embd, 0.f);
        std::vector<float> ye((size_t)cfg.n_embd);
        size_t gi = 0;
        for (int j = 0; j < top; ++j) {
            const int e = order[(size_t)j];
            const float* src;
            if (L.vram_of[(size_t)e] >= 0) {
                src = gpu_out.data() + gi * (size_t)cfg.n_embd;
                ++gi;
            } else {
                src = cpu_ye[(size_t)j].data();
            }
            const float WJ = w[(size_t)j];
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += WJ * src[i];
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

    // ---- P5b: exact one-layer lookahead.  The residual after layer l IS layer l+1's input, so applying layer l+1's
    // router to it gives the real routes.  Free VRAM slots are filled on the side stream while layer l's expert math
    // finishes; the next layer joins the stream before it runs its MoE.  Only empty slots are used (no eviction), and
    // it is opt-in (--minimax-expert-prefetch N) because a card behind a narrow PCIe link should not stream.
    int prefetch_wait = -1;

    void prefetch_for(int64_t li) {
        if (prefetch_slots <= 0 || !tiers_gpu || li < 0 || li >= cfg.n_layers) return;
        Layer& L = layers[(size_t)li];
        if (!L.moe || L.vram.empty() || !L.router.host) return;
        use_layer_device(li);
        Work& w = W();
        if (!w.side) w.side = cu::stream_create();
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        std::vector<float> logits((size_t)nexp), score((size_t)nexp), prob((size_t)nexp);
        // The router reads the post-FFN-norm input, not the raw residual (moe() applies ffn_norm first).
        std::vector<float> hnorm((size_t)cfg.n_embd);
        rms(x.data(), L.ffn_norm.data(), hnorm.data(), (int)cfg.n_embd, eps);
        gemv(L.router, hnorm.data(), logits.data());
        for (int i = 0; i < nexp; ++i) {
            prob[(size_t)i] = sigmoid(logits[(size_t)i]);
            score[(size_t)i] = prob[(size_t)i] + L.bias[(size_t)i];
        }
        std::vector<int> order((size_t)nexp);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
            if (score[(size_t)a] != score[(size_t)b]) return score[(size_t)a] > score[(size_t)b];
            return a < b;
        });
        int started = 0;
        for (int j = 0; j < top && started < prefetch_slots; ++j) {
            const int e = order[(size_t)j];
            if (L.vram_of[(size_t)e] >= 0) continue;
            int free_slot = -1;
            for (int sl = 0; sl < (int)L.vram.size(); ++sl)
                if (L.vram[(size_t)sl].expert < 0) { free_slot = sl; break; }
            if (free_slot < 0) break;
            Layer::VramSlot& slot = L.vram[(size_t)free_slot];
            uint8_t* dst = (uint8_t*)slot.mem.p;
            cu::copy_h2d_async(dst, L.egate.host + (size_t)e * L.expert_gu, L.expert_gu, w.side);
            cu::copy_h2d_async(dst + L.expert_gu, L.eup.host + (size_t)e * L.expert_gu, L.expert_gu, w.side);
            cu::copy_h2d_async(dst + 2 * L.expert_gu, L.edown.host + (size_t)e * L.expert_down, L.expert_down,
                               w.side);
            slot.expert = e;
            L.vram_of[(size_t)e] = free_slot;
            ++started;
        }
        if (started) prefetch_wait = (int)li;
    }

    void prefetch_join(int64_t li) {
        if (prefetch_wait != (int)li) return;
        use_layer_device(li);
        Work& w = W();
        if (w.side) cu::stream_sync(w.side);
        prefetch_wait = -1;
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
        if (any) {
            for (size_t d = 0; d < work.size(); ++d) {
                if (!work[d].side) continue;
                use_device_index((int)d);
                cu::stream_sync(work[d].side);
            }
        }
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
            Work& w = W();
            h2d(w, w.x.p, xin, (size_t)cols * sizeof(float));
            float* dy = (float*)w.y.p;
            if (t.dev_f32) {
                cu::gemv((const float*)t.dev.p, (const float*)w.x.p, dy, rows, cols);
            } else {
                for (int r = 0; r < rows; r += kTile) {
                    const int n = rows - r < kTile ? rows - r : kTile;
                    cu::dequant_f32(t.type, t.dev.p, r, n, cols, (float*)w.scratch.p);
                    cu::gemv((const float*)w.scratch.p, (const float*)w.x.p, dy + r, n, cols);
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
        const int64_t li = (int64_t)(&L - layers.data());
        const bool dbg = dumping(li);
        const std::string pfx = "L" + std::to_string(src_layer[(size_t)li]);
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
        if (dbg) {
            std::vector<int64_t> ids((size_t)top);
            for (int j = 0; j < top; ++j) ids[(size_t)j] = order[(size_t)j];
            dbuf_put(pfx + "/router_logits", logits_r.data(), nexp);
            dbuf_put(pfx + "/router_weights", w.data(), top);
            dbuf_put_i(pfx + "/router_topk_ids", ids.data(), top);
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
            if (dbg) {
                const int e = order[(size_t)j];
                const std::string ep = pfx + "/expert_" + (e < 100 ? (e < 10 ? "00" : "0") : "") + std::to_string(e);
                std::vector<float> gu((size_t)ff * 2);
                std::memcpy(gu.data(), ff_g.data(), (size_t)ff * sizeof(float));
                std::memcpy(gu.data() + ff, ff_u.data(), (size_t)ff * sizeof(float));
                std::vector<float> ew((size_t)cfg.n_embd);
                for (int64_t i = 0; i < cfg.n_embd; ++i) ew[(size_t)i] = WJ * ye[(size_t)i];
                const int64_t tk = cur_pos;
                dbuf_put(ep + "_gate_up", gu.data(), ff * 2);
                dbuf_put(ep + "_hidden", ff_h.data(), ff);
                dbuf_put(ep + "_out", ew.data(), (int)cfg.n_embd);
                dbuf_put_i(ep + "_token_idx", &tk, 1);
            }
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += WJ * ye[(size_t)i];
        }
        const float s = scale_routed;
        for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] *= s;
        if (L.have_shared) {
            mlp(L.gate, L.up, L.down, 0, in, ye.data(), (int)L.gate.ne1);
            if (dbg) dbuf_put(pfx + "/shared_expert_out", ye.data(), (int)cfg.n_embd);
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += ye[(size_t)i];
        }
        if (dbg) dbuf_put(pfx + "/moe_out", yout, (int)cfg.n_embd);
    }

    const uint16_t* k_at(int layer, int64_t pos, int kv) const {
        const size_t stride = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        return k_host + ((size_t)layer * (size_t)ctx + (size_t)pos) * stride + (size_t)kv * (size_t)cfg.head_dim;
    }
    const uint16_t* v_at(int layer, int64_t pos, int kv) const {
        const size_t stride = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        return v_host + ((size_t)layer * (size_t)ctx + (size_t)pos) * stride + (size_t)kv * (size_t)cfg.head_dim;
    }
    const uint16_t* ik_at(int layer, int64_t pos) const {
        return ik_host + ((size_t)layer * (size_t)ctx + (size_t)pos) * (size_t)cfg.idx_dim;
    }

    void store_f16(uint16_t* host, size_t off, const float* src, int n) {
        for (int i = 0; i < n; ++i) host[off + (size_t)i] = ggml_fp32_to_fp16(src[i]);
    }
    void to_f16(const float* src, uint16_t* dst, int n) {
        for (int i = 0; i < n; ++i) dst[i] = ggml_fp32_to_fp16(src[i]);
    }

    // The MSA selection, shared by the host and device attention paths: local blocks are forced in, the rest are
    // ranked by their max-pooled indexer score, and the selected blocks expand to token positions.
    void select_slots(int64_t li, int64_t pos, const float* bs_in, int nb, std::vector<int32_t>& slots,
                      std::vector<int>& off) {
        const int HI = (int)cfg.idx_heads;
        const int BLK = (int)cfg.idx_block;
        std::vector<float> bs(bs_in, bs_in + (size_t)HI * nb);
        for (int h = 0; h < HI; ++h) {
            const int qb = (int)(pos / BLK);
            for (int l = 0; l < (int)cfg.idx_local; ++l) {
                int bi = qb - l;
                if (bi < 0) bi = 0;
                if (bi < nb) bs[(size_t)h * nb + bi] = std::numeric_limits<float>::infinity();
            }
        }
        const int kk = (int)cfg.idx_top_k < nb ? (int)cfg.idx_top_k : nb;
        std::vector<char> sel((size_t)HI * (size_t)nb, 0);
        std::vector<int> order((size_t)nb);
        std::vector<int64_t> ids((size_t)HI * (size_t)cfg.idx_top_k, -1);   // probe dump, -1 padded
        for (int h = 0; h < HI; ++h) {
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int a, int b) {
                const float sa = bs[(size_t)h * nb + a];
                const float sb = bs[(size_t)h * nb + b];
                if (sa != sb) return sa > sb;
                return a < b;
            });
            int at = 0;
            for (int i = 0; i < kk; ++i) {
                const int b = order[(size_t)i];
                if (bs[(size_t)h * nb + b] == -std::numeric_limits<float>::infinity()) continue;
                sel[(size_t)h * nb + b] = 1;
                ids[(size_t)h * (size_t)cfg.idx_top_k + (size_t)at++] = b;
            }
        }
        const int64_t Tk = pos + 1;
        off.assign((size_t)HI + 1, 0);
        slots.clear();
        const int64_t cap = (int64_t)cfg.idx_top_k * BLK;
        slots.reserve((size_t)HI * (size_t)(cap < 2048 ? cap : 2048));
        for (int ih = 0; ih < HI; ++ih) {
            off[(size_t)ih] = (int)slots.size();
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
        off[(size_t)HI] = (int)slots.size();

        if (dumping(li)) {
            const std::string pfx = "L" + std::to_string(src_layer[(size_t)li]);
            const int64_t span = dump_seq > pos + 1 ? dump_seq : pos + 1;
            const int nbmax = (int)((span + BLK - 1) / BLK);
            std::vector<float> padded((size_t)HI * (size_t)nbmax, -std::numeric_limits<float>::infinity());
            for (int h = 0; h < HI; ++h)
                for (int b = 0; b < nb; ++b) padded[(size_t)h * nbmax + b] = bs[(size_t)h * nb + b];
            dbuf_put(pfx + "/index_block_scores", padded.data(), HI * nbmax);
            dbuf_put_i(pfx + "/index_block_indices", ids.data(), HI * (int)cfg.idx_top_k);
        }
    }

    void attend_host(int layer, const Layer& L, int64_t pos, float* att) {
        const int H = (int)cfg.n_head;
        const int HKV = (int)cfg.n_head_kv;
        const int D = (int)cfg.head_dim;
        const int64_t Tk = pos + 1;
        const int q_per_kv = H / HKV;
        const float scale = 1.f / std::sqrt((float)D);
        int nb = 0;
        std::vector<int32_t> slots;
        std::vector<int> slot_off;
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
            }
            select_slots(layer, pos, bs.data(), nb, slots, slot_off);
        }
        const int q_per_idx = L.indexer ? H / (int)cfg.idx_heads : H;
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

    // P4: the same attention on the owning device.  The indexer scores are computed from the device copy of the
    // indexer keys; only the selected slots of each KV head are brought into the staging buffer (a device gather
    // when the KV lives in VRAM, one H2D copy from pinned host memory in the RAM tier).
    void attend_dev(int layer, const Layer& L, int64_t pos, float* att) {
        use_device_index(L.dev);
        Work& w = W();
        const int H = (int)cfg.n_head;
        const int HKV = (int)cfg.n_head_kv;
        const int D = (int)cfg.head_dim;
        const int HI = (int)cfg.idx_heads;
        const int BLK = (int)cfg.idx_block;
        const int64_t Tk = pos + 1;
        const int nb = (int)((Tk + BLK - 1) / BLK);
        std::vector<float> bs((size_t)HI * nb, -std::numeric_limits<float>::infinity());
        if (L.indexer) {
            h2d(w, w.qd.p, iq.data(), (size_t)HI * (size_t)cfg.idx_dim * sizeof(float));
            cu::idx_block_scores(L.d_ik.p, Tk, (const float*)w.qd.p, HI, (int)cfg.idx_dim, BLK, nb,
                                 (float*)w.bs.p, nullptr);
            cu::sync();
            cu::copy_d2h(bs.data(), w.bs.p, (size_t)HI * (size_t)nb * sizeof(float));
        }
        std::vector<int32_t> slots;
        std::vector<int> off;
        select_slots(layer, pos, bs.data(), nb, slots, off);

        // Per KV head: the slot list of the indexer head that feeds it (HI == HKV is enforced at startup).
        const int q_per_kv = H / HKV;
        std::vector<int32_t> sp((size_t)HKV * (size_t)max_slot, 0);
        std::vector<int32_t> cnt((size_t)HKV, 0);
        for (int h = 0; h < HKV; ++h) {
            const int a = off[(size_t)h], b = off[(size_t)h + 1];
            const int n = b - a;
            if (n > (int)max_slot) fail("MSA selected " + std::to_string(n) + " slots, more than --idx-top-k blocks");
            cnt[(size_t)h] = n;
            for (int i = 0; i < n; ++i) sp[(size_t)h * (size_t)max_slot + (size_t)i] = slots[(size_t)(a + i)];
        }
        cu::copy_h2d(w.slots.p, sp.data(), sp.size() * sizeof(int32_t));
        cu::copy_h2d(w.counts.p, cnt.data(), cnt.size() * sizeof(int32_t));
        if (kv_mode == 1) {
            cu::kv_gather(L.d_k.p, L.d_v.p, HKV, D, (const int32_t*)w.slots.p, (const int32_t*)w.counts.p,
                          (int)max_slot, w.kstage.p, w.vstage.p, nullptr);
        } else {
            // RAM tier: the selected rows are copied out of the page-locked K/V into a pinned staging buffer,
            // then one H2D per KV head brings only the used prefix across.
            const size_t row = (size_t)D * sizeof(uint16_t);
            for (int h = 0; h < HKV; ++h) {
                const int n = cnt[(size_t)h];
                for (int i = 0; i < n; ++i) {
                    const int32_t t = sp[(size_t)h * (size_t)max_slot + (size_t)i];
                    const size_t src = ((size_t)layer * (size_t)ctx + (size_t)t) * (size_t)HKV * (size_t)D +
                                       (size_t)h * (size_t)D;
                    const size_t dst = ((size_t)h * (size_t)max_slot + (size_t)i) * (size_t)D;
                    std::memcpy(w.kstage_h + dst, k_host + src, row);
                    std::memcpy(w.vstage_h + dst, v_host + src, row);
                }
                if (n > 0) {
                    const size_t off = (size_t)h * (size_t)max_slot * (size_t)D;
                    cu::copy_h2d((uint16_t*)w.kstage.p + off, w.kstage_h + off, (size_t)n * row);
                    cu::copy_h2d((uint16_t*)w.vstage.p + off, w.vstage_h + off, (size_t)n * row);
                }
            }
        }
        h2d(w, w.qd.p, q.data(), (size_t)H * (size_t)D * sizeof(float));
        cu::msa_attn(w.kstage.p, w.vstage.p, (const float*)w.qd.p, (float*)w.att.p, H, q_per_kv, (int)max_slot,
                     (const int32_t*)w.counts.p, D, nullptr);
        cu::sync();
        cu::copy_d2h(att, w.att.p, (size_t)H * (size_t)D * sizeof(float));
    }

    void attend(int layer, const Layer& L, int64_t pos, float* att) {
        if (kv_mode == 0)
            attend_host(layer, L, pos, att);
        else
            attend_dev(layer, L, pos, att);
    }


    // ===================================================================== P5a: batched prompt processing

    void init_prefill() {
        prefetch_slots = opt.expert_prefetch;
        if (opt.prefill_batch <= 1) return;
        prefill_dense_f32 = opt.prefill_dense != "f16";
        prefill_direct = opt.prefill_read == "direct";
        if (dump_on) {
            std::fprintf(stderr, "strata-minimax: --minimax-prefill-batch is off because the parity dump needs the "
                                 "per-token path\n");
            return;
        }
        batch_cap = opt.prefill_batch;
        if (batch_cap > (int)ctx) batch_cap = (int)ctx;
        rows_cap = batch_cap < 1024 ? batch_cap : 1024;
        prefill_gpu_experts = opt.prefill_experts != "cpu";
        if (prefill_gpu_experts && devs.empty()) {
            prefill_gpu_experts = false;
        }
        if (max_expert_bytes == 0)
            for (const Layer& L : layers)
                if (L.moe && L.expert_bytes > max_expert_bytes) max_expert_bytes = L.expert_bytes;
        max_dense_n = cfg.n_ff_dense > cfg.n_head * cfg.head_dim ? cfg.n_ff_dense : cfg.n_head * cfg.head_dim;
        if (cfg.n_ff_exp > max_dense_n) max_dense_n = cfg.n_ff_exp;
        if (cfg.n_embd > max_dense_n) max_dense_n = cfg.n_embd;
        if (cfg.n_expert > max_dense_n) max_dense_n = cfg.n_expert;

        // Per-token device bytes: two fp16 hidden-state buffers, the fp32 GEMM output and the two fp32 MoE buffers.
        const size_t per_tok = (size_t)cfg.n_embd * 2 * 2 + (size_t)max_dense_n * 4 + (size_t)max_dense_n * 4 * 2 +
                               (size_t)(cfg.n_head * cfg.head_dim) * 2;
        const size_t fixed = (size_t)224 << 20;
        for (size_t d = 0; d < devs.size(); ++d) {
            const cu::MemInfo mi = cu::mem_on(devs[d]);
            const size_t reserve = ((size_t)std::max(opt.vram_reserve_mib, 256) << 20) + fixed;
            if (mi.free_bytes <= reserve) {
                std::fprintf(stderr,
                             "strata-minimax: batched prefill needs workspace on device %d but only %zu MiB is free. "
                             "Lower --minimax-prefill-batch or --minimax-dense-layers-gpu.\n",
                             devs[d], mi.free_bytes >> 20);
                batch_cap = 0;
                return;
            }
            const size_t fit = (mi.free_bytes - reserve) / per_tok;
            if ((size_t)batch_cap > fit) batch_cap = (int)fit;
        }
        if (batch_cap < 2) {
            std::fprintf(stderr, "strata-minimax: batched prefill does not fit in VRAM; keeping the token path\n");
            batch_cap = 0;
            return;
        }
        if (rows_cap > batch_cap) rows_cap = batch_cap;

        if (prefill_gpu_experts && prefill_direct && ssd_files.empty()) {
            ssd_files.resize(model.size());
            for (size_t sh = 0; sh < model.size(); ++sh) {
                ssd_files[sh] = std::make_unique<strata::platform::DirectFile>();
                std::string err;
                if (!ssd_files[sh]->open(model.shard(sh).path(), err)) fail("batched prefill: " + err);
            }
            ssd_tmp = strata::platform::DirectFile::alloc_aligned((size_t)max_expert_bytes + 4096);
            if (!ssd_tmp) fail("batched prefill: cannot allocate the aligned read buffer");
            ssd_tmp_bytes = (size_t)max_expert_bytes + 4096;
        }
        pf.resize(devs.size());
        for (size_t d = 0; d < devs.size(); ++d) {
            use_device_index((int)d);
            PfBuf& p = pf[d];
            const size_t T = (size_t)batch_cap;
            const size_t embd = (size_t)cfg.n_embd;
            p.x16 = (uint16_t*)cu::alloc(T * embd * 2);
            p.h16 = (uint16_t*)cu::alloc(T * embd * 2);
            p.y32 = (float*)cu::alloc(T * (size_t)max_dense_n * 4);
            p.p32 = (float*)cu::alloc(T * (size_t)max_dense_n * 4);
            p.sh = (float*)cu::alloc(T * (size_t)max_dense_n * 4);
            p.w16_elems = std::min<int64_t>(4096, max_dense_n) * (int64_t)cfg.n_embd;
            p.w16 = (uint16_t*)cu::alloc((size_t)p.w16_elems * 2);
            p.w32 = (float*)cu::alloc((size_t)p.w16_elems * 4);
            p.rows_cap = rows_cap;
            const size_t rc = (size_t)rows_cap;
            p.gx = (uint16_t*)cu::alloc(rc * embd * 2);
            p.gx32 = (float*)cu::alloc(rc * embd * 4);
            p.gy = (float*)cu::alloc(rc * (size_t)cfg.n_ff_exp * 4);
            p.gu = (float*)cu::alloc(rc * (size_t)cfg.n_ff_exp * 4);
            p.gh = (uint16_t*)cu::alloc(rc * (size_t)cfg.n_ff_exp * 2);
            p.gh32 = (float*)cu::alloc(rc * (size_t)cfg.n_ff_exp * 4);
            p.gd = (float*)cu::alloc(rc * embd * 4);
            p.xq = cu::alloc(rc * (size_t)q8_1_bytes(cfg.n_embd));
            p.hq = cu::alloc(rc * (size_t)q8_1_bytes(cfg.n_ff_exp));
            p.d_rows = DevPtr(cu::alloc(rc * sizeof(int32_t)));
            p.d_ent_row = DevPtr(cu::alloc(rc * sizeof(int32_t)));
            p.d_ent_w = DevPtr(cu::alloc(rc * sizeof(float)));
            p.d_iq16 = DevPtr(cu::alloc(T * (size_t)cfg.idx_heads * (size_t)cfg.idx_dim * 2 + 4096));
            p.d_q16 = DevPtr(cu::alloc(T * (size_t)(cfg.n_head * cfg.head_dim) * 2 + 4096));
            const int64_t nbmax = (ctx + cfg.idx_block - 1) / cfg.idx_block;
            p.d_bs = DevPtr(cu::alloc(T * (size_t)cfg.idx_heads * (size_t)nbmax * sizeof(float) + 4096));
            pf_max_blk = (int)cfg.idx_top_k;
            p.d_blk = DevPtr(cu::alloc(T * (size_t)cfg.idx_heads * (size_t)pf_max_blk * sizeof(int32_t) + 4096));
            p.d_nblk = DevPtr(cu::alloc(T * (size_t)cfg.idx_heads * sizeof(int32_t) + 4096));
            if (prefill_gpu_experts && max_expert_bytes > 0) {
                p.ring_slot = max_expert_bytes;
                p.ring = cu::alloc(p.ring_slot);
                p.pin = cu::host_alloc(p.ring_slot);
            }
        }
        use_device_index(0);
        batch_on = true;
        pf_h.assign((size_t)batch_cap * (size_t)cfg.n_embd, 0.f);
        pf_q.assign((size_t)batch_cap * (size_t)(cfg.n_head * cfg.head_dim), 0.f);
        pf_kv.assign((size_t)batch_cap * (size_t)(cfg.n_head_kv * cfg.head_dim), 0.f);
        pf_router.assign((size_t)batch_cap * (size_t)cfg.n_expert, 0.f);
        pf_blk.assign((size_t)batch_cap * (size_t)cfg.idx_heads * (size_t)pf_max_blk, 0);
        pf_nblk.assign((size_t)batch_cap * (size_t)cfg.idx_heads, 0);
        pf_entries.reserve((size_t)batch_cap * (size_t)cfg.n_expert_used);
        pf_k16.assign((size_t)batch_cap * (size_t)(cfg.n_head_kv * cfg.head_dim), 0);
        pf_v16.assign((size_t)batch_cap * (size_t)(cfg.n_head_kv * cfg.head_dim), 0);
        pf_gate_h.assign((size_t)rows_cap * (size_t)cfg.n_ff_exp, 0.f);
        pf_up_h.assign((size_t)rows_cap * (size_t)cfg.n_ff_exp, 0.f);
        pf_hid_h.assign((size_t)rows_cap * (size_t)cfg.n_ff_exp, 0.f);
        pf_dense_h.assign((size_t)batch_cap * (size_t)max_dense_n, 0.f);
        pf_moe_h.assign((size_t)batch_cap * (size_t)cfg.n_embd, 0.f);
        pf_out_tile.assign((size_t)batch_cap * (size_t)std::min<int64_t>(4096, cfg.n_vocab), 0.f);
        std::fprintf(stderr, "INFO minimax-prefill batch=%d experts=%s\n", batch_cap,
                     prefill_gpu_experts ? "gpu" : "cpu");
    }

    PfBuf& P() { return pf[(size_t)cur]; }

    // One projection for T rows: dequantize a tile of the weight rows to fp16, cuBLAS the tile, repeat.  The tile
    // size comes from the scratch buffer, so an `lm_head` of 200 K rows costs a few dozen launches, not 200 K.
    // `row0`/`nrows` select a band of the weight rows (the lm_head is projected a vocabulary tile at a time so the
    // device never needs a 200 K x n_embd output buffer); -1/nrows<=0 means the whole matrix.
    void gemm_gpu(const Tensor& t, const uint16_t* X16, const float* X32, float* Y, int T, int ldy, int64_t row0 = -1,
                  int64_t nrows = 0) {
        if (!t.gpu) fail(t.name + ": batched GEMM asked for a host tensor");
        if (t.ne2 != 1) fail(t.name + ": batched GEMM is for a matrix");
        PfBuf& p = P();
        const int64_t N = t.ne1, K = t.ne0;
        if (prefill_dense_f32 && !X32) fail(t.name + ": the fp32 batched path has no activation buffer");
        const int64_t begin = row0 < 0 ? 0 : row0;
        const int64_t last = nrows > 0 ? begin + nrows : N;
        if (begin < 0 || last > N || last <= begin) fail(t.name + ": row band out of range");
        int64_t tile = p.w16_elems / K;
        if (tile > last - begin) tile = last - begin;
        if (tile < 1) fail(t.name + ": the batched-prefill weight scratch is smaller than one row");
        for (int64_t r0 = begin; r0 < last; r0 += tile) {
            const int64_t n = (last - r0) < tile ? (last - r0) : tile;
            const uint8_t* src = (const uint8_t*)t.dev.p + (size_t)r0 * t.row_bytes;
            if (prefill_dense_f32) {
                cu::dequant_to_f32(t.type, src, n * K, p.w32, nullptr);
                cu::gemm_f32(X32, p.w32, Y + (r0 - begin), T, n, K, ldy, 0.f, nullptr);
            } else {
                cu::dequant_to_f16(t.type, src, n * K, p.w16, nullptr);
                cu::gemm_f16(X16, p.w16, Y + (r0 - begin), T, n, K, ldy, 0.f, nullptr);
            }
        }
        cu::sync();
    }

    // The same projection on the CPU: every weight row is read once and dotted with all T activations.
    void gemm_cpu_base(const Tensor& t, const uint8_t* base, const float* X, int T, int ldx, float* Y, int ldy) {
        const int N = (int)t.ne1, K = (int)t.ne0;
        if (t.type == kF32 || t.type == kF16 || t.type == kBF16) {
            pool.run(N, [&](int a, int b) {
                for (int r = a; r < b; ++r) {
                    const uint8_t* row = base + (size_t)r * t.row_bytes;
                    for (int c = 0; c < T; ++c) {
                        const float* xt = X + (size_t)c * ldx;
                        float s;
                        if (t.type == kF32) s = dot_f32((const float*)row, xt, K);
                        else if (t.type == kF16) s = dot_f16w((const uint16_t*)row, xt, K);
                        else s = dot_bf16w((const uint16_t*)row, xt, K);
                        Y[(size_t)c * ldy + r] = s;
                    }
                }
            });
            return;
        }
        if (!t.cpu || !t.cpu->vec_dot) fail(t.name + ": ggml-cpu has no vec_dot for type " + std::to_string(t.type));
        const ggml_type_traits_cpu* act = ggml_get_type_traits_cpu(t.cpu->vec_dot_type);
        if (!act || !act->from_float) fail(t.name + ": ggml-cpu has no from_float for the activation type");
        const size_t act_bytes = ggml_row_size(t.cpu->vec_dot_type, K);
        std::vector<uint8_t> aq(act_bytes * (size_t)T);
        for (int c = 0; c < T; ++c) act->from_float(X + (size_t)c * ldx, aq.data() + (size_t)c * act_bytes, K);
        const ggml_vec_dot_t dot = t.cpu->vec_dot;
        pool.run(N, [&](int a, int b) {
            for (int r = a; r < b; ++r) {
                const uint8_t* row = base + (size_t)r * t.row_bytes;
                for (int c = 0; c < T; ++c) {
                    float s = 0.f;
                    dot(K, &s, 0, row, 0, aq.data() + (size_t)c * act_bytes, 0, 1);
                    Y[(size_t)c * ldy + r] = s;
                }
            }
        });
    }

    void gemm_cpu(const Tensor& t, const float* X, int T, int ldx, float* Y, int ldy) {
        gemm_cpu_base(t, t.host, X, T, ldx, Y, ldy);
    }

    void pf_select_blocks(int64_t li, int64_t pos, const float* bs, int nb, int32_t* out, int& nblk) {
        const int HI = (int)cfg.idx_heads;
        const int BLK = (int)cfg.idx_block;
        std::vector<float> s(bs, bs + (size_t)HI * nb);
        const int qb = (int)(pos / BLK);
        for (int h = 0; h < HI; ++h)
            for (int l = 0; l < (int)cfg.idx_local; ++l) {
                int bi = qb - l;
                if (bi < 0) bi = 0;
                if (bi < nb) s[(size_t)h * nb + bi] = std::numeric_limits<float>::infinity();
            }
        const int kk = (int)cfg.idx_top_k < nb ? (int)cfg.idx_top_k : nb;
        std::vector<int> order((size_t)nb);
        nblk = kk;
        for (int h = 0; h < HI; ++h) {
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int a, int b) {
                const float sa = s[(size_t)h * nb + a], sb = s[(size_t)h * nb + b];
                if (sa != sb) return sa > sb;
                return a < b;
            });
            for (int i = 0; i < kk; ++i) out[(size_t)h * pf_max_blk + i] = order[(size_t)i];
        }
        (void)li;
    }

    void mlp_batch_gpu(const Tensor& gate, const Tensor& up, const Tensor& down, const uint16_t* X16,
                       const float* X32, int T, int ff, float* out_host) {
        PfBuf& p = P();
        for (int c = 0; c < T; c += (int)p.rows_cap) {
            const int n = std::min<int>((int)p.rows_cap, T - c);
            gemm_gpu(gate, X16 + (size_t)c * cfg.n_embd, X32 ? X32 + (size_t)c * cfg.n_embd : nullptr, p.y32, n,
                     (int)gate.ne1);
            // `up` needs its own output: the gate tile is still in y32, so use p32 for up and swiglu from both.
            gemm_gpu(up, X16 + (size_t)c * cfg.n_embd, X32 ? X32 + (size_t)c * cfg.n_embd : nullptr, p.sh, n,
                     (int)up.ne1);
            if (prefill_dense_f32) {
                cu::swiglu_rows_f32(p.y32, p.sh, p.sh, n, ff, alpha, limit, nullptr);
                gemm_gpu(down, p.gh, p.sh, p.gd, n, (int)down.ne1);
            } else {
                cu::swiglu_rows_f16(p.y32, p.sh, p.gh, n, ff, alpha, limit, nullptr);
                gemm_gpu(down, p.gh, nullptr, p.gd, n, (int)down.ne1);
            }
            cu::copy_d2h(out_host + (size_t)c * cfg.n_embd, p.gd, (size_t)n * cfg.n_embd * sizeof(float));
        }
    }

    void forward_token(int32_t token, float* logit_out) {
        if (token < 0 || (int64_t)token >= cfg.n_vocab) fail("token id out of range");
        if (n_past >= ctx) fail("context is full (" + std::to_string(ctx) + ")");
        const int64_t pos = n_past;
        cur_pos = pos;
        tok_stats = TokenStats{};
        decay_usage();
        if (!seed_h.empty()) {
            if (pos >= seed_rows) fail("--minimax-seed-hidden has only " + std::to_string(seed_rows) + " rows");
            std::memcpy(x.data(), seed_h.data() + (size_t)pos * (size_t)cfg.n_embd, (size_t)cfg.n_embd * sizeof(float));
        } else {
            decode_f32(embd, embd.host + (size_t)token * embd.row_bytes, x.data(), cfg.n_embd);
        }

        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            if (kv_mode != 0 || L.want_gpu) use_layer_device(li);
            const bool dbg = dumping(li);
            const std::string pfx = "L" + std::to_string(src_layer[(size_t)li]);
            if (dbg) dbuf_put(pfx + "/residual_in", x.data(), (int)cfg.n_embd);
            rms(x.data(), L.attn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            if (dbg) dbuf_put(pfx + "/attn_input", h.data(), (int)cfg.n_embd);
            gemv(L.wq, h.data(), q.data());
            gemv(L.wk, h.data(), k.data());
            gemv(L.wv, h.data(), v.data());
            rms_heads(q.data(), (int)cfg.n_head, (int)cfg.head_dim, L.q_norm.data(), eps);
            rms_heads(k.data(), (int)cfg.n_head_kv, (int)cfg.head_dim, L.k_norm.data(), eps);
            rope(q.data(), (int)cfg.n_head, (int)cfg.head_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
            rope(k.data(), (int)cfg.n_head_kv, (int)cfg.head_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
            if (dbg) {
                dbuf_put(pfx + "/q_after_rope", q.data(), (int)(cfg.n_head * cfg.head_dim));
                dbuf_put(pfx + "/k_after_rope", k.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            }
            const size_t koff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
            const size_t koff_dev = (size_t)pos * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
            if (kv_mode == 0) {
                store_f16(k_host, koff, k.data(), (int)(cfg.n_head_kv * cfg.head_dim));
                store_f16(v_host, koff, v.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            } else {
                // The owning device's K/V (VRAM tier, one buffer per layer) or the page-locked host K/V (RAM tier).
                const int n = (int)(cfg.n_head_kv * cfg.head_dim);
                if (kv_mode == 2) {
                    store_f16(k_host, koff, k.data(), n);
                    store_f16(v_host, koff, v.data(), n);
                } else {
                    to_f16(k.data(), kv_tmp.data(), n);
                    cu::copy_h2d((uint16_t*)L.d_k.p + koff_dev, kv_tmp.data(), (size_t)n * 2);
                    to_f16(v.data(), kv_tmp.data(), n);
                    cu::copy_h2d((uint16_t*)L.d_v.p + koff_dev, kv_tmp.data(), (size_t)n * 2);
                }
            }
            if (L.indexer) {
                gemv(L.iq, h.data(), iq.data());
                gemv(L.ik, h.data(), ik.data());
                rms_heads(iq.data(), (int)cfg.idx_heads, (int)cfg.idx_dim, L.iq_norm.data(), eps);
                rms(ik.data(), L.ik_norm.data(), ik.data(), (int)cfg.idx_dim, eps);
                rope(iq.data(), (int)cfg.idx_heads, (int)cfg.idx_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
                rope(ik.data(), 1, (int)cfg.idx_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
                if (dbg) {
                    dbuf_put(pfx + "/index_q_after_rope", iq.data(), (int)(cfg.idx_heads * cfg.idx_dim));
                    dbuf_put(pfx + "/index_k_after_rope", ik.data(), (int)cfg.idx_dim);
                }
                const size_t ioff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.idx_dim;
                if (kv_mode == 0) {
                    store_f16(ik_host, ioff, ik.data(), (int)cfg.idx_dim);
                } else {
                    to_f16(ik.data(), kv_tmp.data(), (int)cfg.idx_dim);
                    cu::copy_h2d((uint16_t*)L.d_ik.p + (size_t)pos * (size_t)cfg.idx_dim, kv_tmp.data(),
                                 (size_t)cfg.idx_dim * 2);
                }
            }
            std::vector<float> att((size_t)cfg.n_head * (size_t)cfg.head_dim);
            attend((int)li, L, pos, att.data());
            if (dbg) dbuf_put(pfx + "/attn_core_out", att.data(), (int)(cfg.n_head * cfg.head_dim));
            gemv(L.wo, att.data(), proj.data());
            if (dbg) dbuf_put(pfx + "/attn_out", proj.data(), (int)cfg.n_embd);
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];

            rms(x.data(), L.ffn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            if (!L.moe) {
                mlp(L.gate, L.up, L.down, 0, h.data(), proj.data(), (int)cfg.n_ff_dense);
                if (dbg) {
                    std::vector<float> gu((size_t)cfg.n_ff_dense * 2);
                    std::memcpy(gu.data(), ff_g.data(), (size_t)cfg.n_ff_dense * sizeof(float));
                    std::memcpy(gu.data() + cfg.n_ff_dense, ff_u.data(), (size_t)cfg.n_ff_dense * sizeof(float));
                    dbuf_put(pfx + "/dense_gate_up", gu.data(), (int)(cfg.n_ff_dense * 2));
                    dbuf_put(pfx + "/dense_hidden", ff_h.data(), (int)cfg.n_ff_dense);
                    dbuf_put(pfx + "/dense_out", proj.data(), (int)cfg.n_embd);
                }
            } else {
                prefetch_join(li);
                moe(L, h.data(), proj.data());
            }
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];
            if (prefetch_slots > 0 && li + 1 < cfg.n_layers) prefetch_for(li + 1);
            if (dbg) dbuf_put(pfx + "/layer_out", x.data(), (int)cfg.n_embd);
        }

        rms(x.data(), out_norm.data(), h.data(), (int)cfg.n_embd, eps);
        if (output_gpu) use_device_index(layer_dev[(size_t)(cfg.n_layers - 1)]);
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

    // ===================================================================== P5a: batched MoE and the layer pass

    // sigmoid router for a whole chunk; ids/wts are [T][top] host, and the tier usage counters get the same votes
    // the token path would have cast.
    void pf_routes(const Layer& L, const float* logits, int T, std::vector<int>& ids, std::vector<float>& wts) {
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        ids.assign((size_t)T * top, 0);
        wts.assign((size_t)T * top, 0.f);
        const int64_t li = (int64_t)(&L - layers.data());
        const size_t base_idx = (size_t)li * (size_t)cfg.n_expert;
        std::vector<float> prob((size_t)nexp), score((size_t)nexp);
        std::vector<int> order((size_t)nexp);
        for (int t = 0; t < T; ++t) {
            const float* lg = logits + (size_t)t * nexp;
            for (int i = 0; i < nexp; ++i) {
                prob[(size_t)i] = sigmoid(lg[i]);
                score[(size_t)i] = prob[(size_t)i] + L.bias[(size_t)i];
            }
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
                if (score[(size_t)a] != score[(size_t)b]) return score[(size_t)a] > score[(size_t)b];
                return a < b;
            });
            float sum = 0.f;
            for (int j = 0; j < top; ++j) {
                ids[(size_t)t * top + j] = order[(size_t)j];
                wts[(size_t)t * top + j] = prob[(size_t)order[(size_t)j]];
                sum += wts[(size_t)t * top + j];
            }
            if (renorm && sum > 0.f)
                for (int j = 0; j < top; ++j) wts[(size_t)t * top + j] /= sum;
            if (!usage.empty())
                for (int j = 0; j < top; ++j) usage[base_idx + (size_t)order[(size_t)j]] += 1.f;
            if (!route_counts.empty())
                for (int j = 0; j < top; ++j) route_counts[base_idx + (size_t)order[(size_t)j]] += 1.0;
        }
    }

    void moe_batch_cpu(const Layer& L, const float* H, int T, float* yout) {
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        const int embd = (int)cfg.n_embd, ff = (int)cfg.n_ff_exp;
        std::vector<float> logits((size_t)T * nexp);
        gemm_cpu(L.router, H, T, embd, logits.data(), nexp);
        std::vector<int> ids;
        std::vector<float> wts;
        pf_routes(L, logits.data(), T, ids, wts);
        std::fill(yout, yout + (size_t)T * embd, 0.f);
        std::vector<std::vector<int>> tok_of((size_t)nexp);
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < top; ++s) tok_of[(size_t)ids[(size_t)t * top + s]].push_back(t);
        std::vector<float> xin((size_t)rows_cap * (size_t)embd);
        std::vector<float> hid((size_t)rows_cap * (size_t)ff);
        std::vector<float> yd((size_t)rows_cap * (size_t)embd);
        std::vector<float> g((size_t)rows_cap * (size_t)ff), u((size_t)rows_cap * (size_t)ff);
        for (int e = 0; e < nexp; ++e) {
            const std::vector<int>& toks = tok_of[(size_t)e];
            for (size_t a = 0; a < toks.size(); a += (size_t)rows_cap) {
                const int nt = (int)std::min<size_t>(toks.size() - a, (size_t)rows_cap);
                for (int i = 0; i < nt; ++i)
                    std::memcpy(xin.data() + (size_t)i * embd, H + (size_t)toks[a + (size_t)i] * embd,
                                (size_t)embd * sizeof(float));
                const uint8_t* gb = L.egate.host + (size_t)e * L.expert_gu;
                const uint8_t* ub = L.eup.host + (size_t)e * L.expert_gu;
                const uint8_t* db = L.edown.host + (size_t)e * L.expert_down;
                const float fg = L.egate_scale.empty() ? 1.f : L.egate_scale[(size_t)e];
                const float fu = L.eup_scale.empty() ? 1.f : L.eup_scale[(size_t)e];
                const float fd = L.edown_scale.empty() ? 1.f : L.edown_scale[(size_t)e];
                gemm_cpu_base(L.egate, gb, xin.data(), nt, embd, g.data(), ff);
                gemm_cpu_base(L.eup, ub, xin.data(), nt, embd, u.data(), ff);
                for (size_t i = 0; i < (size_t)nt * ff; ++i)
                    hid[i] = swigluoai(g[i] * fg, u[i] * fu, alpha, limit);
                gemm_cpu_base(L.edown, db, hid.data(), nt, ff, yd.data(), embd);
                if (fd != 1.f)
                    for (size_t i = 0; i < (size_t)nt * embd; ++i) yd[i] *= fd;
                for (int i = 0; i < nt; ++i) {
                    const int t = toks[a + (size_t)i];
                    float wsum = 0.f;
                    for (int s = 0; s < top; ++s)
                        if (ids[(size_t)t * top + s] == e) wsum += wts[(size_t)t * top + s];
                    float* dst = yout + (size_t)t * embd;
                    const float* src = yd.data() + (size_t)i * embd;
                    for (int c = 0; c < embd; ++c) dst[c] += wsum * src[c];
                }
            }
        }
        for (size_t i = 0; i < (size_t)T * embd; ++i) yout[i] *= scale_routed;
        if (L.have_shared) {
            const int sff = (int)L.gate.ne1;
            for (int c = 0; c < T; c += (int)rows_cap) {
                const int nt = std::min<int>((int)rows_cap, T - c);
                gemm_cpu(L.gate, H + (size_t)c * embd, nt, embd, g.data(), sff);
                gemm_cpu(L.up, H + (size_t)c * embd, nt, embd, u.data(), sff);
                for (size_t i = 0; i < (size_t)nt * sff; ++i) hid[i] = swigluoai(g[i], u[i], alpha, limit);
                gemm_cpu(L.down, hid.data(), nt, sff, yd.data(), embd);
                for (int i = 0; i < nt; ++i)
                    for (int c2 = 0; c2 < embd; ++c2)
                        yout[(size_t)(c + i) * embd + c2] += yd[(size_t)i * embd + c2];
            }
        }
    }

    void moe_batch_gpu(const Layer& L, const float* hdev, const uint16_t* h16, int T, float* yout_host) {
        PfBuf& p = P();
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        const int embd = (int)cfg.n_embd, ff = (int)cfg.n_ff_exp;
        gemm_gpu(L.router, h16, hdev, p.y32, T, nexp);
        cu::copy_d2h(pf_router.data(), p.y32, (size_t)T * nexp * sizeof(float));
        std::vector<int> ids;
        std::vector<float> wts;
        pf_routes(L, pf_router.data(), T, ids, wts);

        pf_entries.clear();
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < top; ++s)
                pf_entries.push_back({t, ids[(size_t)t * top + s], wts[(size_t)t * top + s], 0});
        std::stable_sort(pf_entries.begin(), pf_entries.end(),
                         [](const PfEntry& a, const PfEntry& b) { return a.expert < b.expert; });

        // Per projection, not per expert: the IQ2_XS pack has IQ2_XS gate/up and a Q2_0/IQ4_NL/F32 down, and MM-NVFP4
        // has its own type.  `gu_iq` / `d_iq` pick the int8 kernel or the dequantize-to-fp16 + cuBLAS fallback.
        const bool gu_iq = iq_expert_type(L.egate.type) && cfg.n_embd % 32 == 0;
        const bool d_iq = iq_expert_type(L.edown.type) && cfg.n_ff_exp % 32 == 0;
        const int gt = L.egate.type, dt = L.edown.type;
        bool zero = true;
        std::vector<int32_t> row_t((size_t)rows_cap);
        std::vector<float> row_w((size_t)rows_cap);
        size_t at = 0;
        while (at < pf_entries.size()) {
            const int e = pf_entries[at].expert;
            size_t end = at;
            while (end < pf_entries.size() && pf_entries[end].expert == e) ++end;
            const uint8_t* sg;
            const uint8_t* su;
            const uint8_t* sd;
            const int slot = (L.vram_of.empty() ? -1 : L.vram_of[(size_t)e]);
            if (slot >= 0) {
                const uint8_t* base = (const uint8_t*)L.vram[(size_t)slot].mem.p;
                sg = base;
                su = base + L.expert_gu;
                sd = base + 2 * L.expert_gu;
            } else {
                if (!p.ring) fail("batched prefill: no expert streaming buffer");
                uint8_t* dst = (uint8_t*)p.ring;
                if (p.pin && prefill_direct && !ssd_files.empty()) {
                    // Unbuffered 16 MiB reads from the shard into page-locked staging: the page cache cannot hold a
                    // 133 GB expert set, and a cached mmap copy is both slower and evicts other work.
                    uint8_t* pin = (uint8_t*)p.pin;
                    ssd_read_expert(L, e, pin);
                    cu::copy_h2d(dst, pin, L.expert_bytes);
                } else if (p.pin) {
                    uint8_t* pin = (uint8_t*)p.pin;
                    std::memcpy(pin, L.egate.host + (size_t)e * L.expert_gu, L.expert_gu);
                    std::memcpy(pin + L.expert_gu, L.eup.host + (size_t)e * L.expert_gu, L.expert_gu);
                    std::memcpy(pin + 2 * L.expert_gu, L.edown.host + (size_t)e * L.expert_down, L.expert_down);
                    cu::copy_h2d(dst, pin, L.expert_bytes);
                } else {
                    cu::copy_h2d(dst, L.egate.host + (size_t)e * L.expert_gu, L.expert_gu);
                    cu::copy_h2d(dst + L.expert_gu, L.eup.host + (size_t)e * L.expert_gu, L.expert_gu);
                    cu::copy_h2d(dst + 2 * L.expert_gu, L.edown.host + (size_t)e * L.expert_down, L.expert_down);
                }
                sg = dst;
                su = dst + L.expert_gu;
                sd = dst + 2 * L.expert_gu;
            }
            for (size_t a = at; a < end; a += (size_t)rows_cap) {
                const int nt = (int)std::min<size_t>(end - a, (size_t)rows_cap);
                for (int i = 0; i < nt; ++i) {
                    row_t[(size_t)i] = pf_entries[a + (size_t)i].tid;
                    row_w[(size_t)i] = pf_entries[a + (size_t)i].w;
                }
                cu::copy_h2d(p.d_rows.p, row_t.data(), (size_t)nt * sizeof(int32_t));
                cu::copy_h2d(p.d_ent_row.p, row_t.data(), (size_t)nt * sizeof(int32_t));
                cu::copy_h2d(p.d_ent_w.p, row_w.data(), (size_t)nt * sizeof(float));
                const float fg = L.egate_scale.empty() ? 1.f : L.egate_scale[(size_t)e];
                const float fu = L.eup_scale.empty() ? 1.f : L.eup_scale[(size_t)e];
                const float fd = L.edown_scale.empty() ? 1.f : L.edown_scale[(size_t)e];
                if (gu_iq) {
                    cu::gather_rows_f32(hdev, (const int32_t*)p.d_rows.p, nt, embd, p.gx32, nullptr);
                    strata::kernels::quantize_q8_1_rows(p.gx32, nt, embd, p.xq, nullptr);
                    strata::kernels::iq_mmvq(gt, sg, p.xq, p.gy, embd, ff, nt, nullptr);
                    strata::kernels::iq_mmvq(gt, su, p.xq, p.gu, embd, ff, nt, nullptr);
                } else if (prefill_dense_f32) {
                    cu::gather_rows_f32(hdev, (const int32_t*)p.d_rows.p, nt, embd, p.gx32, nullptr);
                    cu::dequant_to_f32(gt, sg, (int64_t)ff * embd, p.w32, nullptr);
                    cu::gemm_f32(p.gx32, p.w32, p.gy, nt, ff, embd, ff, 0.f, nullptr);
                    cu::dequant_to_f32(gt, su, (int64_t)ff * embd, p.w32, nullptr);
                    cu::gemm_f32(p.gx32, p.w32, p.gu, nt, ff, embd, ff, 0.f, nullptr);
                } else {
                    cu::gather_rows_f16(hdev, (const int32_t*)p.d_rows.p, nt, embd, p.gx, nullptr);
                    cu::dequant_to_f16(gt, sg, (int64_t)ff * embd, p.w16, nullptr);
                    cu::gemm_f16(p.gx, p.w16, p.gy, nt, ff, embd, ff, 0.f, nullptr);
                    cu::dequant_to_f16(gt, su, (int64_t)ff * embd, p.w16, nullptr);
                    cu::gemm_f16(p.gx, p.w16, p.gu, nt, ff, embd, ff, 0.f, nullptr);
                }
                if (fg != 1.f) cu::scale_rows(p.gy, (int64_t)nt * ff, fg, nullptr);
                if (fu != 1.f) cu::scale_rows(p.gu, (int64_t)nt * ff, fu, nullptr);
                cu::swiglu_rows_f32(p.gy, p.gu, p.gh32, nt, ff, alpha, limit, nullptr);
                if (d_iq) {
                    strata::kernels::quantize_q8_1_rows(p.gh32, nt, ff, p.hq, nullptr);
                    strata::kernels::iq_mmvq(dt, sd, p.hq, p.gd, ff, embd, nt, nullptr);
                } else if (prefill_dense_f32) {
                    cu::dequant_to_f32(dt, sd, (int64_t)embd * ff, p.w32, nullptr);
                    cu::gemm_f32(p.gh32, p.w32, p.gd, nt, embd, ff, embd, 0.f, nullptr);
                } else {
                    cu::f32_to_f16(p.gh32, p.gh, (int64_t)nt * ff, nullptr);
                    cu::dequant_to_f16(dt, sd, (int64_t)embd * ff, p.w16, nullptr);
                    cu::gemm_f16(p.gh, p.w16, p.gd, nt, embd, ff, embd, 0.f, nullptr);
                }
                if (fd != 1.f) cu::scale_rows(p.gd, (int64_t)nt * embd, fd, nullptr);
                cu::moe_combine_rows(p.gd, (const int32_t*)p.d_ent_row.p, (const float*)p.d_ent_w.p, nt, embd, p.sh,
                                     T, zero, nullptr);
                zero = false;
            }
            at = end;
        }
        cu::sync();
        cu::copy_d2h(yout_host, p.sh, (size_t)T * embd * sizeof(float));
        for (size_t i = 0; i < (size_t)T * embd; ++i) yout_host[i] *= scale_routed;
        if (L.have_shared) {
            // The shared expert is a plain 1-head dense MLP: run it through the batched projection path.
            PfBuf& q = P();
            const int sff = (int)L.gate.ne1;
            std::vector<float> sh((size_t)T * embd);
            // gate into y32, up into p32, swiglu to gh, down to gd, in row chunks of rows_cap
            for (int c = 0; c < T; c += (int)q.rows_cap) {
                const int nt = std::min<int>((int)q.rows_cap, T - c);
                gemm_gpu(L.gate, h16 + (size_t)c * embd, hdev + (size_t)c * embd, q.y32, nt, sff);
                gemm_gpu(L.up, h16 + (size_t)c * embd, hdev + (size_t)c * embd, q.sh, nt, sff);
                if (prefill_dense_f32) {
                    cu::swiglu_rows_f32(q.y32, q.sh, q.sh, nt, sff, alpha, limit, nullptr);
                    gemm_gpu(L.down, q.gh, q.sh, q.gd, nt, embd);
                } else {
                    cu::swiglu_rows_f16(q.y32, q.sh, q.gh, nt, sff, alpha, limit, nullptr);
                    gemm_gpu(L.down, q.gh, nullptr, q.gd, nt, embd);
                }
                cu::copy_d2h(sh.data() + (size_t)c * embd, q.gd, (size_t)nt * embd * sizeof(float));
            }
            for (size_t i = 0; i < (size_t)T * embd; ++i) yout_host[i] += sh[i];
        }
    }

    void forward_batch(const int32_t* toks, int T, float* logit_out, int logit_rows) {
        if (!batch_on) fail("internal: the batched path is off");
        if (T < 1) fail("empty batch");
        if (n_past + T > ctx) fail("context is full (" + std::to_string(ctx) + ")");
        const int64_t pos0 = n_past;
        const int ne = (int)cfg.n_embd;
        const int H = (int)cfg.n_head, HKV = (int)cfg.n_head_kv, D = (int)cfg.head_dim;
        const int HI = (int)cfg.idx_heads;
        const int qlen = H * D, klen = HKV * D;
        const int BLK = (int)cfg.idx_block;
        tok_stats = TokenStats{};
        const Tensor& embd_t = embd;

        for (int t = 0; t < T; ++t) {
            if (toks[t] < 0 || (int64_t)toks[t] >= cfg.n_vocab) fail("token id out of range");
            if (!seed_h.empty()) {
                if (pos0 + t >= seed_rows) fail("--minimax-seed-hidden has too few rows");
                std::memcpy(pf_h.data() + (size_t)t * ne, seed_h.data() + (size_t)(pos0 + t) * ne,
                            (size_t)ne * sizeof(float));
            } else {
                decode_f32(embd_t, embd_t.host + (size_t)toks[t] * embd_t.row_bytes, pf_h.data() + (size_t)t * ne,
                           ne);
            }
        }
        std::vector<float> att((size_t)qlen);          // one row's attention output (host path)
        std::vector<float> qf((size_t)T * qlen);       // attention queries, fp32 (host path and the q8 upload)
        std::vector<float> vf((size_t)T * klen);       // attention values
        std::vector<float> iqf, ikf;
        std::vector<uint16_t> iq16;
        std::vector<float> host_bs;

        const bool time_on = std::getenv("STRATA_MM_TIME") != nullptr;
        using pfclock = std::chrono::steady_clock;
        auto ms = [](pfclock::time_point a) {
            return std::chrono::duration<double, std::milli>(pfclock::now() - a).count();
        };
        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            const bool gpu = !devs.empty() && L.want_gpu;
            const auto t_layer = pfclock::now();
            const bool dev_attn = gpu && kv_mode == 1;
            if (gpu || kv_mode != 0) use_layer_device(li);
            PfBuf p_dummy;
            PfBuf& p = pf.empty() ? p_dummy : P();
            // ---- attention block
            for (int t = 0; t < T; ++t)
                rms(pf_h.data() + (size_t)t * ne, L.attn_norm.data(), pf_q.data() + (size_t)t * ne, ne, eps);
            const float* hq = pf_q.data();   // [T][ne] normed input
            if (gpu) {
                cu::copy_h2d(p.p32, hq, (size_t)T * ne * sizeof(float));
                cu::f32_to_f16(p.p32, p.h16, (size_t)T * ne, nullptr);
                cu::sync();
                gemm_gpu(L.wq, p.h16, p.p32, p.y32, T, qlen);
                cu::copy_d2h(qf.data(), p.y32, (size_t)T * qlen * sizeof(float));
                gemm_gpu(L.wk, p.h16, p.p32, p.y32, T, klen);
                cu::copy_d2h(pf_kv.data(), p.y32, (size_t)T * klen * sizeof(float));
                gemm_gpu(L.wv, p.h16, p.p32, p.y32, T, klen);
                cu::copy_d2h(vf.data(), p.y32, (size_t)T * klen * sizeof(float));
            } else {
                gemm_cpu(L.wq, hq, T, ne, qf.data(), qlen);
                gemm_cpu(L.wk, hq, T, ne, pf_kv.data(), klen);
                gemm_cpu(L.wv, hq, T, ne, vf.data(), klen);
            }
            for (int t = 0; t < T; ++t) {
                rms_heads(qf.data() + (size_t)t * qlen, H, D, L.q_norm.data(), eps);
                rms_heads(pf_kv.data() + (size_t)t * klen, HKV, D, L.k_norm.data(), eps);
                rope(qf.data() + (size_t)t * qlen, H, D, (int)cfg.n_rot, (int)(pos0 + t), cfg.rope_theta);
                rope(pf_kv.data() + (size_t)t * klen, HKV, D, (int)cfg.n_rot, (int)(pos0 + t), cfg.rope_theta);
                const size_t koff = ((size_t)li * (size_t)ctx + (size_t)(pos0 + t)) * (size_t)klen;
                if (kv_mode != 1) {
                    store_f16(k_host, koff, pf_kv.data() + (size_t)t * klen, klen);
                    store_f16(v_host, koff, vf.data() + (size_t)t * klen, klen);
                }
            }
            if (kv_mode == 1) {
                pf_k16.resize((size_t)T * klen);
                pf_v16.resize((size_t)T * klen);
                for (int t = 0; t < T; ++t) {
                    to_f16(pf_kv.data() + (size_t)t * klen, pf_k16.data() + (size_t)t * klen, klen);
                    to_f16(vf.data() + (size_t)t * klen, pf_v16.data() + (size_t)t * klen, klen);
                }
                cu::copy_h2d((uint16_t*)L.d_k.p + (size_t)pos0 * klen, pf_k16.data(), (size_t)T * klen * 2);
                cu::copy_h2d((uint16_t*)L.d_v.p + (size_t)pos0 * klen, pf_v16.data(), (size_t)T * klen * 2);
            }
            if (L.indexer) {
                const int ilen = HI * (int)cfg.idx_dim;
                iqf.resize((size_t)T * ilen);
                ikf.resize((size_t)T * cfg.idx_dim);
                if (gpu) {
                    gemm_gpu(L.iq, p.h16, p.p32, p.y32, T, ilen);
                    cu::copy_d2h(iqf.data(), p.y32, iqf.size() * sizeof(float));
                    gemm_gpu(L.ik, p.h16, p.p32, p.y32, T, (int)cfg.idx_dim);
                    cu::copy_d2h(ikf.data(), p.y32, ikf.size() * sizeof(float));
                } else {
                    gemm_cpu(L.iq, hq, T, ne, iqf.data(), ilen);
                    gemm_cpu(L.ik, hq, T, ne, ikf.data(), (int)cfg.idx_dim);
                }
                for (int t = 0; t < T; ++t) {
                    rms_heads(iqf.data() + (size_t)t * ilen, HI, (int)cfg.idx_dim, L.iq_norm.data(), eps);
                    rms(ikf.data() + (size_t)t * cfg.idx_dim, L.ik_norm.data(), ikf.data() + (size_t)t * cfg.idx_dim,
                        (int)cfg.idx_dim, eps);
                    rope(iqf.data() + (size_t)t * ilen, HI, (int)cfg.idx_dim, (int)cfg.n_rot, (int)(pos0 + t),
                         cfg.rope_theta);
                    rope(ikf.data() + (size_t)t * cfg.idx_dim, 1, (int)cfg.idx_dim, (int)cfg.n_rot, (int)(pos0 + t),
                         cfg.rope_theta);
                    const size_t ioff = ((size_t)li * (size_t)ctx + (size_t)(pos0 + t)) * (size_t)cfg.idx_dim;
                    if (kv_mode == 0)
                        store_f16(ik_host, ioff, ikf.data() + (size_t)t * cfg.idx_dim, (int)cfg.idx_dim);
                }
                if (kv_mode != 0) {
                    std::vector<uint16_t> ik16((size_t)T * cfg.idx_dim);
                    for (int t = 0; t < T; ++t)
                        to_f16(ikf.data() + (size_t)t * cfg.idx_dim, ik16.data() + (size_t)t * cfg.idx_dim,
                               (int)cfg.idx_dim);
                    cu::copy_h2d((uint16_t*)L.d_ik.p + (size_t)pos0 * cfg.idx_dim, ik16.data(),
                                 (size_t)T * cfg.idx_dim * 2);
                }
                if (dev_attn) {
                    iq16.resize((size_t)T * ilen);
                    for (int t = 0; t < T; ++t)
                        to_f16(iqf.data() + (size_t)t * ilen, iq16.data() + (size_t)t * ilen, ilen);
                }
            }
            if (dev_attn) {
                // Device attention: block scores on the device, top-k on the host (as the decode path does).
                int nb = 0;
                if (L.indexer) {
                    cu::copy_h2d(p.d_iq16.p, iq16.data(), iq16.size() * 2);
                    nb = (int)((pos0 + T + BLK - 1) / BLK);
                    cu::idx_block_scores_batch(L.d_ik.p, (const uint16_t*)p.d_iq16.p, HI, (int)cfg.idx_dim, BLK, nb, T,
                                               pos0, (float*)p.d_bs.p, nullptr);
                    cu::sync();
                    host_bs.resize((size_t)T * HI * (size_t)nb);
                    cu::copy_d2h(host_bs.data(), p.d_bs.p, host_bs.size() * sizeof(float));
                    for (int t = 0; t < T; ++t) {
                        int nblk = 0;
                        pf_select_blocks(li, pos0 + t, host_bs.data() + (size_t)t * HI * (size_t)nb, nb,
                                         pf_blk.data() + (size_t)t * HI * pf_max_blk, nblk);
                        for (int h = 0; h < HI; ++h) pf_nblk[(size_t)t * HI + h] = nblk;
                    }
                    cu::copy_h2d(p.d_blk.p, pf_blk.data(), pf_blk.size() * sizeof(int32_t));
                    cu::copy_h2d(p.d_nblk.p, pf_nblk.data(), pf_nblk.size() * sizeof(int32_t));
                }
                cu::copy_h2d(p.y32, qf.data(), (size_t)T * qlen * sizeof(float));
                cu::f32_to_f16(p.y32, (uint16_t*)p.d_q16.p, (size_t)T * qlen, nullptr);
                cu::sync();
                cu::msa_attn_batch(L.d_k.p, L.d_v.p, (const uint16_t*)p.d_q16.p, (const int32_t*)p.d_blk.p,
                                   (const int32_t*)p.d_nblk.p, H, H / HKV, L.indexer ? H / HI : 1, D, BLK,
                                   L.indexer ? 0 : 1, pos0, T, pf_max_blk, p.p32, nullptr);
                cu::sync();
                if (!prefill_dense_f32) cu::f32_to_f16(p.p32, p.h16, (size_t)T * qlen, nullptr);
                cu::sync();
                gemm_gpu(L.wo, p.h16, p.p32, p.y32, T, ne);
                cu::copy_d2h(pf_moe_h.data(), p.y32, (size_t)T * ne * sizeof(float));
            } else {
                // Host attention: the same per-token math as the decode path, over the chunk's own rows.
                for (int t = 0; t < T; ++t) {
                    std::memcpy(q.data(), qf.data() + (size_t)t * qlen, (size_t)qlen * sizeof(float));
                    if (L.indexer)
                        std::memcpy(iq.data(), iqf.data() + (size_t)t * (size_t)HI * cfg.idx_dim,
                                    (size_t)HI * cfg.idx_dim * sizeof(float));
                    attend_host((int)li, L, pos0 + t, att.data());
                    std::memcpy(pf_dense_h.data() + (size_t)t * qlen, att.data(), (size_t)qlen * sizeof(float));
                }
                gemm_cpu(L.wo, pf_dense_h.data(), T, qlen, pf_moe_h.data(), ne);
            }
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < ne; ++c) pf_h[(size_t)t * ne + c] += pf_moe_h[(size_t)t * ne + c];

            const auto t_ffn = pfclock::now();
            // ---- FFN block
            for (int t = 0; t < T; ++t)
                rms(pf_h.data() + (size_t)t * ne, L.ffn_norm.data(), pf_q.data() + (size_t)t * ne, ne, eps);
            const float* hn = pf_q.data();
            if (!L.moe) {
                const int ff = (int)L.gate.ne1;
                if (gpu) {
                    cu::copy_h2d(p.p32, hn, (size_t)T * ne * sizeof(float));
                    cu::f32_to_f16(p.p32, p.h16, (size_t)T * ne, nullptr);
                    cu::sync();
                    mlp_batch_gpu(L.gate, L.up, L.down, p.h16, p.p32, T, ff, pf_moe_h.data());
                } else {
                    for (int c = 0; c < T; c += (int)rows_cap) {
                        const int nt = std::min<int>((int)rows_cap, T - c);
                        gemm_cpu(L.gate, hn + (size_t)c * ne, nt, ne, pf_gate_h.data(), ff);
                        gemm_cpu(L.up, hn + (size_t)c * ne, nt, ne, pf_up_h.data(), ff);
                        for (size_t i = 0; i < (size_t)nt * ff; ++i)
                            pf_hid_h[i] = swigluoai(pf_gate_h[i], pf_up_h[i], alpha, limit);
                        gemm_cpu(L.down, pf_hid_h.data(), nt, ff, pf_dense_h.data(), ne);
                        std::memcpy(pf_moe_h.data() + (size_t)c * ne, pf_dense_h.data(),
                                    (size_t)nt * ne * sizeof(float));
                    }
                }
            } else if (gpu && prefill_gpu_experts) {
                cu::copy_h2d(p.p32, hn, (size_t)T * ne * sizeof(float));
                cu::f32_to_f16(p.p32, p.h16, (size_t)T * ne, nullptr);
                cu::sync();
                moe_batch_gpu(L, p.p32, p.h16, T, pf_moe_h.data());
            } else {
                moe_batch_cpu(L, hn, T, pf_moe_h.data());
            }
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < ne; ++c) pf_h[(size_t)t * ne + c] += pf_moe_h[(size_t)t * ne + c];
            if (time_on) {
                const double f = ms(t_ffn);
                std::fprintf(stderr, "INFO batch-time layer=%lld attn=%.1f ffn=%.1f moe=%d\n", (long long)li,
                             ms(t_layer) - f, f, L.moe ? 1 : 0);
            }
        }

        // ---- final norm and output projection
        const int rows = logit_rows <= 0 || logit_rows > T ? T : logit_rows;
        const int r0 = T - rows;
        for (int t = 0; t < T; ++t)
            rms(pf_h.data() + (size_t)t * ne, out_norm.data(), pf_q.data() + (size_t)t * ne, ne, eps);
        const int vocab = (int)cfg.n_vocab;
        if (output_gpu && !devs.empty()) {
            use_device_index(layer_dev[(size_t)(cfg.n_layers - 1)]);
            PfBuf& p = P();
            cu::copy_h2d(p.p32, pf_q.data() + (size_t)r0 * ne, (size_t)rows * ne * sizeof(float));
            cu::f32_to_f16(p.p32, p.h16, (size_t)rows * ne, nullptr);
            cu::sync();
            for (int n0 = 0; n0 < vocab; n0 += 4096) {
                const int nt = std::min(4096, vocab - n0);
                if ((int64_t)rows * nt > (int64_t)batch_cap * max_dense_n) fail("output tile exceeds the workspace");
                gemm_gpu(output, p.h16, p.p32, p.y32, rows, nt, n0, nt);
                cu::copy_d2h(pf_out_tile.data(), p.y32, (size_t)rows * nt * sizeof(float));
                for (int i = 0; i < rows; ++i)
                    std::memcpy(logit_out + (size_t)i * vocab + n0, pf_out_tile.data() + (size_t)i * nt,
                                (size_t)nt * sizeof(float));
            }
        } else {
            for (int t = 0; t < rows; ++t)
                gemm_cpu(output, pf_q.data() + (size_t)(r0 + t) * ne, 1, ne, logit_out + (size_t)t * vocab,
                         vocab);
        }
        n_past += T;
        cur_pos = pos0 + T - 1;
        if (tiers_on) {
            ++tokens_seen;
            total_stats.vram += tok_stats.vram;
            total_stats.ram += tok_stats.ram;
            total_stats.ssd += tok_stats.ssd;
            total_stats.ssd_bytes += tok_stats.ssd_bytes;
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
    o << "prefill=" << (impl_->batch_on ? "batch" : "token") << " dense_gpu_layers=" << impl_->dense_gpu
      << " output_gpu=" << (impl_->output_gpu ? 1 : 0) << " experts=";
    if (!impl_->tiers_on) {
        o << "cpu";
    } else {
        o << "tiered vram_slots_per_layer=" << impl_->vram_per_layer << " ram_experts_per_layer="
          << impl_->ram_per_layer << " profile=" << (impl_->profile.empty() ? "none" : "loaded")
          << " adaptive=" << (impl_->opt.adaptive ? 1 : 0);
    }
    o << " kv=f16 norm_add1=" << (impl_->norm_add1 ? 1 : 0) << " threads=" << impl_->threads
      << " ctx=" << impl_->ctx;
    if (impl_->devs.size() > 1) {
        o << " gpus=";
        for (size_t d = 0; d < impl_->devs.size(); ++d) {
            if (d) o << ",";
            o << impl_->devs[d];
        }
        o << " split=";
        for (size_t d = 0; d + 1 < impl_->devs.size(); ++d) {
            int first = 0;
            for (size_t i = 0; i < impl_->layer_dev.size(); ++i)
                if (impl_->layer_dev[i] == (int)d + 1) { first = (int)i; break; }
            if (d) o << ",";
            o << first;
        }
        o << " vram_used_mib=";
        for (size_t d = 0; d < impl_->devs.size(); ++d) {
            const cu::MemInfo mi = cu::mem_on(impl_->devs[d]);
            if (d) o << ",";
            o << ((mi.total_bytes - mi.free_bytes) >> 20);
        }
    }
    o << " kv_space=" << (impl_->kv_mode == 0 ? "host" : impl_->kv_mode == 1 ? "vram" : "ram");
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
    impl_->dump_seq = n;
    std::vector<float> all((size_t)n * vocab);
    const int batch = impl_->batch_on ? impl_->batch_cap : 1;
    for (int i = 0; i < n;) {
        const int k = std::min(batch, n - i);
        if (k > 1) {
            impl_->forward_batch(tokens.data() + i, k, all.data() + (size_t)i * vocab, k);
            for (int j = 0; j < k; ++j)
                if (!finite_row(all.data() + (size_t)(i + j) * vocab, vocab))
                    fail("non-finite logits at position " + std::to_string(i + j));
            i += k;
        } else {
            impl_->forward_token(tokens[(size_t)i], all.data() + (size_t)i * vocab);
            if (!finite_row(all.data() + (size_t)i * vocab, vocab))
                fail("non-finite logits at position " + std::to_string(i));
            ++i;
        }
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
    const int batch = impl_->batch_on ? impl_->batch_cap : 1;
    for (int i = (int)begin; i < (int)prompt.size();) {
        if (stop.load(std::memory_order_relaxed)) {
            gr.finish = "cancel";
            gr.prompt_ms = ms_since(t0);
            return gr;
        }
        if (reforward && i == (int)begin) impl_->n_past = begin;
        const int k = std::min(batch, (int)prompt.size() - i);
        if (k > 1) {
            impl_->forward_batch(prompt.data() + i, k, impl_->last_logits.data(), 1);
            for (int j = 0; j < k; ++j) impl_->held.push_back(prompt[(size_t)(i + j)]);
            i += k;
        } else {
            impl_->forward_token(prompt[(size_t)i], impl_->last_logits.data());
            impl_->held.push_back(prompt[(size_t)i]);
            ++i;
        }
        impl_->have_logits = true;
        const int done = i;
        if (on_pp && (done == (int)prompt.size() || done % chunk == 0 || done > (int)prompt.size() - batch))
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
