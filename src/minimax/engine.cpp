// src/minimax/engine.cpp - MiniMax-M3.1-Flash forward pass.
//
// Token-by-token, prefill and decode share one path (P2b; a batched prefill can replace it later without changing
// the math). Dense GEMVs are fp32. Quantized routed experts are ggml-cpu vec_dot (activation quantized with the
// type's vec_dot_type) plus swigluoai. KV is fp16 on the host only; nothing is mirrored to the device until a GPU
// attention kernel exists. See include/strata/minimax/forward.hpp.
#include "strata/minimax/forward.hpp"

#include "strata/kernels/cpu/iqk_quants.hpp"
#include "strata/kernels/iqk_types.hpp"

#include "cuda_ops.hpp"
#include "strata/minimax/vision.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
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
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>   // STRATA_LANE_PF's _mm_prefetch
#endif

#include "ggml-cpu.h"
#include "ggml.h"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/platform/direct_file.hpp"
#include "strata/minimax/vram_plan.hpp"

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
    // P5c: the per-head norm weights on the device, for the fused norm+rope kernel.
    DevPtr d_qn, d_kn, d_iqn, d_ikn;
};

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

int64_t bigger(int64_t a, int64_t b) { return a > b ? a : b; }

// One pool for the whole engine. parallel_for used to spawn and join on every GEMV row-split.
// P8: the band index is readable inside a job as `tls_part`, so a job that needs a private scratch slice (host
// attention's score row) can index one array by band instead of allocating per row.
thread_local int tls_part = 0;

struct Pool {
    int nthreads = 1;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    const std::function<void(int, int)>* fn = nullptr;   // the caller's job; valid until run() returns
    int rows = 0;
    int parts = 1;
    int chunk = 0;                  // > 0: rows are handed out `chunk` at a time from `next` (run_dyn)
    std::atomic<int> next{0};
    std::atomic<int> epoch{0};
    int pending = 0;
    std::atomic<bool> stop{false};
    int spin_us = 0;      // P7b: how long an idle worker spins before it sleeps on the condition variable

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
            epoch.fetch_add(1);
        }
        cv_work.notify_all();
        for (auto& t : workers)
            if (t.joinable()) t.join();
    }

    static inline void cpu_relax() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#endif
    }

    // Static bands (`run`) or chunks taken from `next` (`run_dyn`).  Each row is computed by exactly one call
    // either way, so what a row computes does not depend on the mode.
    void work(int p, const std::function<void(int, int)>& job) {
        if (chunk > 0) {
            for (;;) {
                const int a = next.fetch_add(chunk, std::memory_order_relaxed);
                if (a >= rows) break;
                job(a, std::min(rows, a + chunk));
            }
            return;
        }
        const int a = (int)((int64_t)rows * p / parts);
        const int b = (int)((int64_t)rows * (p + 1) / parts);
        if (a < b) job(a, b);
    }

    void worker(int p) {
        tls_part = p;
        int seen = 0;
        for (;;) {
            if (spin_us > 0) {
                // P7b: spin before sleeping.  A token's layers hand the pool several small jobs in a row, and a
                // condvar wake per job is a context switch each (Project Maya's pool spins 20 ms through a token's
                // layers for that reason, ds4_model.cu:5560).  The spin reads the epoch without the mutex.
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
                int k = 0;
                while (!stop.load(std::memory_order_relaxed) && epoch.load(std::memory_order_acquire) == seen) {
                    cpu_relax();
                    if ((++k & 255) == 0 && std::chrono::steady_clock::now() >= until) break;
                }
            }
            const std::function<void(int, int)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [&] { return stop.load() || epoch.load() != seen; });
                if (stop) return;
                seen = epoch.load();
                job = fn;
            }
            if (job) work(p, *job);
            std::lock_guard<std::mutex> lk(mu);
            if (--pending == 0) cv_done.notify_one();
        }
    }

    void run(int nrows, const std::function<void(int, int)>& job, bool force = false) { dispatch(nrows, job, force, 0); }

    // P14: the same job with rows handed out `chunk` at a time, so a band whose thread is late (the engine's own
    // CUDA and SSD threads share the cores) is picked up by the others instead of holding the barrier.  Only for
    // jobs whose rows are independent and that keep no per-band state (tls_part scratch is fine, a per-band sum
    // is not).
    void run_dyn(int nrows, int chunk_rows, const std::function<void(int, int)>& job) {
        dispatch(nrows, job, false, std::max(1, chunk_rows));
    }

    void dispatch(int nrows, const std::function<void(int, int)>& job, bool force, int chunk_rows) {
        if (nrows <= 0) return;
        if (nthreads <= 1 || (!force && nrows < nthreads * 4)) {
            tls_part = 0;
            job(0, nrows);
            return;
        }
        tls_part = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            fn = &job;
            rows = nrows;
            parts = nthreads;
            chunk = chunk_rows;
            next.store(0, std::memory_order_relaxed);
            pending = nthreads - 1;
            epoch.fetch_add(1, std::memory_order_release);
        }
        cv_work.notify_all();
        work(0, job);
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
    if (strata::kernels::ikq::is_iqk_type(t.type)) {
        strata::kernels::ikq::dequantize_row(t.type, src, dst, n);
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

// The packed types `native_mmvq` reads directly (llama.cpp's CUDA MMVQ, `native_mmvq.cu`).  Types 11-14, 23 and the
// i-quants are 256-value blocks; Q4_0/Q5_0/Q8_0/IQ4_NL/Q2_0 are 32/64-value ones.
bool mmvq_dense_type(int t) {
    switch (t) {
        case 2: case 6: case 8: case 20:              // Q4_0, Q5_0, Q8_0, IQ4_NL
        case 11: case 12: case 13: case 14: case 23:  // Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS
        case 42:                                      // Q2_0
            return true;
        default:
            return strata::kernels::iq_supported(t);
    }
}
// A dense projection with `cols` values per row can take the packed path only on a whole number of blocks.
bool mmvq_dense_fits(int t, int64_t cols) {
    if (!mmvq_dense_type(t)) return false;
    switch (t) {
        case 2: case 6: case 8: case 20: return cols % 32 == 0;
        case 42: return cols % 64 == 0;
        default: return cols % 256 == 0;
    }
}

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

// Can this process back `bytes` of new host memory?  Windows counts commit against RAM + pagefile; elsewhere shape
// it from the available memory.  Used to size the long-context KV cache before touching any page.
bool host_commit_ok(size_t bytes) {
#if defined(_WIN32)
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (!GlobalMemoryStatusEx(&mem)) return true;
    return (unsigned long long)mem.ullAvailPageFile >= (unsigned long long)bytes;
#else
    const size_t avail = host_available_bytes();
    if (avail == 0) return true;
    return avail >= bytes;
#endif
}

// Physical cores (one per RelationProcessorCore entry), for the automatic CPU thread count.
int physical_cores() {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
        return 0;
    int n = 0;
    for (DWORD at = 0; at < len;) {
        const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + at);
        if (e->Relationship == RelationProcessorCore) ++n;
        if (e->Size == 0) break;
        at += e->Size;
    }
    return n;
#else
    // One core_id per logical CPU; the first core_id of each physical package pair is the thread's sibling.
    // Counting distinct (package, core) pairs is what matters, so read both.
    std::vector<std::pair<int, int>> seen;
    for (int i = 0; i < 4096; ++i) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/";
        std::ifstream fc(base + "core_id"), fp(base + "physical_package_id");
        int core = 0, pkg = 0;
        if (!(fc >> core) || !(fp >> pkg)) break;
        const std::pair<int, int> id{pkg, core};
        bool known = false;
        for (const auto& s : seen) known = known || s == id;
        if (!known) seen.push_back(id);
    }
    return (int)seen.size();
#endif
}

// Total physical memory, for the automatic pinned-tier budget.
size_t host_total_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) return (size_t)m.ullTotalPhys;
    return 0;
#else
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t v = 0;
    std::string unit;
    while (f >> key >> v >> unit) {
        if (key == "MemTotal:") return v * 1024;
    }
    return 0;
#endif
}

// P7: what to keep out of the pinned tier on an automatic budget.  A fixed 2 GiB was both too small on a 128 GB
// box (Windows, the desktop and the tools that talk to the engine live in 6-10 GiB, and a pinned page cannot be
// paged out) and too large on a 32 GB one, where it is 6% of the machine.  1/16 of physical, clamped to 3-10 GiB,
// is the shape Project Maya exposes as STRATA_GLM_RAM_HEADROOM_GB (6 GiB default) and the one Strata-DS's laptop
// notes use; the number below is sized from the machine rather than fixed, so 32 GB, 64 GB and 128 GB all work.
size_t ram_headroom_auto(size_t phys) {
    size_t v = phys / 16;
    if (v < ((size_t)3 << 30)) v = (size_t)3 << 30;
    if (v > ((size_t)10 << 30)) v = (size_t)10 << 30;
    return v;
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

    std::vector<uint8_t> k_host_q_v, v_host_q_v;   // host-mode q8 storage (std::vector)
    std::vector<uint16_t> k_host_v, v_host_v, ik_host_v;   // host-mode storage (std::vector)
    uint16_t* k_host = nullptr;      // the active K/V/indexer-K storage: pinned when CUDA is up
    uint16_t* v_host = nullptr;
    uint16_t* ik_host = nullptr;
    void* kv_arena = nullptr;        // cudaHostAlloc'd bytes behind k_host (null in host mode)
    size_t kv_arena_bytes = 0;
    // ---- P8: quantized host/pinned K/V.  q8_0 blocks of 32 values (2-byte fp16 scale + 32 int8 = 34 B) along
    // head_dim, so one head's 128 values are 4 whole blocks and a row is [n_head_kv] of those.  The indexer keys
    // stay fp16: they only rank blocks, and a flipped rank is a discrete change in the output.
    int kv_q = 0;                    // 0 fp16, 1 q8_0 (host and pinned-RAM tiers only)
    uint8_t* k_host_q = nullptr;     // q8 arena: K then V, [layer][ctx][n_head_kv][head_dim/32] blocks
    uint8_t* v_host_q = nullptr;
    size_t kv_q_row = 0;             // q8 bytes in one token's K (or V) row
    void* kv_q_arena = nullptr;      // cudaHostAlloc'd bytes behind k_host_q (null in host mode)
    size_t kv_q_arena_bytes = 0;
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
        DevPtr xq;                      // q8_1 image of a dense projection's activation (dense MMVQ)
        DevPtr ex_sg, ex_su, ex_sd;     // MM-NVFP4: per-selected-expert divisors, one per device
        DevPtr qd, bs, slots, counts, kstage, vstage, att;
        // P5c: the fused norm+rope step's device operands.  `qn`/`kn`/`vn` are the projection outputs of the
        // token path (normed/rotated in place), `iqn`/`ikn` the indexer's; `pos1` is the one-token position array.
        DevPtr qn, kn, vn, iqn, ikn, pos1;
        DevPtr kn16, vn16, ik16;    // fp16 staging the graph writes; the cache copy is one D2D outside it
        DevPtr rope_cos, rope_sin;   // [rope_pos_cap][n_rot/2] fp32
        int64_t rope_pos_cap = 0;
        uint16_t* kstage_h = nullptr;   // pinned staging for the RAM tier
        uint16_t* vstage_h = nullptr;
        void* hop_h = nullptr;          // pinned staging for the hidden state that crosses devices
        size_t hop_bytes = 0;
        void* side = nullptr;
        void* ds = nullptr;             // dense GEMV stream (native_mmvq needs an explicit stream)
        void* dsb = nullptr;            // blocking dense-GEMV stream for the dev_out (norm+rope) chain
        // P10: the shared expert's own buffers and stream.  It is independent of routing, so it is launched on `shs`
        // at the top of the MoE and waited for after the CPU lane, where the card would otherwise be idle.
        DevPtr sh_x, sh_xq, sh_g, sh_u, sh_h, sh_hq, sh_y;
        void* shs = nullptr;
        void* sh_pin = nullptr;         // pinned staging for the shared MLP's input, gate/up and hidden state
        size_t sh_pin_bytes = 0;
        void* prof_ev[2] = {nullptr, nullptr};   // P8: timing events around the expert side stream
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

    // ---- P8: host attention scratch.  The P2 loop allocated 65 vectors per token per layer; these are sized
    // once.  `ah_prob`/`ah_prob_d` are one slice per pool band (see tls_part).
    bool host_attn_fast = true;
    std::vector<float> ah_bs, ah_bs2, ah_prob, ah_prob_d, ah_prob_s, ah_prob_ds;
    std::vector<float> kv_f32;   // q8 store scratch (fp16 -> fp32 -> blocks)
    std::vector<float> ah_stage; // H2D staging for one dequantized head row
    std::vector<float> att_buf;  // the token path's attention output (P8: was a per-layer allocation)
    std::vector<int32_t> ah_slots;
    std::vector<int> ah_off, ah_order;
    std::vector<char> ah_sel;
    std::vector<int64_t> ah_ids;
    std::vector<int32_t> held;
    // ---- P6: images for the current request.  `img_row[pos]` is the embedding row that replaces the token
    // embedding at that prompt position, or -1.  Cleared by reset().
    std::unique_ptr<Vision> vision;
    std::vector<float> img_data;   // [rows, n_embd]
    std::vector<int> img_row;      // per prompt position
    int64_t img_rows = 0;
    int64_t n_past = 0;
    int64_t cur_pos = 0;
    bool have_logits = false;
    std::mt19937_64 rng{0};
    Pool pool;

    // ---- P5a: batched prompt processing.  `batch_cap` tokens per layer at a time; every buffer below lives on the
    // device that owns the layer being computed (one set per device).  Off unless --minimax-prefill-batch > 1.
    bool batch_on = false;
    int batch_cap = 0;
    // P12: the batched-prompt workspace and the VRAM expert slots share the card.  `prefill_ws_swap` says the
    // prompt pass must free slots to make room (and put them back afterwards); `prefill_ws_bytes` is the workspace's
    // over-estimated size per device; `ws_taken[layer]` counts the slots currently freed on that layer, so the
    // restore re-allocates exactly what came out.
    bool prefill_ws_swap = false;
    size_t prefill_ws_bytes = 0;
    bool ws_live = false;            // the device side of the workspace is allocated right now
    std::vector<int> ws_taken;       // per layer: slots freed for the prompt
    bool prefill_gpu_experts = true;
    bool prefill_dense_f32 = true;
    // P10: the routed experts' batched projection.  "mmvq" keeps the int8 iq_mmvq dots; "tc" dequantizes each
    // expert's weights to fp16 once per chunk and runs the tokens routed to it through the FP16 tensor-core GEMM.
    std::string prefill_expert_gemm = "mmvq";
    int tc_expert_min_rows = 16;   // "auto": a group this small is not worth a weight dequantization
    // P16: per device index, the device index whose card runs that card's layers' routed prompt experts.  A card
    // on a narrow host link (rig-2 / the 2x5070 PC: GPU0 on Gen3 x4, ~3.5 GB/s) spends most of a layer waiting
    // for the expert stream, so its layers hand the routed experts to the widest link; router and shared expert
    // stay on the layer's card.  Resolved by a bandwidth probe on the first prompt.
    std::vector<int> pf_xdev;
    bool prefill_direct = false;   // unbuffered expert reads (--minimax-prefill-read direct)
    bool prefill_kv_stage = false; // P8: stage one layer's K/V on the card for the batched prompt pass
    bool dense_mmvq = true;        // packed dense projections via native_mmvq (--minimax-dense-mmvq off)
    bool sh_pending = false;       // P10: the shared expert's gate/up work is in flight on the shared stream
    int sh_ff = 0;                 // its feed-forward width this layer
    bool gpu_norm_rope = true;     // fused device norm+rope (--minimax-gpu-normrope off)
    bool expert_resident = true;   // pin the RAM expert tier at startup (--minimax-expert-resident off)
    struct PfBuf {
        uint16_t* x16 = nullptr;     // [T][embd] fp16 hidden state
        uint16_t* h16 = nullptr;     // [T][embd] fp16 normed input to the projections
        uint16_t* a16 = nullptr;     // [T][max(qlen,klen)] fp16 attention output / K+V staging
        float* y32 = nullptr;        // [max(T*wide, rows_cap*maxN)] fp32 GEMM output
        float* p32 = nullptr;        // [max(T*wide, rows_cap*maxN)] fp32 projection / routed MoE output
        float* sh = nullptr;         // [max(T*embd, rows_cap*maxN)] fp32 shared/MoE sum and the dense tile's up
        size_t y32_elems = 0, p32_elems = 0, sh_elems = 0;
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
        // P10: with unbuffered reads the three roles are read straight into the pinned slot, each at a 4 KiB
        // boundary with 4 KiB of headroom in front of it (DirectFile's contract), so the read no longer has to be
        // copied out of a shared staging buffer.  The role pointer is `slot + role_off[r] + (file_off & 4095)`.
        size_t role_off[3] = {0, 0, 0};   // offsets of gate / up / down inside one pinned slot
        size_t pin_slot_bytes = 0;        // capacity of one pinned slot (headroom + roles + tail)
        // P5b: the ring is multi-slot.  Every slot has its own page-locked staging buffer, its own H2D event on
        // the copy stream and a "compute finished" event on the main stream, so a copy can be in flight while the
        // previous expert's GEMM runs and a slot is only rewritten after the math that reads it has completed.
        int ring_n = 0;
        std::vector<void*> ring_v, pin_v;
        std::vector<void*> ev_copy, ev_comp;
        void* copy_stream = nullptr;
        DevPtr d_iq16, d_bs, d_blk, d_nblk, d_rows, d_ent_row, d_ent_w;

        DevPtr d_q16;                // fp16 attention queries for the batched MSA kernel
        DevPtr d_pos;                // [T] token positions for the fused norm+rope kernel
        DevPtr kv_k, kv_v;           // P8: one layer's whole K/V prefix staged on the card (RAM tier prefill)
        int64_t rows_cap = 0;
    };
    std::vector<PfBuf> pf;
    int64_t max_dense_n = 0;         // widest projection (rows)
    int64_t rows_cap = 0;            // rows per expert/dense chunk in the batched path
    int pf_max_blk = 0;              // top_k blocks per query (device shared-memory bound)
    // P9: a pinned-tier budget bounds the arena, not the OS page cache.  With `--minimax-prefill-read direct` (and
    // by default when `--memory-limit-mib` is set) every expert that is not resident in VRAM or pinned RAM is read
    // straight from the shard file into a staging buffer, so streaming the 146 GB model cannot grow the working set.
    bool stream_direct = false;
    std::vector<uint8_t> pf_expert_h;   // one streamed expert for the CPU MoE path (direct reads)
    int mem_probe_n = 0;                // experts read since the last memory probe line

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
    std::vector<float> pf_att_h;   // P7: device attention output when the layer's projection is on the CPU
    std::vector<int32_t> pf_pos;      // [T] chunk positions for the fused norm+rope kernel
    std::vector<float> pf_gate_h, pf_up_h, pf_hid_h;
    std::vector<float> pf_moe_h;     // [T][embd] the layer's FFN output on the host
    std::vector<float> pf_dense_h;   // [T][max_dense_n] host intermediate for the CPU dense MLP
    std::vector<float> pf_out_tile;  // [T][lm_head tile] host staging for the output projection

    // ---- P5b: decode.  Exactly one layer ahead, the router of layer l+1 is applied to the residual layer l just
    // produced (that IS layer l+1's input), and its non-resident experts are copied into free VRAM slots on a side
    // stream while layer l's expert math runs.  `prefetch_pending` is the stream's work for the layer above.
    int prefetch_slots = 0;   // --minimax-expert-prefetch (0 = off)
    int prefetch_pending = -1;

    // ---- P5c: CUDA graphs for the token path's per-layer GPU prefix (q8_1 -> dense MMVQ q/k/v + indexer ->
    // norm+rope -> fp16).  One graph per layer, captured on the first token that reaches it.
    bool graphs_on = true;
    std::vector<void*> tok_graph;      // [n_layers], null when not captured (or capture failed)

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

    // P6: the token path's SSD staging.  One 4 KiB-aligned slot per routed entry, each holding the three role
    // buffers; a role is read straight into its final place (the aligned superset of the tensor's bytes starts
    // `off & 4095` before it), so no copy out of a shared read buffer is needed any more, and every read of a layer
    // is queued at once so the drive and the CPU experts work at the same time.
    uint8_t* ssd_slots = nullptr;
    size_t ssd_slot_bytes = 0;
    size_t ssd_slot_cap = 0;       // slots: one route per slot, 2 x top per prefetch region
    size_t ssd_gu_cap = 0;
    size_t ssd_dn_cap = 0;
    std::vector<int> ssd_pending;   // queued, not yet completed, per shard (the layer's own misses, tag 0)
    std::vector<int> ssd_pf_out;    // queued, not yet completed, per prefetched layer (tag = layer + 1)
    std::vector<int> ssd_pf_out_shard;   // the same split per shard, so a layer wait blocks only where it reads
    double ssd_wait_ms = 0.0;       // the last token's total time spent waiting for the SSD

    std::vector<float> usage;       // decayed routing counts, [n_layers * n_expert]
    std::vector<double> route_counts;   // undecayed counts for --minimax-profile-save
    uint64_t rebalance_tick = 0;
    uint64_t lru_tick = 0;
    uint64_t swaps = 0;

    struct TokenStats {
        int64_t vram = 0, ram = 0, ssd = 0, ssd_bytes = 0, skipped = 0, ssd_mapped = 0;
    };
    TokenStats tok_stats;
    TokenStats total_stats;
    int64_t tokens_seen = 0;
    int64_t profile_saves = 0;

    // ---- P8: --minimax-profile-time.  One decode token's wall time, split by the phase that spent it, on stderr
    // after every token; `stats()` prints the mean of the run.  Every timer is a steady_clock pair around work that
    // already exists, and with the flag off only a `bool` is tested per phase, so the default path is unchanged.
    // `cpu_*` are the CPU lane's own passes (so the 20 GB/s question has a numerator and a denominator),
    // `gpu_stream` is the DEVICE-busy time of the expert side stream between two timing events and `gpu_finish` the
    // host time blocked on it: the pair separates "the card was slow" from "the card was idle".
    struct Prof {
        double embed = 0, attn = 0, attn_idx_sync = 0, attn_sync = 0, router = 0, pick = 0, gpu_begin = 0,
               cpu_ram = 0, ssd_wait = 0, cpu_ssd = 0, gpu_finish = 0, combine = 0, shared = 0, admit = 0,
               prefetch = 0, lm_head = 0, total = 0;
        double cpu_bytes = 0, ssd_bytes = 0, gpu_stream = 0;
        int64_t cpu_experts = 0, gpu_experts = 0, ram_experts = 0, ssd_experts = 0;
    };
    Prof prof, prof_sum;
    int64_t prof_tokens = 0;
    static double pms(std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    }
    static void prof_acc(Prof& a, const Prof& b) {
        a.embed += b.embed; a.attn += b.attn; a.attn_idx_sync += b.attn_idx_sync; a.attn_sync += b.attn_sync;
        a.router += b.router; a.pick += b.pick; a.gpu_begin += b.gpu_begin; a.cpu_ram += b.cpu_ram;
        a.ssd_wait += b.ssd_wait; a.cpu_ssd += b.cpu_ssd; a.gpu_finish += b.gpu_finish; a.combine += b.combine;
        a.shared += b.shared; a.admit += b.admit; a.prefetch += b.prefetch; a.lm_head += b.lm_head;
        a.total += b.total; a.cpu_bytes += b.cpu_bytes; a.ssd_bytes += b.ssd_bytes; a.gpu_stream += b.gpu_stream;
        a.cpu_experts += b.cpu_experts; a.gpu_experts += b.gpu_experts; a.ram_experts += b.ram_experts;
        a.ssd_experts += b.ssd_experts;
    }
    // The phases that are timed, added up.  `other` is whatever the clock did not see (a float comparison, a loop
    // the phases do not name): if it is large, a timer is missing, not work.
    static std::string prof_line(const Prof& p, int64_t idx) {
        const double named = p.embed + p.attn + p.router + p.pick + p.gpu_begin + p.cpu_ram + p.ssd_wait + p.cpu_ssd +
                             p.gpu_finish + p.combine + p.shared + p.admit + p.prefetch + p.lm_head;
        // bytes/ms -> GB/s, both from this token's own counters.
        const double gbs = p.cpu_ram + p.cpu_ssd > 0 ? p.cpu_bytes / (p.cpu_ram + p.cpu_ssd) / 1.0e6 : 0.0;
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "INFO minimax-profile token=%lld total=%.1f embed=%.2f attn=%.1f idx_sync=%.1f msa_sync=%.1f "
                      "router=%.1f pick=%.1f gpu_begin=%.2f cpu_ram=%.1f ssd_wait=%.1f cpu_ssd=%.1f gpu_finish=%.1f "
                      "gpu_stream=%.1f combine=%.1f shared=%.1f admit=%.1f prefetch=%.1f lm_head=%.1f other=%.1f "
                      "cpu_exp=%lld cpu_gbs=%.1f ssd_exp=%lld",
                      (long long)idx, p.total, p.embed, p.attn, p.attn_idx_sync, p.attn_sync, p.router, p.pick,
                      p.gpu_begin, p.cpu_ram, p.ssd_wait, p.cpu_ssd, p.gpu_finish, p.gpu_stream, p.combine,
                      p.shared, p.admit, p.prefetch, p.lm_head, p.total - named, (long long)p.cpu_experts, gbs,
                      (long long)p.ssd_experts);
        return std::string(buf);
    }
    static Prof prof_mean(const Prof& s, int64_t n) {
        Prof m;
        if (n <= 0) return m;
        const double d = (double)n;
        m.embed = s.embed / d; m.attn = s.attn / d; m.attn_idx_sync = s.attn_idx_sync / d;
        m.attn_sync = s.attn_sync / d; m.router = s.router / d; m.pick = s.pick / d; m.gpu_begin = s.gpu_begin / d;
        m.cpu_ram = s.cpu_ram / d; m.ssd_wait = s.ssd_wait / d; m.cpu_ssd = s.cpu_ssd / d;
        m.gpu_finish = s.gpu_finish / d; m.combine = s.combine / d; m.shared = s.shared / d; m.admit = s.admit / d;
        m.prefetch = s.prefetch / d; m.lm_head = s.lm_head / d; m.total = s.total / d;
        m.gpu_stream = s.gpu_stream / d;
        m.cpu_bytes = s.cpu_bytes / d; m.ssd_bytes = s.ssd_bytes / d;
        m.cpu_experts = (int64_t)std::llround((double)s.cpu_experts / d);
        m.gpu_experts = (int64_t)std::llround((double)s.gpu_experts / d);
        m.ram_experts = (int64_t)std::llround((double)s.ram_experts / d);
        m.ssd_experts = (int64_t)std::llround((double)s.ssd_experts / d);
        return m;
    }

    // P5c: --minimax-expert-skip.  Drop a routed expert whose RENORMALIZED weight is below `expert_skip` and
    // renormalize what is left, so the routed sum keeps its scale.  Returns the number kept; a route whose every
    // weight is below the threshold keeps everything rather than produce a zero vector.
    float expert_skip = 0.f;
    int apply_expert_skip_range(float* w, int n) {
        if (expert_skip <= 0.f || n <= 0) return n;
        int kept = 0;
        float sum = 0.f;
        for (int i = 0; i < n; ++i)
            if (w[i] >= expert_skip) {
                ++kept;
                sum += w[i];
            }
        if (kept == 0 || kept == n) return n;
        for (int i = 0; i < n; ++i) w[i] = (w[i] >= expert_skip && sum > 0.f) ? w[i] / sum : 0.f;
        tok_stats.skipped += n - kept;
        return kept;
    }
    int apply_expert_skip(std::vector<float>& w) { return apply_expert_skip_range(w.data(), (int)w.size()); }

    bool ram_inplace = false;      // P7b: --minimax-ram-mode inplace: the page cache is the tier, nothing is copied
    // P6: --minimax-ssd-chunks and --minimax-expert-skip-ssd.  P7: --minimax-ssd-zerocopy.
    int ssd_chunks = 1;
    int ssd_skip_rank = 0;
    bool ssd_zero_copy = true;
    // P10: --minimax-page-probe N: pages probed per expert for a page-cache hit.  0 = walk every page.  32 spread
    // pages is the default: a cold expert fails the first few, a resident one passes all 32, and the full 4-8 K page
    // walk (1.5 ms an expert) was most of the token's `pick` phase.
    int page_probe = 32;

    // ---- P7b: the route nudge.  Xolotl (Indras-Mirror/Xolotl, MIT) measures 9.7 -> 19.1 -> 21.7 tok/s on one
    // RTX 3090 + 128 GB answering a 148 GB model by biasing the *selection* toward experts that are already
    // resident, and its quality table says why the guards are not optional: the default flat bias of 1.0 costs 14
    // points of top-1, while 0.25 keeps 85.2% against 86.2% exact.  Its three guards, here:
    //   PROTECT P  the exact top-P experts are never displaced (`+1e4` in the biased score);
    //   POOL M     a pick must stay inside the exact top-M (arXiv:2412.00099's max-rank window);
    //   MARGIN d   the nudge only applies to experts whose exact score is within d of the k-th exact score;
    //              anything confidently above the cutoff is kept, anything confidently below is not promoted.
    // The nudge changes which experts are chosen and nothing else: the weights are still the unbiased sigmoid
    // probabilities of the picks, renormalised as the model does, so the layer's scale is unchanged.  With the
    // flag off (the default) the selection is the model's own choice, bit for bit.
    bool route_bias = false;
    float rb_in = 0.25f;         // VRAM or pinned-RAM resident
    float rb_out = 0.15f;        // in neither tier: would be read from the file
    int rb_protect = 2;
    int rb_pool = 12;
    float rb_margin = 0.f;
    int64_t rb_swapped = 0;      // route meter: positions where a pick differs from the exact pick
    int64_t rb_picks = 0;
    double rb_wnum = 0, rb_wden = 0;   // router weight of the picks against the weight of the exact picks

    // Rank the experts of one layer.  `prob` is the unbiased sigmoid router output; `order` comes back with the
    // `top` selected ids, best first, under the model's own bias plus (when enabled) the residency nudge.
    void select_routes(const Layer& L, const std::vector<float>& prob, std::vector<int>& order, int top,
                       bool meter = true) {
        const int nexp = (int)prob.size();
        std::vector<float> sc((size_t)nexp);
        for (int i = 0; i < nexp; ++i) sc[(size_t)i] = prob[(size_t)i] + L.bias[(size_t)i];
        order.resize((size_t)nexp);
        std::iota(order.begin(), order.end(), 0);
        auto better = [&](int a, int b) {
            if (sc[(size_t)a] != sc[(size_t)b]) return sc[(size_t)a] > sc[(size_t)b];
            return a < b;
        };
        if (!route_bias || !tiers_on) {
            std::partial_sort(order.begin(), order.begin() + top, order.end(), better);
            return;
        }
        // The exact ranking first: it defines the cutoff, the protected prefix and the pool.
        const int pool = std::min(nexp, std::max(top, rb_pool));
        std::vector<int> ex(order);
        std::partial_sort(ex.begin(), ex.begin() + pool, ex.end(), better);
        const float cutoff = sc[(size_t)ex[(size_t)(top - 1)]];
        std::vector<float> sel(sc);
        std::vector<char> eligible((size_t)nexp, 0);
        for (int r = 0; r < pool; ++r) {
            const int e = ex[(size_t)r];
            eligible[(size_t)e] = 1;
            if (r < rb_protect) {
                sel[(size_t)e] += 1e4f;
                continue;
            }
            if (rb_margin > 0.f && std::fabs(sc[(size_t)e] - cutoff) > rb_margin) continue;
            const bool resident = L.vram_of[(size_t)e] >= 0 || (L.ram_n > 0 && L.ram_of[(size_t)e] >= 0);
            sel[(size_t)e] += resident ? rb_in : -rb_out;
        }
        auto better_sel = [&](int a, int b) {
            if (sel[(size_t)a] != sel[(size_t)b]) return sel[(size_t)a] > sel[(size_t)b];
            return a < b;
        };
        std::vector<int> cand;
        cand.reserve((size_t)pool);
        for (int r = 0; r < pool; ++r) cand.push_back(ex[(size_t)r]);
        std::partial_sort(cand.begin(), cand.begin() + top, cand.end(), better_sel);
        for (int j = 0; j < top; ++j) order[(size_t)j] = cand[(size_t)j];
        // The meter: how far the nudge moved the choice, and how much of the exact picks' router weight it kept.
        if (!meter) return;
        int swaps = 0;
        for (int j = 0; j < top; ++j) swaps += order[(size_t)j] != ex[(size_t)j];
        rb_swapped += swaps;
        for (int j = 0; j < top; ++j) {
            rb_wnum += prob[(size_t)order[(size_t)j]];
            rb_wden += prob[(size_t)ex[(size_t)j]];
        }
        rb_picks += top;
    }

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
        dense_mmvq = opt.dense_mmvq != "off";
        gpu_norm_rope = opt.gpu_normrope != "off";
        expert_resident = opt.expert_resident != "off";
        expert_skip = opt.expert_skip > 0.f ? opt.expert_skip : 0.f;
        graphs_on = opt.graphs != "off";
        if (const char* pf = std::getenv("STRATA_LANE_PF")) {
            lane_pf = std::atoi(pf);
            if (lane_pf < 0) lane_pf = 0;
            if (lane_pf > 0) std::fprintf(stderr, "INFO minimax-lane-prefetch rows=%d\n", lane_pf);
        }
        if (expert_skip > 1.f) fail("--minimax-expert-skip is a renormalized weight in (0, 1]");
        if (expert_skip > 0.f)
            std::fprintf(stderr, "INFO minimax-expert-skip threshold=%.3f\n", expert_skip);
        ssd_chunks = opt.ssd_chunks > 0 ? opt.ssd_chunks : 1;
        if (ssd_chunks > 16) ssd_chunks = 16;
        ssd_skip_rank = opt.expert_skip_ssd;
        ssd_zero_copy = opt.ssd_zerocopy != "off";
        page_probe = opt.page_probe;
        ssd_prefetch = opt.ssd_prefetch != "off" && !opt.ssd_prefetch.empty();
        ssd_prefetch_depth = std::clamp(opt.ssd_prefetch_depth, 1, 8);
        if (ssd_prefetch)
            std::fprintf(stderr, "INFO minimax-ssd-prefetch depth=%d slots=%zu\n", ssd_prefetch_depth,
                         (size_t)cfg.n_expert_used * 2 * (size_t)(ssd_prefetch_depth + 1));
        route_bias = opt.route_bias == "on";
        ram_inplace = opt.ram_mode == "inplace";
        if (ram_inplace && opt.memory_limit_mib > 0)
            std::fprintf(stderr, "INFO minimax-ram-mode=inplace (--memory-limit-mib still sets how much of the "
                                 "model the ranking calls resident)\n");
        // P9: `direct` is explicit; with a pinned-tier byte budget it is implied, because the budget cannot bound
        // the page cache the mmap path grows.  `mmap` stays available (and is the default without a budget): it is
        // faster when the whole expert set is meant to be cached.
        stream_direct = opt.prefill_read == "direct" ||
                        (opt.prefill_read.empty() && opt.memory_limit_mib > 0);
        if (stream_direct && opt.prefill_read != "direct")
            std::fprintf(stderr, "INFO minimax-prefill-read=direct (implied by --memory-limit-mib; "
                                 "--minimax-prefill-read mmap keeps the page cache)\n");
        rb_in = opt.route_bias_in;
        rb_out = opt.route_bias_out;
        rb_protect = std::max(0, opt.route_protect);
        rb_pool = std::max(1, opt.route_pool);
        rb_margin = std::max(0.f, opt.route_margin);
        if (route_bias)
            std::fprintf(stderr,
                         "INFO minimax-route-bias in=%.2f out=%.2f protect=%d pool=%d margin=%.2f (selection only; "
                         "the weights stay unbiased)\n",
                         rb_in, rb_out, rb_protect, rb_pool, rb_margin);
        if (ssd_skip_rank > 0)
            std::fprintf(stderr, "INFO minimax-expert-skip-ssd rank=%d (a route at or past it is left out when it "
                                 "would be read from the SSD)\n", ssd_skip_rank);
        ctx = cfg.context;
        if (opt.max_context > 0 && opt.max_context < ctx) ctx = opt.max_context;
        if (ctx < 1) fail("context length is 0");
        tok_graph.assign((size_t)cfg.n_layers, nullptr);

        ggml_cpu_init();
        load_tensors();
        decide_norm();
        build_layers();
        read_eos();
        init_probes();

        threads = opt.cpu_threads;
        if (threads <= 0) {
            // P7: the default is the physical core count, not the logical one.  Decode here is bandwidth-bound:
            // the CPU expert dots stream 2.4-4 GB per token out of the RAM tier, and on a 12-core/24-thread part
            // the second SMT thread per core adds contention rather than throughput (measured on the 15-layer
            // slice, see tools/minimax/README.md).  A machine that reports no SMT is unaffected.
            const int phys = physical_cores();
            const int logical = (int)std::thread::hardware_concurrency();
            threads = phys > 0 ? phys : (logical > 1 && logical % 2 == 0 ? logical / 2 : logical);
        }
        if (threads <= 0) threads = 4;
        if (threads > 64) threads = 64;
        pool.spin_us = std::max(0, opt.pool_spin_us);
        std::fprintf(stderr, "INFO minimax-threads=%d (source=%s, logical=%d)\n", threads,
                     opt.cpu_threads > 0 ? "--cpu-threads" : "physical cores",
                     (int)std::thread::hardware_concurrency());
        pool.start(threads);

        // N=0 is the all-CPU dense path: no device, no CUDA runtime calls. Automatic placement (-1), any
        // positive layer count, and the P3 expert tiers still need a device to size VRAM.
        const bool cpu_only = opt.dense_layers_gpu == 0 && !opt.tiers && opt.layer_split.empty() &&
                              (opt.kv_space.empty() || opt.kv_space == "host");
        if (!cpu_only) {
            cu::init();
            setup_devices();
            choose_split_and_kv();
        } else {
            // P8: the host tier does not go through choose_split_and_kv, but the q8 request and the attention
            // implementation still apply to it.
            kv_row_bytes = (int64_t)cfg.n_head_kv * cfg.head_dim * 2;
            ik_row_bytes = (int64_t)cfg.idx_dim * 2;
            max_slot = (int32_t)(cfg.idx_top_k * cfg.idx_block);
            if (max_slot < (int32_t)cfg.idx_block) max_slot = (int32_t)cfg.idx_block;
            host_attn_fast = opt.host_attn != "legacy";
            if (opt.kv_quant == "q8_0" || opt.kv_quant == "q8") kv_q = 1;
            if (kv_q && (cfg.head_dim % kQ8Block) != 0)
                fail("--minimax-kv-quant q8_0 quantizes 32 values per block along head_dim; this file's head_dim is " +
                     std::to_string(cfg.head_dim) + ", which is not a multiple of 32. Use f16.");
            print_kv_plan(0, opt.kv_space.empty() ? "auto" : "cpu-only");
        }
        alloc_cache();
        // The row/col limits are also the CPU gemv guard. Device buffers exist only when a layer can be uploaded.
        size_workspace();
        if (!cpu_only) alloc_workspace();
        place_weights();
        if (!cpu_only && kv_mode != 0) alloc_device_kv();
        if (opt.tiers || opt.expert_resident == "all") {
            init_tiers();
        } else if (!opt.profile_save_path.empty() || !opt.routing_counts_path.empty()) {
            for (const Layer& L : layers)
                if (L.moe) ++moe_layers;
            route_counts.assign((size_t)cfg.n_layers * (size_t)cfg.n_expert, 0.0);
        }
        init_prefill();
        // P9: drop the mapping pages the weight upload just faulted in.  With a pinned tier the budget is a hard
        // promise; leaving 10 GiB of dense weights in the standby list next to it is what a desktop feels.
        mem_probe("prefill-init");
        trim_mapped_pages();
        mem_probe("ready");

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

        if (!opt.mmproj.empty()) {
            const auto tv0 = std::chrono::steady_clock::now();
            vision = std::make_unique<Vision>(opt.mmproj, opt.cpu_threads);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
            std::fprintf(stderr,
                         "INFO minimax-vision mmproj=%s layers=%lld hidden=%lld merge=%lld loaded_in=%.1fs\n",
                         opt.mmproj.c_str(), (long long)vision->config().layers,
                         (long long)vision->config().hidden, (long long)vision->config().merge, sec);
        }
    }

    ~Impl() {
        write_own_profile();
        flush_dumps();
        for (void* g : tok_graph)
            if (g) cu::graph_destroy(g);
        tok_graph.clear();
        for (Work& w : work) {
            if (w.side) cu::stream_destroy(w.side);
            w.side = nullptr;
            if (w.ds) cu::stream_destroy(w.ds);
            w.ds = nullptr;
            if (w.dsb) cu::stream_destroy(w.dsb);
            w.dsb = nullptr;
            if (w.shs) cu::stream_destroy(w.shs);
            w.shs = nullptr;
            if (w.kstage_h) cu::host_free(w.kstage_h);
            if (w.vstage_h) cu::host_free(w.vstage_h);
            w.kstage_h = w.vstage_h = nullptr;
        }
        if (kv_arena) cu::host_free(kv_arena);
        pf_free();   // P12: the prompt workspace is allocated only while a prompt runs; free it if the run ended mid-prompt
        if (ssd_stage) strata::platform::DirectFile::free_aligned(ssd_stage);
        if (ssd_tmp) strata::platform::DirectFile::free_aligned(ssd_tmp);
        if (ssd_slots) strata::platform::DirectFile::free_aligned(ssd_slots);
    }

    bool try_must(const std::string& name, Tensor& t) {
        size_t shard = 0;
        const TensorInfo* info = model.find(name, &shard);
        if (!info) return false;
        t.name = name;
        t.type = (int)info->type;
        // The shape is untrusted: refuse a width/depth the engine's int64 offsets cannot hold before any
        // multiplication, and refuse anything whose payload does not lie inside its shard (`in_bounds` does the
        // offset+bytes check against the file size, without letting it wrap).
        const uint64_t kMaxI64 = (uint64_t) (std::numeric_limits<int64_t>::max)();
        if (info->shape.empty() || info->shape[0] == 0 || info->shape[0] > kMaxI64) fail(name + ": bad shape");
        for (size_t d = 1; d < info->shape.size(); ++d)
            if (info->shape[d] == 0 || info->shape[d] > kMaxI64) fail(name + ": bad shape");
        t.ne0 = (int64_t)info->shape[0];
        t.ne1 = info->shape.size() > 1 ? (int64_t)info->shape[1] : 1;
        t.ne2 = info->shape.size() > 2 ? (int64_t)info->shape[2] : 1;
        if (info->shape.size() > 3) fail(name + " has more than 3 dimensions");
        if (!model.in_bounds(*info, shard))
            fail(name + ": has an unknown type or lies past the end of its shard");
        t.host = model.shard(shard).tensor_data(*info);
        t.shard = shard;
        t.file_offset = model.shard(shard).data_start() + info->offset;
        t.bytes = tensor_payload_bytes(*info);
        if (strata::kernels::ikq::is_iqk_type(t.type)) {
            // An ik_llama.cpp format ported into the engine: ggml has no ids for these, so the geometry and
            // the codec are the engine's own (src/kernels/cpu/iqk_quants.cpp).
            if (!strata::kernels::ikq::has_codec(t.type))
                fail(name + ": " + strata::kernels::ikq::type_name(t.type) + " is not implemented in this build");
            t.row_bytes = strata::kernels::ikq::row_bytes(t.type, t.ne0);
        } else {
            t.row_bytes = ggml_row_size((ggml_type)t.type, t.ne0);
        }
        // row_bytes * ne1 * ne2, checked: with the checks above each factor fits an int64, but a crafted file can
        // still make the (non-wrapping) product disagree with the payload, which is the rejection we want.
        const uint64_t rows = (uint64_t)t.ne1 * (uint64_t)t.ne2;
        if (t.row_bytes == 0 || t.bytes == 0 || (uint64_t)t.ne1 > (std::numeric_limits<uint64_t>::max)() / (uint64_t)t.ne2 ||
            (uint64_t)t.row_bytes > (std::numeric_limits<uint64_t>::max)() / rows || t.bytes != (uint64_t)t.row_bytes * rows)
            fail(name + ": row size does not match the GGUF payload (type " + std::to_string(t.type) + ")");
        if (strata::kernels::ikq::is_iqk_type(t.type)) return true;
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
        if (kv_q) {
            // The commit check below is about the bytes actually asked for, so measure the q8 footprint.
            const int qrow = cfg.n_head_kv * cfg.head_dim / kQ8Block * kQ8Bytes;
            bytes = (size_t)qrow * 2 * (size_t)ctx * (size_t)cfg.n_layers + ip * sizeof(uint16_t);
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
        if (kv_q) {
            // P8: q8_0 K/V, 34 bytes per 32 values.  The indexer keys stay fp16 (they only rank blocks).
            kv_q_row = (size_t)(cfg.n_head_kv * cfg.head_dim / kQ8Block) * (size_t)kQ8Bytes;
            size_t qbytes = 0;
            if (!mul_size(kv_q_row * 2, (size_t)ctx * (size_t)cfg.n_layers, qbytes)) qbytes = 0;
            size_t ibytes = 0;
            if (!mul_size(ip, sizeof(uint16_t), ibytes)) ibytes = 0;
            const size_t total = qbytes + ibytes;
            if (kv_mode == 2 && !work.empty()) {
                kv_q_arena = cu::host_alloc(qbytes);
                k_host_q = (uint8_t*)kv_q_arena;
                v_host_q = k_host_q + qbytes / 2;
                std::memset(kv_q_arena, 0, qbytes);
                kv_q_arena_bytes = qbytes;
                ik_host = nullptr;
                return;
            }
            try {
                k_host_q_v.assign(qbytes / 2, 0);
                v_host_q_v.assign(qbytes / 2, 0);
                ik_host_v.assign(ip, 0);
            } catch (const std::exception& e) {
                fail("failed to allocate the host q8_0 KV cache for context " + std::to_string(ctx) + ": " +
                     std::to_string(total) + " bytes are required. Lower --max-context. " + e.what());
            }
            k_host_q = k_host_q_v.data();
            v_host_q = v_host_q_v.data();
            ik_host = ik_host_v.data();
            return;
        }
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
            // q8_1 image of a dense projection's activation (P5b dense MMVQ): 36 bytes per 32 values.
            w.xq = DevPtr(cu::alloc(((size_t)max_cols / 32) * 36 + 64));
            if (kv_mode != 0) {
                w.qd = DevPtr(cu::alloc((size_t)cfg.n_head * cfg.head_dim * sizeof(float)));
                w.att = DevPtr(cu::alloc((size_t)cfg.n_head * cfg.head_dim * sizeof(float)));
                // P5c: fused norm+rope operands.  `qn` is the normed/rotated query the device attention reads
                // directly; `kn`/`vn` are the projection outputs before they are written to the fp16 cache.
                w.qn = DevPtr(cu::alloc((size_t)cfg.n_head * cfg.head_dim * sizeof(float)));
                w.kn = DevPtr(cu::alloc((size_t)cfg.n_head_kv * cfg.head_dim * sizeof(float)));
                w.vn = DevPtr(cu::alloc((size_t)cfg.n_head_kv * cfg.head_dim * sizeof(float)));
                w.iqn = DevPtr(cu::alloc((size_t)cfg.idx_heads * cfg.idx_dim * sizeof(float)));
                w.ikn = DevPtr(cu::alloc((size_t)cfg.idx_dim * sizeof(float)));
                w.pos1 = DevPtr(cu::alloc(sizeof(int32_t)));
                w.kn16 = DevPtr(cu::alloc((size_t)cfg.n_head_kv * cfg.head_dim * 2 + 64));
                w.vn16 = DevPtr(cu::alloc((size_t)cfg.n_head_kv * cfg.head_dim * 2 + 64));
                w.ik16 = DevPtr(cu::alloc((size_t)cfg.idx_dim * 2 + 64));
                // The cos/sin table the kernel indexes.  It is built once per device up to a cap: at 256 KiB of
                // table per 1 K positions an uncapped 1 M context would cost 256 MiB on every card, so the table
                // stops at 128 K and a longer context keeps the host rope (correct, just not this fast path).
                int64_t rp = ctx < 131072 ? ctx : 131072;
                if (rp < 1) rp = 1;
                const int half = (int)cfg.n_rot / 2;
                w.rope_pos_cap = rp;
                std::vector<float> ct((size_t)rp * (size_t)half), st((size_t)rp * (size_t)half);
                // The SAME double-precision expression the host `rope()` evaluates per call, tabulated.  Using
                // build_rope_table's `p * theta^-x` form instead differs by ~1e-13 rad, which is enough to flip an
                // fp16 K/V rounding and (measured) a router top-4 tie: the device path then stops being bit-equal
                // to the host path it replaces.
                for (int64_t pp = 0; pp < rp; ++pp)
                    for (int i = 0; i < half; ++i) {
                        const double freq =
                            (double)pp / std::pow(cfg.rope_theta, (2.0 * (double)i) / (double)cfg.n_rot);
                        ct[(size_t)pp * (size_t)half + (size_t)i] = (float)std::cos(freq);
                        st[(size_t)pp * (size_t)half + (size_t)i] = (float)std::sin(freq);
                    }
                w.rope_cos = DevPtr(cu::alloc(ct.size() * sizeof(float)));
                w.rope_sin = DevPtr(cu::alloc(st.size() * sizeof(float)));
                cu::copy_h2d(w.rope_cos.p, ct.data(), ct.size() * sizeof(float));
                cu::copy_h2d(w.rope_sin.p, st.data(), st.size() * sizeof(float));
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
            // P10: the shared expert runs on its own stream beside the routed-expert CPU lane.  The buffers are tiny
            // (the widest shared MLP is n_ff_exp * n_shared_experts), so they are allocated for every device.
            const int64_t shff = cfg.n_ff_exp * (cfg.n_shared_experts > 0 ? cfg.n_shared_experts : 0);
            if (shff > 0 && cfg.n_embd > 0) {
                const size_t embd4 = (size_t)cfg.n_embd * sizeof(float);
                w.sh_x = DevPtr(cu::alloc(embd4));
                w.sh_xq = DevPtr(cu::alloc(q8_1_bytes(cfg.n_embd) + 64));
                w.sh_g = DevPtr(cu::alloc((size_t)shff * sizeof(float)));
                w.sh_u = DevPtr(cu::alloc((size_t)shff * sizeof(float)));
                w.sh_h = DevPtr(cu::alloc((size_t)shff * sizeof(float)));
                w.sh_hq = DevPtr(cu::alloc(q8_1_bytes(shff) + 64));
                w.sh_y = DevPtr(cu::alloc(embd4));
                w.sh_pin_bytes = 2 * (size_t)(shff > cfg.n_embd ? shff : cfg.n_embd) * sizeof(float);
                w.sh_pin = cu::host_alloc(w.sh_pin_bytes);
                w.shs = cu::stream_create();
            }
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
        return kv_layer_bytes(ctx);
    }
    size_t kv_layer_bytes(int64_t t) const {
        const size_t kv = kv_q ? (size_t)(cfg.n_head_kv * cfg.head_dim / kQ8Block * kQ8Bytes) : (size_t)kv_row_bytes;
        return (size_t)2 * kv * (size_t)t + (size_t)ik_row_bytes * (size_t)t;
    }

    void choose_split_and_kv() {
        kv_row_bytes = (int64_t)cfg.n_head_kv * cfg.head_dim * 2;
        ik_row_bytes = (int64_t)cfg.idx_dim * 2;
        max_slot = (int32_t)(cfg.idx_top_k * cfg.idx_block);
        if (max_slot < (int32_t)cfg.idx_block) max_slot = (int32_t)cfg.idx_block;
        host_attn_fast = opt.host_attn != "legacy";

        // ---- P8: the KV quantization.  Where the q8 tier is not implemented (the VRAM tier's device kernels read
        // fp16 rows) the request is refused by name rather than silently ignored.
        if (!opt.kv_quant.empty() && opt.kv_quant != "auto") {
            if (opt.kv_quant == "q8_0" || opt.kv_quant == "q8") kv_q = 1;
            else if (opt.kv_quant != "f16") fail("--minimax-kv-quant: unknown value '" + opt.kv_quant + "'");
        }
        if (kv_q == 0 && opt.kv_quant == "auto") kv_q = 0;   // decided below, once the placement is known
        if (kv_q && (cfg.head_dim % kQ8Block) != 0)
            fail("--minimax-kv-quant q8_0 quantizes 32 values per block along head_dim; this file's head_dim is " +
                 std::to_string(cfg.head_dim) + ", which is not a multiple of 32. Use f16.");
        if (kv_q && !opt.kv_space.empty() && opt.kv_space != "auto" && opt.kv_space != "ram" &&
            opt.kv_space != "host")
            fail("--minimax-kv-quant q8_0 covers the host and pinned-RAM KV tiers. The VRAM tier's MSA kernels "
                 "read fp16 rows; use --minimax-kv ram (or host) for a quantized cache, or f16 for vram.");

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

        const bool auto_mode = opt.kv_space.empty() || opt.kv_space == "auto";
        // Mode 2 keeps the K/V in pinned host RAM, so it must not be charged against the VRAM budget (P8: it used
        // to be, which made `--minimax-kv ram` refuse any context whose KV exceeded the cards' free memory -- the
        // exact case the tier exists for).
        const bool kv_on_gpu = mode == 1;
        assign(kv_on_gpu);
        std::string bad = check_budget(kv_on_gpu);
        if (!bad.empty() && mode == 1 && auto_mode) {
            std::fprintf(stderr, "INFO minimax-kv: VRAM does not fit (%s); keeping K/V in pinned RAM\n", bad.c_str());
            mode = 2;
            assign(false);
            bad = check_budget(false);
        }
        if (bad.empty() && mode == 2 && opt.kv_quant == "auto" && !kv_q) {
            // The quant is only chosen by capacity: fp16 unless the process cannot commit it.
            const size_t need = kv_layer_bytes() * (size_t)cfg.n_layers;
            if (!host_commit_ok(need)) {
                kv_q = 1;
                std::fprintf(stderr,
                             "INFO minimax-kv: q8_0 K/V (f16 needs %.1f GiB and the box cannot commit it)\n",
                             (double)need / 1073741824.0);
            }
        }
        if (!bad.empty())
            fail(bad + ". Use --layer-split to choose the boundary, --vram-reserve-mib to change the reserve, or "
                 "--minimax-kv ram to move the KV cache to RAM.");
        if (mode == 2) {
            size_t need = kv_layer_bytes() * (size_t)cfg.n_layers;
            if (!host_commit_ok(need))
                fail("the K/V cache for context " + std::to_string(ctx) + " needs " +
                     std::to_string((unsigned long long)need) + " bytes of " +
                     (kv_q ? "q8_0" : "fp16") + " host memory and this box cannot commit it. Lower "
                     "--max-context, or pass --minimax-kv-quant q8_0" +
                     std::string(kv_q ? "" : " (it halves the cache)") + ".");
        }
        kv_mode = mode;
        for (int64_t i = 0; i < cfg.n_layers; ++i) layers[(size_t)i].dev = layer_dev[(size_t)i];

        // The plan, in one line: this is what a long-context run has to be explainable by after the fact.
        print_kv_plan(mode, opt.kv_space.empty() ? "auto" : opt.kv_space);
    }

    // One parseable line naming the KV decision.  `key=value` tokens, the same keys on every path.
    void print_kv_plan(int mode, const std::string& source) {
        const size_t bytes = kv_layer_bytes() * (size_t)cfg.n_layers;
        std::string plan = "INFO minimax-kv plan: ctx=" + std::to_string(ctx) +
                           " quant=" + (kv_q ? "q8_0" : "f16") + " kv=" +
                           std::to_string((double)bytes / 1073741824.0) + "GiB placement=" +
                           (mode == 0 ? "host" : mode == 1 ? "vram" : "ram") + " source=" + source +
                           " indexer=f16 rows=" + std::to_string((unsigned long long)kv_layer_bytes()) +
                           "B/layer host_attn=" + (host_attn_fast ? "fast" : "legacy");
        if (mode == 1 && !devs.empty()) {
            plan += " per_card=[";
            for (int d = 0; d < (int)devs.size(); ++d) {
                int n = 0;
                for (int64_t i = 0; i < cfg.n_layers; ++i)
                    if (layer_dev[(size_t)i] == d) ++n;
                plan += (d ? "," : "") + std::to_string((unsigned long long)((size_t)n * kv_layer_bytes()) >> 20) +
                        "MiB";
            }
            plan += "]";
        }
        std::fprintf(stderr, "%s\n", plan.c_str());
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
            // P5c: the head-norm weights the fused device norm+rope kernel multiplies by.  Only when the KV (and
            // therefore the fused path) lives on a device.
            if (kv_mode != 0) {
                auto upn = [&](DevPtr& d, const std::vector<float>& src) {
                    if (src.empty()) return;
                    d = DevPtr(cu::alloc(src.size() * sizeof(float)));
                    cu::copy_h2d(d.p, src.data(), src.size() * sizeof(float));
                };
                upn(L.d_qn, L.q_norm);
                upn(L.d_kn, L.k_norm);
                upn(L.d_iqn, L.iq_norm);
                upn(L.d_ikn, L.ik_norm);
            }
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
        mem_probe("weights");
    }

    // ===================================================================================== P3: the expert tiers

    static size_t q8_1_bytes(int64_t n) { return (size_t)(n / 32) * 36; }

    // The device bytes init_prefill will allocate when --minimax-prefill-batch is on.  Auto slot sizing subtracts this
    // so the prefill allocation that happens later cannot run the card out of memory.  It mirrors init_prefill's
    // buffers for the requested batch (the prompt cap is also ctx); the 224 MiB fixed headroom is the same one
    // init_prefill keeps for itself.  This is an over-estimate when the batch is later reduced to fit, which is the
    // safe direction: it can only cost a slot, never a failed allocation.
    size_t prefill_device_bytes(int t_override = 0) const {
        if (opt.prefill_batch <= 1 || dump_on) return 0;
        const int64_t T = t_override > 0 ? std::min<int64_t>(t_override, ctx)
                                         : std::min<int64_t>(opt.prefill_batch, ctx);
        if (T < 2) return 0;
        const size_t t = (size_t)T;
        const size_t rc = t < 1024 ? t : 1024;
        const size_t embd = (size_t)cfg.n_embd;
        size_t maxn = (size_t)cfg.n_ff_dense;
        maxn = maxn > (size_t)cfg.n_head * (size_t)cfg.head_dim ? maxn : (size_t)cfg.n_head * (size_t)cfg.head_dim;
        maxn = maxn > (size_t)cfg.n_ff_exp ? maxn : (size_t)cfg.n_ff_exp;
        maxn = maxn > embd ? maxn : embd;
        maxn = maxn > (size_t)cfg.n_expert ? maxn : (size_t)cfg.n_expert;
        const size_t per_tok = embd * 2 * 2 + maxn * 4 + maxn * 4 * 2 +
                               (size_t)(cfg.n_head * cfg.head_dim) * 2;
        size_t b = t * per_tok;
        // per-token indexer/attention staging, with the +4096 rounding each cu::alloc takes
        b += t * ((size_t)cfg.idx_heads * (size_t)cfg.idx_dim * 2 + 4096);
        b += t * ((size_t)(cfg.n_head * cfg.head_dim) * 2 + 4096);
        const size_t nbmax = (size_t)((ctx + cfg.idx_block - 1) / cfg.idx_block);
        b += t * ((size_t)cfg.idx_heads * nbmax * sizeof(float) + 4096);
        b += t * ((size_t)cfg.idx_heads * (size_t)cfg.idx_top_k * sizeof(int32_t) + 4096);
        b += t * ((size_t)cfg.idx_heads * sizeof(int32_t) + 4096);
        b += t * sizeof(int32_t) + 64;
        // the rows_cap buffers
        b += rc * embd * 2 + rc * embd * 4 + rc * (size_t)cfg.n_ff_exp * 4 * 2 + rc * maxn * 2 + rc * maxn * 4 +
             rc * embd * 4;
        b += rc * (q8_1_bytes(cfg.n_embd) + q8_1_bytes(cfg.n_ff_exp));
        b += rc * (sizeof(int32_t) * 2 + sizeof(float));
        b += (size_t)std::min<int64_t>(4096, (int64_t)maxn) * embd * (2 + 4);
        // the expert ring: init_prefill takes four slots when they fit, and this must assume it does
        if (opt.prefill_experts != "cpu") b += 4 * max_expert_bytes + ((size_t)64 << 20);
        // the RAM tier's staged prompt K/V, when that path is on (it is not for the vram tier)
        if (kv_mode == 2 && opt.prefill_kv != "off" && cfg.n_head_kv == cfg.idx_heads && cfg.idx_first_layer == 0 &&
            kv_q == 0)
            b += 2 * (size_t)kv_row_bytes * (size_t)ctx + ((size_t)64 << 20);
        b += (size_t)224 << 20;
        return b;
    }

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
        // `--minimax-expert-resident all` with the tiers off is a RAM-residency request, not a VRAM one: keep the
        // device for dense/KV unless the caller named a slot count explicitly.
        if (!opt.tiers && opt.expert_resident == "all" && opt.vram_slots_per_layer < 0) q = 0;
        // P10: the batched prompt pass allocates its workspace on every card that owns a layer, after this
        // sizing.  Auto slot sizing must leave it room or a card large enough for the slots to fill the card dies in
        // cudaMalloc (the 96 GB / 153 GB-model startup OOM).  The rule is in vram_plan.hpp and unit-tested; it also
        // covers the rows_cap scratch, the expert ring and the RAM tier's staged K/V, not just the per-token buffers.
        //
        // P12: the workspace never coexists with the *decode* slots -- it is only needed while a prompt is being
        // processed.  `--minimax-prefill-vram swap` therefore sizes the slots as if the workspace did not exist and
        // lets the prompt pass free the slots it displaces (`pf_acquire`), re-allocating them when the prompt
        // finishes, so a batched prompt leaves decode the decode-only slot count.  The default ("reserve") keeps the
        // P10 rule: measured a wash on a 2 x RTX 5070 12 GB box whose slot pool is 2 experts a layer either way (see
        // tools/minimax/README.md), so the swap stays opt-in until a placement with a real slot pool is measured.
        const size_t prefill_need = tiers_gpu_possible() ? prefill_device_bytes() : 0;
        prefill_ws_bytes = prefill_need;
        const bool prefill_swap = prefill_need > 0 && opt.vram_slots_per_layer < 0 && opt.prefill_vram == "swap";
        prefill_ws_swap = prefill_swap;
        if (prefill_need > 0 && opt.vram_slots_per_layer < 0) {
            if (prefill_swap)
                std::fprintf(stderr,
                             "INFO minimax-vram-slots the %d-token batched prompt workspace (%.0f MiB/device) "
                             "borrows the expert slots for the prompt and returns them after it; decode keeps the "
                             "full slot count\n",
                             (int)std::min<int64_t>(opt.prefill_batch, ctx), (double)prefill_need / 1048576.0);
            else
                std::fprintf(stderr,
                             "INFO minimax-vram-slots reserving %.0f MiB/device for the %d-token batched prompt "
                             "workspace before sizing expert slots\n",
                             (double)prefill_need / 1048576.0, (int)std::min<int64_t>(opt.prefill_batch, ctx));
        }
        if (q < 0) {
            q = (int)cfg.n_expert;
            for (size_t d = 0; d < devs.size(); ++d) {
                if (dev_moe[d] <= 0) continue;
                // dev_budget is already free - reserve; the plan takes the batched-prefill workspace out before it
                // counts slots -- except in the swap mode, where the prompt pass takes the slots back for the
                // duration of the prompt instead.  The explicit-count path (`requested >= 0`) is unchanged:
                // init_prefill shrinks the batch to whatever room that count left, as before.
                const VramSlotPlan pl = plan_vram_slots(dev_budget[d], 0, prefill_swap ? 0 : prefill_need,
                                                        max_expert_bytes, (int)dev_moe[d], (int)cfg.n_expert, -1);
                if (pl.slots < q) q = pl.slots;
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
                         " MiB is free after dense/KV/workspace, the " + std::to_string(opt.vram_reserve_mib) +
                         " MiB reserve, and the batched-prompt workspace. Lower --minimax-vram-slots-per-layer or "
                         "--vram-reserve-mib.");
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

        // ---- pinned RAM tier.
        //
        // P7: the budget is sized from the box, not from "available - 2 GiB".  Three things are subtracted before a
        // page is pinned, because a pinned page can never be paged out again:
        //   * an OS + page-cache headroom (1/16 of physical, 3-10 GiB; --minimax-ram-headroom-mib overrides it),
        //   * the dense weights that stay on the host, which are read on every token and must stay resident,
        //   * the host K/V and the workspace, which grow with the context.
        // Then `budget = min(physical, available now) - those`.  Using the physical total as well as the available
        // figure is what makes the same formula right on a 32 GB and a 128 GB machine: `available` alone over-counts
        // when another process holds RAM, and the physical total alone over-counts when the page cache does.
        size_t dense_host_bytes = 0;
        for (const Layer& L : layers) {
            if (L.moe) {
                dense_host_bytes += L.have_shared ? L.gate.bytes + L.up.bytes + L.down.bytes : 0;
            } else if (!L.gate.gpu) {
                dense_host_bytes += L.gate.bytes + L.up.bytes + L.down.bytes;
            }
            for (const Tensor* t : {&L.wq, &L.wk, &L.wv, &L.wo, &L.router, &L.iq, &L.ik})
                if (!t->gpu) dense_host_bytes += t->bytes;
        }
        if (!output_gpu) dense_host_bytes += output.bytes;
        dense_host_bytes += embd.bytes;
        // Host K/V: fp16 K, V and the indexer K for the whole context.  With --minimax-kv vram none of it is host
        // memory (it is the device's, counted there), and with `ram` half of it is page-locked, which the pinned
        // tier must not take.
        const size_t kv_rows = (size_t)cfg.n_layers * (size_t)ctx;
        const size_t host_kv_bytes =
            kv_mode == 1 ? 0
                         : kv_rows * (size_t)cfg.head_dim * (size_t)cfg.n_head_kv * sizeof(uint16_t) * 2 +
                               kv_rows * (size_t)cfg.idx_dim * sizeof(uint16_t);
        // Host work buffers: the token vectors, the MLP scratch and the full-vocabulary logits.
        const size_t host_work_bytes = ((size_t)cfg.n_vocab + 8 * (size_t)cfg.n_ff_exp +
                                        16 * (size_t)cfg.n_embd + 4096) * sizeof(float);

        const size_t phys = host_total_bytes();
        const size_t avail = host_available_bytes();
        size_t headroom = opt.ram_headroom_mib > 0 ? ((size_t)opt.ram_headroom_mib << 20)
                                                   : ram_headroom_auto(phys ? phys : avail);
        size_t ram_budget = 0;
        const size_t overhead = dense_host_bytes + host_kv_bytes + host_work_bytes;
        if (opt.memory_limit_mib > 0) {
            ram_budget = (size_t)opt.memory_limit_mib << 20;
        } else {
            const size_t hold = phys ? phys : avail;
            const size_t room = hold > headroom + overhead ? hold - headroom - overhead : 0;
            const size_t free_now = avail > headroom / 2 ? avail - headroom / 2 : 0;
            ram_budget = std::min(room, free_now);
        }
        // The per-layer split: hottest first from the profile, so a layer whose experts are routed more often gets
        // more slots than a cold one.  With a uniform count the layer budgets ignore the routing profile entirely;
        // the profile mass of MiniMax's layers is not uniform, so a proportional split holds more of the routes in
        // the same number of pinned bytes.
        int r = opt.ram_experts_per_layer;
        std::vector<int> ram_n((size_t)cfg.n_layers, 0);
        if (r >= 0) {
            if (r > cfg.n_expert) r = (int)cfg.n_expert;
            for (int64_t i = 0; i < cfg.n_layers; ++i)
                if (layers[(size_t)i].moe) ram_n[(size_t)i] = r;
        } else if (!profile.empty()) {
            // Walk the global ranked list; each entry offers its layer one more slot for the next-hottest expert
            // that is not already going to VRAM.  The cost of a slot is that layer's expert_bytes, so a file whose
            // layers are not uniform is sized correctly.
            std::vector<int> seen((size_t)cfg.n_layers, 0);
            size_t spent = 0;
            const int cap = (int)cfg.n_expert;
            for (const auto& [pl, pe] : profile) {
                (void)pe;
                Layer& L = layers[(size_t)(pl + cfg.leading_dense)];
                if (!L.moe) continue;
                const int s = seen[(size_t)(pl + cfg.leading_dense)]++;
                const int nn = ram_n[(size_t)(pl + cfg.leading_dense)];
                if (s < vram_per_layer) continue;   // the profile's top q per layer went to VRAM
                if (nn >= cap) continue;
                if (spent + L.expert_bytes > ram_budget) break;
                ram_n[(size_t)(pl + cfg.leading_dense)] = nn + 1;
                spent += L.expert_bytes;
            }
            // Anything the profile did not cover (a truncated STRP, or a model with no profile entry for a layer)
            // is filled round-robin from whatever budget is left, so the tier is never left half empty.
            for (bool progress = true; progress;) {
                progress = false;
                for (int64_t i = 0; i < cfg.n_layers; ++i) {
                    Layer& L = layers[(size_t)i];
                    if (!L.moe || ram_n[(size_t)i] >= cap) continue;
                    if (spent + L.expert_bytes > ram_budget) continue;
                    ++ram_n[(size_t)i];
                    spent += L.expert_bytes;
                    progress = true;
                }
            }
        } else {
            const int uniform = (int)(ram_budget / ((size_t)moe_layers * max_expert_bytes));
            const int capped = uniform > cfg.n_expert ? (int)cfg.n_expert : uniform;
            r = capped < 0 ? 0 : capped;
            for (int64_t i = 0; i < cfg.n_layers; ++i)
                if (layers[(size_t)i].moe) ram_n[(size_t)i] = r;
        }
        int rmin = cfg.n_expert, rmax = 0;
        auto recount = [&]() {
            rmin = cfg.n_expert;
            rmax = 0;
            for (int64_t i = 0; i < cfg.n_layers; ++i)
                if (layers[(size_t)i].moe) {
                    rmin = std::min(rmin, ram_n[(size_t)i]);
                    rmax = std::max(rmax, ram_n[(size_t)i]);
                }
            if (rmax == 0) rmin = 0;
        };
        recount();
        for (Layer& L : layers)
            if (L.moe) L.ram_of.assign((size_t)cfg.n_expert, kNotResident);
        if (rmax > 0) {
            constexpr uint64_t kRegion = (uint64_t)2 << 20;   // 2 MiB-aligned regions: large pages, and a flat TLB
            uint64_t off = 0;
            size_t slots = 0, bytes = 0;
            auto lay_out = [&]() {
                off = 0;
                slots = 0;
                bytes = 0;
                for (int64_t li = 0; li < cfg.n_layers; ++li) {
                    Layer& L = layers[(size_t)li];
                    if (!L.moe) continue;
                    off = (off + kRegion - 1) & ~(kRegion - 1);
                    L.ram_off = off;
                    L.ram_n = ram_n[(size_t)li];
                    off += (uint64_t)L.ram_n * L.expert_bytes;
                    slots += (size_t)L.ram_n;
                    bytes += (size_t)L.ram_n * L.expert_bytes;
                }
                ram_bytes = off;
            };
            lay_out();
            if (ram_inplace) {
                // No arena at all: every slot is a ranking decision (see ram_fill).
                ram_arena.reset();
            } else {
                ram_arena = std::make_unique<strata::core::PinnedArena>(ram_bytes);
            }
            // An automatic budget that the OS will not let us pin is not an error: the estimate is the operating
            // system's to refuse, and a box with a smaller locked-page allowance (or a competing process holding
            // RAM) would otherwise fail to start the model at all.  Step down until it fits, and say so.
            int attempt = 0;
            while (!ram_inplace && !ram_arena->valid() && ram_bytes > 0) {
                if (opt.memory_limit_mib > 0 || opt.ram_experts_per_layer >= 0 || ++attempt > 8) {
                    fail("the pinned RAM expert tier could not reserve " +
                         std::to_string(ram_bytes / (1024 * 1024)) + " MiB (" + ram_arena->note +
                         "). Lower --memory-limit-mib or --minimax-ram-experts-per-layer.");
                }
                const size_t want = slots > 8 ? (size_t)((double)slots * 0.85) : 0;
                while (slots > want) {
                    int big = -1;
                    for (int64_t i = 0; i < cfg.n_layers; ++i)
                        if (layers[(size_t)i].moe && ram_n[(size_t)i] > 0 &&
                            (big < 0 || ram_n[(size_t)i] > ram_n[(size_t)big]))
                            big = (int)i;
                    if (big < 0) break;
                    --ram_n[(size_t)big];
                    --slots;
                }
                lay_out();
                std::fprintf(stderr,
                             "INFO minimax-ram-tier: %zu MiB will not pin (%s); retrying with %zu slots "
                             "(%.1f GiB)\n",
                             (size_t)(bytes / (size_t)(bytes && slots ? slots : 1)),
                             ram_arena->note.c_str(), slots, (double)ram_bytes / 1073741824.0);
                ram_arena = std::make_unique<strata::core::PinnedArena>(ram_bytes);
                if (slots == 0) break;
            }
            recount();
            ram_per_layer = rmax;
            if (ram_bytes > 0) {
                for (Layer& L : layers) {
                    if (!L.moe) continue;
                    L.ram_of.assign((size_t)cfg.n_expert, kNotResident);
                    L.ram_expert_of.assign((size_t)L.ram_n, kNotResident);
                    L.ram_tick.assign((size_t)L.ram_n, 0);
                }
                std::fprintf(stderr,
                             "INFO minimax-ram-tier plan budget=%.1f GiB headroom=%.1f GiB dense_host=%.1f GiB "
                             "kv=%.1f GiB reserved=%.1f GiB slots=%zu experts/layer=%d..%d mean=%.1f "
                             "source=%s mode=%s\n",
                             (double)ram_budget / 1073741824.0, (double)headroom / 1073741824.0,
                             (double)dense_host_bytes / 1073741824.0, (double)host_kv_bytes / 1073741824.0,
                             (double)ram_bytes / 1073741824.0, slots, rmin, rmax,
                             slots ? (double)slots / (double)moe_layers : 0.0,
                             opt.memory_limit_mib > 0
                                 ? "--memory-limit-mib"
                                 : (opt.ram_experts_per_layer >= 0 ? "--minimax-ram-experts-per-layer" : "auto"),
                             ram_inplace ? "inplace" : "pinned");
                mem_probe("tier-plan");
            }
            (void)bytes;
        }

        // ---- SSD: unbuffered reads straight from the GGUF shards.  DirectFile owns the aligned OVERLAPPED /
        // O_DIRECT I/O; this keeps the offsets and the aligned superset rule.
        //
        // P9: this comes before the profile-driven fill, because with `--minimax-prefill-read direct` (implied by a
        // --memory-limit-mib budget) the pinned arena is filled through these files rather than through the mapping.
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
        ssd_tmp = strata::platform::DirectFile::alloc_aligned(ssd_stage_bytes + 4096);
        if (!ssd_tmp) fail("minimax SSD tier: cannot allocate the 4 KiB-aligned read buffer");
        ssd_tmp_bytes = ssd_stage_bytes + 4096;

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
                for (int s = 0; s < L.ram_n && at < (int)order.size(); ++s, ++at)
                    ram_fill(L, s, order[(size_t)at].second);
            }
        }

        // ---- SSD staging buffers for the token path (the files and ssd_tmp were opened above, before the fill).
        ssd_stage = (uint8_t*)strata::platform::DirectFile::alloc_aligned(ssd_stage_bytes);
        if (!ssd_stage) fail("minimax SSD tier: cannot allocate the 4 KiB-aligned staging buffer");

        // P6: the token path reads each role straight into its own place.  A role's buffer holds the aligned
        // superset of the tensor's bytes (`off & 4095` bytes in front, up to 4095 behind), so the read lands where
        // the dot kernels read it and there is no copy out of a shared buffer.
        {
            size_t max_gu = 0, max_dn = 0;
            for (const Layer& L : layers)
                if (L.moe) {
                    if (L.expert_gu > max_gu) max_gu = L.expert_gu;
                    if (L.expert_down > max_dn) max_dn = L.expert_down;
                }
            const size_t A = strata::platform::DirectFile::alignment();
            auto up = [&](size_t v) { return (v + A - 1) & ~(A - 1); };
            ssd_gu_cap = up(max_gu + 2 * A);
            ssd_dn_cap = up(max_dn + 2 * A);
            ssd_slot_bytes = up(2 * ssd_gu_cap + ssd_dn_cap);
            // P7 --minimax-ssd-prefetch guesses the next layer's routes from the residual, which is only a
            // prediction (the next layer's attention moves the residual before its router runs), so a layer can
            // need slots for routes nobody predicted.  Each of the `depth` prefetch regions holds 2 x top slots:
            // the predicted routes first, then room for the layer's own unpredicted misses.  A region is reused
            // only after `ssd_wait_layer` has drained the reads that were in it, so an in-flight read can never
            // be overwritten by the next prediction.
            const size_t per_region = (size_t)cfg.n_expert_used * 2;
            ssd_slot_cap = per_region * (size_t)(ssd_prefetch ? ssd_prefetch_depth + 1 : 1);
            ssd_slots = (uint8_t*)strata::platform::DirectFile::alloc_aligned(ssd_slot_cap * ssd_slot_bytes);
            if (!ssd_slots) fail("minimax SSD tier: cannot allocate the 4 KiB-aligned expert staging slots");
            ssd_pending.assign(ssd_files.size(), 0);
            ssd_pf_out.assign((size_t)cfg.n_layers, 0);
            ssd_pf_out_shard.assign((size_t)cfg.n_layers * ssd_files.size(), 0);
            ssd_pf_exp.assign((size_t)cfg.n_layers * (size_t)cfg.n_expert_used, -1);
            ssd_pf_slot.assign((size_t)cfg.n_layers * (size_t)cfg.n_expert_used, -1);
            ssd_pf_n.assign((size_t)cfg.n_layers, 0);
        }

        const size_t cells = (size_t)cfg.n_layers * (size_t)cfg.n_expert;
        if (opt.adaptive) usage.assign(cells, 0.f);
        if (!opt.profile_save_path.empty() || !opt.routing_counts_path.empty()) route_counts.assign(cells, 0.0);

        // P5c: every static RAM slot is filled here, once, in parallel, instead of on the first token that
        // routes to it.
        if (expert_resident && (ram_bytes > 0 || (ram_inplace && ram_per_layer > 0)) &&
            ((ram_arena && ram_arena->valid()) || ram_inplace))
            fill_ram_static();
    }

    bool tiers_gpu_possible() const { return opt.dense_layers_gpu != 0 || opt.tiers; }

    void vram_fill(Layer& L, int slot, int expert) {
        if (slot < 0 || slot >= (int)L.vram.size()) fail("VRAM slot out of range");
        use_device_index(L.dev);
        Layer::VramSlot& s = L.vram[(size_t)slot];
        if (s.expert >= 0) L.vram_of[(size_t)s.expert] = kNotResident;
        uint8_t* p = (uint8_t*)s.mem.p;
        // P12: if the expert is already in the pinned RAM tier, copy it from there.  It used to be read out of the
        // file mapping even when the arena held the same bytes, so every compulsory admission re-faulted 15 MB of
        // the model through the page cache (or the drive) instead of a 20 GB/s memcpy from RAM.  Same bytes.
        const uint8_t* g = L.egate.host + (size_t)expert * L.expert_gu;
        const uint8_t* u = L.eup.host + (size_t)expert * L.expert_gu;
        const uint8_t* d = L.edown.host + (size_t)expert * L.expert_down;
        if (!ram_inplace && ram_arena && ram_arena->valid() && L.ram_n > 0) {
            const int rs = L.ram_of[(size_t)expert];
            if (rs >= 0 && rs < L.ram_n) {
                const uint8_t* base = ram_arena->data() + L.ram_off + (uint64_t)rs * L.expert_bytes;
                g = base;
                u = base + L.expert_gu;
                d = base + 2 * L.expert_gu;
            }
        }
        cu::copy_h2d(p, g, L.expert_gu);
        cu::copy_h2d(p + L.expert_gu, u, L.expert_gu);
        cu::copy_h2d(p + 2 * L.expert_gu, d, L.expert_down);
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
        if (ram_inplace) fail("internal: an in-place RAM tier has no arena slot");
        return ram_arena->data() + L.ram_off + (uint64_t)slot * L.expert_bytes;
    }

    void ram_fill(Layer& L, int slot, int expert) {
        if (ram_inplace) {
            // --minimax-ram-mode inplace: the "slot" is a ranking decision, not storage.  The expert stays where it
            // is in the mapping and the OS page cache is the tier (Project Maya's DeepSeek engine reads in place
            // and measured its own opt-in RAM copy as no faster: ds4_model.cu:1206-1211, 5164-5166).
            for (int32_t& v : L.ram_of)
                if (v == slot) v = kNotResident;
            L.ram_of[(size_t)expert] = slot;
            L.ram_expert_of[(size_t)slot] = expert;
            L.ram_tick[(size_t)slot] = ++lru_tick;
            return;
        }
        if (!ram_arena || !ram_arena->valid()) fail("RAM tier is not allocated");
        uint8_t* p = ram_slot_ptr(L, slot);
        ram_copy(L, expert, p);
        for (int32_t& v : L.ram_of)
            if (v == slot) v = kNotResident;
        L.ram_of[(size_t)expert] = slot;
        L.ram_expert_of[(size_t)slot] = expert;
        L.ram_tick[(size_t)slot] = ++lru_tick;
    }

    // One expert's three role slices into `p`, spread over the pool.  The source is the mmap, so the cost is the
    // soft page fault per 4 KiB: many threads is the difference between 8 and 20 GB/s on a cold mapping.
    // P9: with a byte budget the mapping is not the source -- an unbuffered read is, so filling the tier costs no
    // page cache.  Same bytes either way.
    void ram_copy(const Layer& L, int expert, uint8_t* p) {
        if (stream_direct) {
            read_expert_roles(L, expert, p);
            if ((++mem_probe_n % 512) == 0) mem_probe("fill");
            return;
        }
        const size_t gu = L.expert_gu, dn = L.expert_down;
        const uint8_t* src[3] = {L.egate.host + (size_t)expert * gu, L.eup.host + (size_t)expert * gu,
                                 L.edown.host + (size_t)expert * dn};
        const size_t off[3] = {0, gu, 2 * gu};
        const size_t len[3] = {gu, gu, dn};
        const int pieces = 8 * (threads > 0 ? threads : 4);
        pool.run(pieces * 3, [&](int a, int b) {
            for (int i = a; i < b; ++i) {
                const int m = i / pieces, c = i % pieces;
                const size_t p0 = len[m] * (size_t)c / (size_t)pieces;
                const size_t p1 = len[m] * (size_t)(c + 1) / (size_t)pieces;
                if (p1 > p0) std::memcpy(p + off[m] + p0, src[m] + p0, p1 - p0);
            }
        });
    }

    // P9: one line per stage with the process's working set, its high-water mark, the machine's available RAM and
    // the system commit charge.  A budget is only kept if it can be seen; this is what bench_longctx.py's samples
    // are cross-checked against.
    static void mem_probe(const char* stage) {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS pmc{};
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        GlobalMemoryStatusEx(&ms);
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
            std::fprintf(stderr,
                         "INFO minimax-mem %-12s ws=%.1f GiB peak=%.1f GiB avail=%.1f GiB commit=%.1f GiB\n",
                         stage, (double)pmc.WorkingSetSize / 1073741824.0,
                         (double)pmc.PeakWorkingSetSize / 1073741824.0, (double)ms.ullAvailPhys / 1073741824.0,
                         (double)(ms.ullTotalPageFile - ms.ullAvailPageFile) / 1073741824.0);
        }
#else
        (void)stage;
#endif
    }

    // P9: hand the process's clean mapped pages back to the standby list.  Windows only trims a working set under
    // memory pressure, which is why the round-1 32 K run ended with 0.3 GB of available RAM while the pinned arena
    // was inside its budget.  Locked/pinned pages are not trimmable, so the tier the budget paid for stays resident;
    // what leaves the working set is the file mapping the reads went through.
    void trim_mapped_pages() const {
#if defined(_WIN32)
        if (!stream_direct) return;
        EmptyWorkingSet(GetCurrentProcess());
#endif
    }

    // P5c: fill every static RAM slot at startup instead of on first use.  The on-demand fill re-read the same
    // bytes through the mmap once per expert (at 16 GB/s in the measured decode, i.e. exactly the fault cost the
    // tier exists to avoid) and left the arena a cold, partly-empty cache for the first tokens.
    void fill_ram_static() {
        struct Job {
            Layer* L;
            int slot;
            int expert;
        };
        std::vector<Job> jobs;
        // Priority: the profile's rank order when there is one, then the lowest unused expert ids.
        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            if (!L.moe || L.ram_n <= 0) continue;
            std::vector<char> used((size_t)cfg.n_expert, 0);
            std::vector<int> rank((size_t)cfg.n_expert, -1);
            if (!profile.empty()) {
                const int ml = (int)(li - cfg.leading_dense);
                for (size_t i = 0; i < profile.size(); ++i) {
                    const auto [pl, pe] = profile[i];
                    if (pl == ml && pe >= 0 && pe < (int)cfg.n_expert) rank[(size_t)pe] = (int)i;
                }
            }
            std::vector<std::pair<int, int>> order;
            for (int e = 0; e < (int)cfg.n_expert; ++e) order.push_back({rank[(size_t)e] < 0 ? INT32_MAX : rank[(size_t)e], e});
            std::sort(order.begin(), order.end());
            for (int s = 0; s < L.ram_n; ++s) {
                if (L.ram_expert_of[(size_t)s] >= 0) {
                    used[(size_t)L.ram_expert_of[(size_t)s]] = 1;
                    continue;
                }
                int pick = -1;
                for (const auto& [r, e] : order)
                    if (!used[(size_t)e] && L.ram_of[(size_t)e] < 0) {
                        pick = e;
                        break;
                    }
                if (pick < 0) break;
                used[(size_t)pick] = 1;
                jobs.push_back({&L, s, pick});
            }
        }
        if (jobs.empty()) return;
        if (ram_inplace) {
            // P7b: in-place mode.  Record the ranked set (so the route nudge and the counters know what is meant to
            // be resident) and hint the pages into the page cache, but copy nothing.  Project Maya's Windows
            // prefetch of only the non-resident pages measured +6-7% decode (CHANGELOG v1.1.26, ds4_os.hpp:250-332).
            for (const Job& j : jobs) {
                j.L->ram_of[(size_t)j.expert] = j.slot;
                j.L->ram_expert_of[(size_t)j.slot] = j.expert;
                j.L->ram_tick[(size_t)j.slot] = ++lru_tick;
            }
            const auto t0 = std::chrono::steady_clock::now();
            size_t bytes = 0;
#if defined(_WIN32)
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            ranges.reserve(jobs.size() * 3);
#endif
            for (const Job& j : jobs) {
                const Layer& L = *j.L;
                const uint8_t* src[3] = {L.egate.host + (size_t)j.expert * L.expert_gu,
                                         L.eup.host + (size_t)j.expert * L.expert_gu,
                                         L.edown.host + (size_t)j.expert * L.expert_down};
                const size_t len[3] = {L.expert_gu, L.expert_gu, L.expert_down};
                for (int r = 0; r < 3; ++r) {
                    bytes += len[r];
#if defined(_WIN32)
                    ranges.push_back({(PVOID)src[r], len[r]});
#else
                    (void)src[r];
                    madvise((void*)src[r], len[r], MADV_WILLNEED);
#endif
                }
            }
#if defined(_WIN32)
            // One call per 512 ranges: PrefetchVirtualMemory is a page-table walk (~50 ns per 4 KiB page).
            for (size_t at = 0; at < ranges.size(); at += 512) {
                const DWORD n = (DWORD)std::min<size_t>(512, ranges.size() - at);
                PrefetchVirtualMemory(GetCurrentProcess(), n, ranges.data() + at, 0);
            }
#endif
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr,
                         "INFO minimax-ram-tier inplace experts=%zu gib=%.1f prefetch_seconds=%.1f reads_from=mmap\n",
                         jobs.size(), (double)bytes / 1073741824.0, s);
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (stream_direct) {
            // P9: fill the arena with unbuffered reads instead of memcpy out of the mapping.  The bytes are the same,
            // so the tier is bit-identical; what changes is that the 74 GiB that just became pinned is not also left
            // in the page cache (which is what took the working set to 112.8 GB in the round-1 32 K run).
            size_t bytes = 0;
            for (const Job& j : jobs) {
                read_expert_roles(*j.L, j.expert, ram_slot_ptr(*j.L, j.slot));
                j.L->ram_of[(size_t)j.expert] = j.slot;
                j.L->ram_expert_of[(size_t)j.slot] = j.expert;
                j.L->ram_tick[(size_t)j.slot] = ++lru_tick;
                bytes += j.L->expert_bytes;
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr,
                         "INFO minimax-ram-tier experts=%zu gib=%.1f seconds=%.1f rate=%.1f GiB/s backing=%s "
                         "reads_from=direct\n",
                         jobs.size(), (double)bytes / 1073741824.0, s,
                         s > 0 ? (double)bytes / 1073741824.0 / s : 0.0,
                         ram_arena ? (ram_arena->backing == strata::core::PageBacking::LargePages
                                          ? "large-pages"
                                          : ram_arena->backing == strata::core::PageBacking::PinnedByCuda ? "cuda-pinned"
                                                                                                          : "locked")
                                   : "none");
            trim_mapped_pages();
            return;
        }
        // ONE pool.run OVER PIECES, NOT a pool.run per expert from our own thread band: Pool::run is not
        // reentrant, and a band of threads each calling it deadlocked the tier fill (measured: 27 s of CPU and no
        // progress).  ~1 MiB pieces are small enough that every core gets several and large enough that the
        // syscall/memcpy overhead disappears.
        struct Piece {
            const uint8_t* src;
            uint8_t* dst;
            size_t len;
        };
        std::vector<Piece> pieces;
        const size_t kPiece = (size_t)1 << 20;
        for (const Job& j : jobs) {
            const Layer& L = *j.L;
            uint8_t* dst = ram_slot_ptr(*j.L, j.slot);
            const size_t len[3] = {L.expert_gu, L.expert_gu, L.expert_down};
            const size_t off[3] = {0, L.expert_gu, 2 * L.expert_gu};
            const uint8_t* src[3] = {L.egate.host + (size_t)j.expert * L.expert_gu,
                                     L.eup.host + (size_t)j.expert * L.expert_gu,
                                     L.edown.host + (size_t)j.expert * L.expert_down};
            for (int r = 0; r < 3; ++r)
                for (size_t at = 0; at < len[r]; at += kPiece)
                    pieces.push_back({src[r] + at, dst + off[r] + at,
                                      std::min(kPiece, len[r] - at)});
        }
        pool.run((int)pieces.size(), [&](int a, int b) {
            for (int i = a; i < b; ++i) std::memcpy(pieces[(size_t)i].dst, pieces[(size_t)i].src, pieces[(size_t)i].len);
        });
        uint64_t bytes = 0;
        for (const Job& j : jobs) {
            j.L->ram_of[(size_t)j.expert] = j.slot;
            j.L->ram_expert_of[(size_t)j.slot] = j.expert;
            j.L->ram_tick[(size_t)j.slot] = ++lru_tick;
            bytes += j.L->expert_bytes;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
                     "INFO minimax-ram-tier experts=%zu gib=%.1f seconds=%.1f rate=%.1f GiB/s backing=%s\n",
                     jobs.size(), (double)bytes / 1073741824.0, s,
                     s > 0 ? (double)bytes / 1073741824.0 / s : 0.0,
                     ram_arena ? (ram_arena->backing == strata::core::PageBacking::LargePages
                                      ? "large-pages"
                                      : ram_arena->backing == strata::core::PageBacking::PinnedByCuda ? "cuda-pinned"
                                                                                                      : "locked")
                               : "none");
        if (ram_arena && ram_arena->backing != strata::core::PageBacking::LargePages)
            std::fprintf(stderr, "INFO minimax-ram-tier note: %s\n", ram_arena->note.c_str());
    }

    // Copy an expert that was just read from the SSD into the least-recently-used RAM slot.  The bytes are already
    // in `src`; a later SSD miss for this expert then costs a RAM read instead.
    void ram_admit(Layer& L, int expert, const uint8_t* src) {
        if (!opt.adaptive || ram_inplace || !ram_arena || L.ram_n <= 0) return;
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

    // P6: the same admission from the three role buffers of a staging slot (the P6 read path keeps a role's bytes
    // in place instead of assembling one contiguous blob).
    void ram_admit3(Layer& L, int expert, const uint8_t* g, const uint8_t* u, const uint8_t* d) {
        if (!opt.adaptive || ram_inplace || !ram_arena || L.ram_n <= 0) return;
        int victim = 0;
        for (int s = 1; s < L.ram_n; ++s)
            if (L.ram_tick[(size_t)s] < L.ram_tick[(size_t)victim]) victim = s;
        uint8_t* p = ram_slot_ptr(L, victim);
        std::memcpy(p, g, L.expert_gu);
        std::memcpy(p + L.expert_gu, u, L.expert_gu);
        std::memcpy(p + 2 * L.expert_gu, d, L.expert_down);
        for (int32_t& v : L.ram_of)
            if (v == victim) v = kNotResident;
        L.ram_of[(size_t)expert] = victim;
        L.ram_expert_of[(size_t)victim] = expert;
        L.ram_tick[(size_t)victim] = ++lru_tick;
    }

    // An SSD-tier route whose bytes are in a staging slot now, to be admitted into the RAM tier after the layer.
    struct SsdAdmit {
        int e;
        const uint8_t* g;
        const uint8_t* u;
        const uint8_t* d;
    };

    // P8: a whole layer's admissions in one parallel copy.  `ram_admit3` moves 17.4 MB on the calling thread for
    // every SSD miss, and on the 15-layer slice that is 16-60 ms a token (--minimax-profile-time, `admit=`), which
    // sat between two layers with nothing to hide behind.  The victim choice is still serial and LRU; only the
    // memcpys are spread, in ~1 MiB pieces (the same shape as `fill_ram_static`).  A slot picked by an earlier
    // admission of the same layer is skipped, so two copies never land on the same bytes.
    void ram_admit_many(Layer& L, const std::vector<SsdAdmit>& list) {
        if (!opt.adaptive || ram_inplace || !ram_arena || L.ram_n <= 0 || list.empty()) return;
        struct Piece {
            const uint8_t* src;
            uint8_t* dst;
            size_t len;
        };
        std::vector<Piece> pieces;
        pieces.reserve(list.size() * 12);
        std::vector<int> used;
        used.reserve(list.size());
        std::vector<std::pair<int, int>> placed;   // (expert, slot)
        placed.reserve(list.size());
        const size_t kPiece = (size_t)1 << 20;
        for (const SsdAdmit& a : list) {
            if (L.ram_of[(size_t)a.e] >= 0) continue;   // one route of this layer already admitted it
            int victim = -1;
            for (int s = 0; s < L.ram_n; ++s) {
                if (std::find(used.begin(), used.end(), s) != used.end()) continue;
                if (victim < 0 || L.ram_tick[(size_t)s] < L.ram_tick[(size_t)victim]) victim = s;
            }
            if (victim < 0) break;
            used.push_back(victim);
            uint8_t* p = ram_slot_ptr(L, victim);
            const size_t len[3] = {L.expert_gu, L.expert_gu, L.expert_down};
            const uint8_t* src[3] = {a.g, a.u, a.d};
            const size_t off[3] = {0, L.expert_gu, 2 * L.expert_gu};
            for (int r = 0; r < 3; ++r)
                for (size_t at = 0; at < len[r]; at += kPiece)
                    pieces.push_back({src[r] + at, p + off[r] + at, std::min(kPiece, len[r] - at)});
            placed.push_back({a.e, victim});
        }
        if (pieces.empty()) return;
        pool.run((int)pieces.size(), [&](int x, int y) {
            for (int i = x; i < y; ++i)
                std::memcpy(pieces[(size_t)i].dst, pieces[(size_t)i].src, pieces[(size_t)i].len);
        });
        for (const auto& pr : placed) {
            for (int32_t& v : L.ram_of)
                if (v == pr.second) v = kNotResident;
            L.ram_of[(size_t)pr.first] = pr.second;
            L.ram_expert_of[(size_t)pr.second] = pr.first;
            L.ram_tick[(size_t)pr.second] = ++lru_tick;
        }
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

    // P9: the same three role slices, read unbuffered, without touching the tier counters.  `dst` receives
    // [gate|up|down] exactly as the mapping would present them, so a caller can hand it to the same kernels.
    void read_expert_roles(const Layer& L, int64_t expert, uint8_t* dst) {
        ssd_read_role(L.egate, expert, dst);
        ssd_read_role(L.eup, expert, dst + L.expert_gu);
        ssd_read_role(L.edown, expert, dst + 2 * L.expert_gu);
    }

    // P10: one expert's three roles read unbuffered straight into the pinned ring slot, the three requests in flight
    // together (the old path read them one at a time into one staging buffer and then copied 19 MiB).  `slot` and
    // each `slot + P().role_off[r]` are 4 KiB aligned; a file offset is not, so `out[r]` is the byte the role starts
    // at inside its window.  Returns false with `err` set on a failed or short read.
    bool ssd_read_roles_direct(const Layer& L, int64_t expert, uint8_t* slot, const uint8_t* out[3],
                               std::string& err) {
        const Tensor* ts[3] = {&L.egate, &L.eup, &L.edown};
        const size_t A = strata::platform::DirectFile::alignment();
        strata::platform::DirectFile* files[3] = {nullptr, nullptr, nullptr};
        int need[3] = {0, 0, 0};
        int nf = 0;
        uint64_t want[3] = {0, 0, 0};
        for (int r = 0; r < 3; ++r) {
            const Tensor& t = *ts[r];
            const size_t len = t.bytes / (size_t)t.ne2;
            const uint64_t off = t.file_offset + (uint64_t)expert * (uint64_t)len;
            const uint64_t aoff = off & ~(uint64_t)(A - 1);
            const uint64_t aend = (off + len + A - 1) & ~(uint64_t)(A - 1);
            const uint32_t alen = (uint32_t)(aend - aoff);
            uint8_t* dst = slot + pf[(size_t)cur].role_off[r];
            if (((uintptr_t)dst & (uintptr_t)(A - 1)) != 0) {
                err = "the pinned expert slot is not 4 KiB aligned";
                return false;
            }
            auto* f = ssd_files[t.shard].get();
            if (!f->submit(aoff, dst, alen, (uint64_t)r, err)) return false;
            // A read of the file's last block can be short in its aligned tail; only the role's own bytes matter.
            want[r] = (off - aoff) + (uint64_t)len;
            out[r] = dst + (size_t)(off - aoff);
            int k = 0;
            while (k < nf && files[k] != f) ++k;
            if (k == nf) {
                files[nf] = f;
                need[nf] = 0;
                ++nf;
            }
            ++need[k];
        }
        bool done[3] = {false, false, false};
        int left = 3;
        for (int spin = 0; left > 0 && spin < 1000000; ++spin) {
            bool any = false;
            for (int k = 0; k < nf; ++k) {
                if (need[k] <= 0) continue;
                strata::platform::Completion c{};
                if (files[k]->wait(&c, 1, 0) != 1) continue;
                if (c.tag == strata::platform::DirectFile::WAKE_TAG) continue;
                const int r = (int)c.tag;
                if (r < 0 || r > 2 || done[r]) {
                    err = "minimax SSD read: unexpected completion tag";
                    return false;
                }
                if (!c.ok || (uint64_t)c.bytes < want[r]) {
                    err = "minimax SSD read: I/O failed or short read";
                    return false;
                }
                done[r] = true;
                --need[k];
                --left;
                any = true;
            }
            if (!any && left > 0) {
                for (int k = 0; k < nf; ++k) {
                    if (need[k] <= 0) continue;
                    strata::platform::Completion c{};
                    if (files[k]->wait(&c, 1, -1) != 1) continue;
                    if (c.tag == strata::platform::DirectFile::WAKE_TAG) continue;
                    const int r = (int)c.tag;
                    if (r < 0 || r > 2 || done[r]) {
                        err = "minimax SSD read: unexpected completion tag";
                        return false;
                    }
                    if (!c.ok || (uint64_t)c.bytes < want[r]) {
                        err = "minimax SSD read: I/O failed or short read";
                        return false;
                    }
                    done[r] = true;
                    --need[k];
                    --left;
                    break;
                }
            }
        }
        if (left != 0) {
            err = "minimax SSD read: timed out";
            return false;
        }
        ++tok_stats.ssd;
        tok_stats.ssd_bytes += (int64_t)L.expert_bytes;
        return true;
    }

    // ---- P6: the token path's SSD staging, queued rather than waited on.
    //
    // `ssd_slot_role(slot, role)` is a 4 KiB-aligned buffer with 4 KiB of headroom in front of the tensor's bytes.
    // The aligned superset of a role's range is read into `base` and the kernel reads the data at
    // `base + (off & 4095)`, so nothing is copied after the read.  A role is cut into `ssd_chunks` pieces of 4 KiB
    // multiples that are all queued at once: one expert is then one queue depth instead of three serial round trips,
    // and the layer's whole miss set is in flight while the CPU and the GPU work on the other experts.
    uint8_t* ssd_slot_role(int slot, int role) const {
        uint8_t* b = ssd_slots + (size_t)slot * ssd_slot_bytes;
        return b + (role == 0 ? 0 : role == 1 ? ssd_gu_cap : 2 * ssd_gu_cap);
    }

    // A layer's staging region: `2 * top` slots, the first `top` for the routes prefetched for it and the rest for
    // misses the prediction did not name.  With depth 1 the region is the whole pool, as before.
    size_t ssd_region_slots() const { return (size_t)cfg.n_expert_used * 2; }
    // depth + 1 regions: while layer `li` computes in region `li % (depth+1)`, the prediction for `li + depth`
    // lands in the region vacated by `li - 1`, which is finished.  Neither can touch the other's slots.
    size_t ssd_region_base(int64_t li) const {
        if (!ssd_prefetch || ssd_prefetch_depth < 1) return 0;
        return (size_t)(li % (int64_t)(ssd_prefetch_depth + 1)) * ssd_region_slots();
    }

    // Queue one role's bytes so they land in their final place; returns the pointer the dot kernels read.
    // `tag` is 0 for the layer's own miss and `layer + 1` for a read queued a layer (or more) early; the tag is
    // what lets a prefetch region be drained without waiting for the current layer's reads.
    uint8_t* ssd_submit_role(const Tensor& t, int64_t expert, uint8_t* base, uint64_t tag = 0) {
        const size_t len = t.bytes / (size_t)t.ne2;
        const uint64_t off = t.file_offset + (uint64_t)expert * (uint64_t)len;
        constexpr uint64_t A = strata::platform::DirectFile::alignment();
        const uint64_t aoff = off & ~(A - 1);
        const uint64_t aend = (off + len + A - 1) & ~(A - 1);
        const uint32_t alen = (uint32_t)(aend - aoff);
        strata::platform::DirectFile& file = *ssd_files[t.shard];
        const int nch = ssd_chunks < 1 ? 1 : ssd_chunks;
        uint32_t step = alen;
        if (nch > 1) {
            step = (uint32_t)((((uint64_t)alen + (uint64_t)nch - 1) / (uint64_t)nch + A - 1) & ~(A - 1));
            if (step == 0) step = (uint32_t)A;
        }
        std::string err;
        for (int c = 0; c < nch; ++c) {
            const uint32_t o = (uint32_t)c * step;
            if (o >= alen) break;
            const uint32_t l = alen - o < step ? alen - o : step;
            if (!file.submit(aoff + o, base + o, l, tag, err)) fail("minimax SSD read: " + err);
            if (tag == 0) {
                ++ssd_pending[t.shard];
            } else {
                ++ssd_pf_out[(size_t)(tag - 1)];
                ++ssd_pf_out_shard[(size_t)(tag - 1) * ssd_files.size() + t.shard];
            }
        }
        return base + (size_t)(off - aoff);
    }

    // Collect up to 64 completions from one shard and account them by tag.  Returns the number the port
    // delivered (0 when `timeout_ms` is 0 and nothing was ready).  A completion is the only place a prefetch
    // read stops being outstanding, whichever layer's wait happens to reap it.
    int ssd_pump(int f, int max, int timeout_ms) {
        strata::platform::Completion buf[64];
        const int want = max < 64 ? max : 64;
        const int n = ssd_files[(size_t)f]->wait(buf, want, timeout_ms);
        if (n <= 0) return 0;
        for (int i = 0; i < n; ++i) {
            if (buf[i].tag == strata::platform::DirectFile::WAKE_TAG) continue;
            if (!buf[i].ok) fail("minimax SSD read: I/O failed");
            if (buf[i].tag == 0) {
                --ssd_pending[(size_t)f];
            } else {
                const size_t li = (size_t)(buf[i].tag - 1);
                if (li >= ssd_pf_out.size() || ssd_pf_out[li] <= 0)
                    fail("minimax SSD read: completion for a prefetch layer that is not in flight");
                --ssd_pf_out[li];
                int& by_shard = ssd_pf_out_shard[li * ssd_files.size() + (size_t)f];
                if (by_shard > 0) --by_shard;
            }
        }
        return n;
    }

    void ssd_wait_all() {
        if (ssd_pending.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t f = 0; f < ssd_pending.size(); ++f) {
            // Drain a batch, not one completion at a time: a layer of a 57-layer model queues a few hundred
            // reads, and one completion-port call each was measurable host time of its own.  Prefetch
            // completions that arrive here are simply accounted and dropped; the data is already in its slot.
            while (ssd_pending[f] > 0) {
                if (ssd_pump((int)f, 64, -1) == 0) continue;
            }
        }
        ssd_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // Wait until every read queued for layer `li` a few layers ago has completed, so its staging region can be
    // reused.  This is deliberately per layer: the current layer's own misses are `ssd_wait_all`, and blocking on
    // the deeper predictions there would throw away the lookahead.
    void ssd_wait_layer(int li) {
        if (ssd_pf_out.empty() || li < 0 || li >= (int)ssd_pf_out.size()) return;
        while (ssd_pf_out[(size_t)li] > 0) {
            int progressed = 0;
            int block_on = -1;
            for (size_t f = 0; f < ssd_files.size() && ssd_pf_out[(size_t)li] > 0; ++f) {
                if (ssd_pf_out_shard[(size_t)li * ssd_files.size() + f] <= 0) continue;
                if (block_on < 0) block_on = (int)f;
                progressed += ssd_pump((int)f, 64, 0);
            }
            // Nothing was ready: sleep on a shard this layer actually reads from, never on shard 0 by assumption.
            if (progressed == 0 && block_on >= 0 && ssd_pf_out[(size_t)li] > 0) ssd_pump(block_on, 64, -1);
        }
    }

    // P7: is every page covering [p, p + len) in this process's working set?  If so, reading the expert's bytes
    // through the mapping is a memory read and the disk need not be touched at all.  This is Strata 0.2.0's
    // `mincore` zero-copy (docs/DETAILS.md: "before, every such expert was first assembled into a staging buffer")
    // with the Windows call that answers the same question; a Linux build uses mincore.
    bool pages_resident(const uint8_t* p, size_t len) {
#if defined(_WIN32)
        using Qws = BOOL(WINAPI*)(HANDLE, PVOID, DWORD);
        static Qws qws = [] {
            HMODULE k = GetModuleHandleA("kernel32.dll");
            return k ? (Qws)(void*)GetProcAddress(k, "K32QueryWorkingSetEx") : nullptr;
        }();
        if (!qws || len == 0) return false;
        const uintptr_t A = 4096;
        const uintptr_t a0 = (uintptr_t)p & ~(A - 1);
        const uintptr_t a1 = ((uintptr_t)p + len + A - 1) & ~(A - 1);
        const size_t n = (a1 - a0) / A;
        if (n == 0 || n > ((size_t)1 << 21)) return false;
        // P10: probe `page_probe` spread pages instead of the whole range.  The old code sampled 8 and then walked
        // all 4-8 K pages when that sample passed, which at 1.5 ms an expert was most of the token's `pick` phase.
        // A sampled "yes" is only ever a page fault on the pages it did not see, never a wrong byte; a sampled "no"
        // is a missed opportunity (the read is issued, the expert is still correct).
        const size_t want = page_probe > 0 ? (size_t)page_probe : n;
        const size_t probes = want < n ? want : n;
        if (probes < n) {
            std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q(probes);
            for (size_t i = 0; i < probes; ++i)
                q[i].VirtualAddress = (PVOID)(probes == 1 ? a0 : a0 + (A * i * (n - 1)) / (probes - 1));
            if (!qws(GetCurrentProcess(), q.data(), (DWORD)(probes * sizeof(q[0])))) return false;
            for (size_t i = 0; i < probes; ++i)
                if (!q[i].VirtualAttributes.Valid) return false;
            return true;
        }
        std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q(n);
        for (size_t i = 0; i < n; ++i) q[i].VirtualAddress = (PVOID)(a0 + i * A);
        if (!qws(GetCurrentProcess(), q.data(), (DWORD)(n * sizeof(q[0])))) return false;
        for (size_t i = 0; i < n; ++i)
            if (!q[i].VirtualAttributes.Valid) return false;
        return true;
#else
        if (len == 0) return false;
        const long A = sysconf(_SC_PAGESIZE);
        const uintptr_t a0 = (uintptr_t)p & ~((uintptr_t)A - 1);
        const uintptr_t a1 = ((uintptr_t)p + len + (uintptr_t)A - 1) & ~((uintptr_t)A - 1);
        const size_t n = (a1 - a0) / (size_t)A;
        if (n == 0 || n > ((size_t)1 << 21)) return false;
        const size_t want = page_probe > 0 ? (size_t)page_probe : n;
        const size_t probes = want < n ? want : n;
        unsigned char vec = 0;
        for (size_t i = 0; i < probes; ++i) {
            const uintptr_t at = probes == 1 ? a0 : a0 + ((size_t)A * i * (n - 1)) / (probes - 1);
            if (mincore((void*)at, (size_t)A, &vec) != 0) return false;
            if (!(vec & 1)) return false;
        }
        return true;
#endif
    }

    // Queue one expert's three roles into `slot`; fills the three data pointers.
    void ssd_submit_expert(const Layer& L, int64_t expert, int slot, const uint8_t*& g, const uint8_t*& u,
                           const uint8_t*& d, uint64_t tag = 0) {
        g = ssd_submit_role(L.egate, expert, ssd_slot_role(slot, 0), tag);
        u = ssd_submit_role(L.eup, expert, ssd_slot_role(slot, 1), tag);
        d = ssd_submit_role(L.edown, expert, ssd_slot_role(slot, 2), tag);
        ++tok_stats.ssd;
        tok_stats.ssd_bytes += (int64_t)L.expert_bytes;
    }

    // The pointer a role's bytes are read at, without submitting anything (a read queued a layer earlier lands in
    // the same place, and the dot kernels need the address, not the I/O).
    uint8_t* ssd_role_ptr(const Tensor& t, int64_t expert, uint8_t* base) const {
        const size_t len = t.bytes / (size_t)t.ne2;
        const uint64_t off = t.file_offset + (uint64_t)expert * (uint64_t)len;
        const uint64_t A = strata::platform::DirectFile::alignment();
        return base + (size_t)(off - (off & ~(A - 1)));
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
        if (opt.profile_time) {
            if (!w.prof_ev[0]) w.prof_ev[0] = cu::event_create_timed();
            if (!w.prof_ev[1]) w.prof_ev[1] = cu::event_create_timed();
            cu::event_record(w.prof_ev[0], st);
        }
        h2d(w, w.x.p, in, (size_t)n_embd * sizeof(float));
        strata::kernels::quantize_q8_1_rows((const float*)w.x.p, 1, n_embd, w.ex_xq.p, st);
        strata::kernels::minimax_moe_gu(gt, (const void* const*)ptr, (const void* const*)(ptr + top), k, n_embd,
                                        n_ff, L.egate.row_bytes, w.ex_xq.p, (float*)w.ex_h.p, alpha, limit, st, sg,
                                        su);
        strata::kernels::quantize_q8_1_rows((const float*)w.ex_h.p, k, n_ff, w.ex_hq.p, st);
        strata::kernels::minimax_moe_down(dt, (const void* const*)(ptr + 2 * top), k, n_embd, n_ff,
                                          L.edown.row_bytes, w.ex_hq.p, (float*)w.ex_out.p, st, sd);
        if (opt.profile_time && w.prof_ev[1]) cu::event_record(w.prof_ev[1], st);
        return true;
    }

    void gpu_experts_finish(int k, std::vector<float>& out) {
        Work& w = W();
        cu::stream_sync(w.side);
        if (opt.profile_time && w.prof_ev[0] && w.prof_ev[1])
            prof.gpu_stream += cu::event_elapsed_ms(w.prof_ev[0], w.prof_ev[1]);
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
                 std::to_string(dt) + " (supported: F32/F16/BF16, Q8_0, the IQ types and the ik_llama.cpp formats)");
        }
        out.resize((size_t)k * (size_t)n_embd);
        cu::copy_d2h(out.data(), w.ex_out.p, out.size() * sizeof(float));
    }

    // One CPU-lane expert of the current pass: where its three matrices are, and where its [n_embd] result goes.
    // Hoisted out of `moe_tiered` so `mlp_ptr_multi` can take a whole pass at once.
    struct CpuExpertRef {
        int e;
        int j;
        bool ssd;
        const uint8_t* gate;
        const uint8_t* up;
        const uint8_t* down;
        float* out;
    };

    // ---- P10: the shared expert, overlapped with the routed-expert CPU lane.
    //
    // The shared MLP is three dense GEMVs, and the old path ran them with three host round trips between the routed
    // experts and the next layer.  Nothing about it depends on routing: its input is the layer's MoE input, and its
    // result is added after the routed combine.  So the gate/up pair is enqueued on its own stream before the router
    // and the CPU lane, and only the swiglu + the down GEMV are waited for at the end.  Same tensors, same q8_1
    // activation, same native MMVQ kernels and the same host swigluoai as `mlp`: the result is bit-identical.
    bool shared_begin(const Layer& L, const float* in) {
        if (!L.have_shared) return false;
        use_device_index(L.dev);
        Work& w = W();
        const int embd = (int)cfg.n_embd;
        const int ff = (int)L.gate.ne1;
        if (!w.sh_x.p || !w.shs || ff <= 0) return false;
        if (!gemv_dev_in_ok(L.gate, embd) || !gemv_dev_in_ok(L.up, embd) || !gemv_dev_in_ok(L.down, ff))
            return false;
        if ((size_t)ff > (size_t)cfg.n_ff_exp * (size_t)(cfg.n_shared_experts > 0 ? cfg.n_shared_experts : 1))
            return false;
        std::memcpy(w.sh_pin, in, (size_t)embd * sizeof(float));
        cu::copy_h2d_async(w.sh_x.p, w.sh_pin, (size_t)embd * sizeof(float), w.shs);
        strata::kernels::quantize_q8_1_rows((const float*)w.sh_x.p, 1, embd, w.sh_xq.p, w.shs);
        strata::kernels::native_mmvq(L.gate.type, L.gate.dev.p, w.sh_xq.p, (float*)w.sh_g.p, embd, ff, 1, w.shs);
        strata::kernels::native_mmvq(L.up.type, L.up.dev.p, w.sh_xq.p, (float*)w.sh_u.p, embd, ff, 1, w.shs);
        // P17: the SwiGLU and the down projection go on the same stream, so the whole shared MLP hides behind the CPU
        // lane; it used to wait for the lane, then round-trip gate/up through the host for a host SwiGLU (~7 ms a
        // token on rig-2).
        cu::swiglu_rows_f32((const float*)w.sh_g.p, (const float*)w.sh_u.p, (float*)w.sh_h.p, 1, ff, alpha, limit,
                            w.shs);
        strata::kernels::quantize_q8_1_rows((const float*)w.sh_h.p, 1, ff, w.sh_hq.p, w.shs);
        strata::kernels::native_mmvq(L.down.type, L.down.dev.p, w.sh_hq.p, (float*)w.sh_y.p, ff, embd, 1, w.shs);
        cu::copy_d2h_async(w.sh_pin, w.sh_y.p, (size_t)embd * sizeof(float), w.shs);
        sh_pending = true;
        sh_ff = ff;
        return true;
    }

    bool shared_finish(const Layer& L, std::vector<float>& out) {
        (void)L;
        if (!sh_pending) return false;
        sh_pending = false;
        Work& w = W();
        const int embd = (int)cfg.n_embd;
        cu::stream_sync(w.shs);
        out.resize((size_t)embd);
        std::memcpy(out.data(), w.sh_pin, (size_t)embd * sizeof(float));
        return true;
    }

    void moe_tiered(const Layer& L, const float* in, float* yout) {
        // The router and the combine are the CPU path's, unchanged.  Only where each selected expert's bytes come
        // from (VRAM / pinned RAM / SSD) and who computes it change.
        const bool pt = opt.profile_time;
        auto t_ph = std::chrono::steady_clock::now();
        bool shared_async = false;   // P10: set once routing is done, so the shared streams overlap only the lane
        const int nexp = (int)cfg.n_expert;
        const int top = (int)cfg.n_expert_used;
        std::vector<float> logits_r((size_t)nexp), prob((size_t)nexp);
        gemv(L.router, in, logits_r.data());
        for (int i = 0; i < nexp; ++i) prob[(size_t)i] = sigmoid(logits_r[(size_t)i]);
        std::vector<int> order;
        select_routes(L, prob, order, top);
        std::vector<float> w((size_t)top);
        float sum = 0.f;
        for (int j = 0; j < top; ++j) {
            w[(size_t)j] = prob[(size_t)order[(size_t)j]];
            sum += w[(size_t)j];
        }
        if (renorm && sum > 0.f)
            for (float& v : w) v /= sum;
        apply_expert_skip(w);
        if (pt) {
            prof.router += pms(t_ph);
            t_ph = std::chrono::steady_clock::now();
        }
        const int64_t li = (int64_t)(&L - layers.data());
        // Start the prediction for a later layer now, while this layer's experts compute.  `in` is this layer's
        // post-attention FFN input, the predictor Maya's device uses; the target layer's own router scores it.
        if (ssd_prefetch && li + ssd_prefetch_depth < cfg.n_layers) ssd_prefetch_for(li + ssd_prefetch_depth, in);
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
            int j;            // the route it belongs to
            bool ssd;
            const uint8_t* gate;
            const uint8_t* up;
            const uint8_t* down;
        };
        std::vector<CpuExpert> cpu;
        std::vector<SsdAdmit> ssd_admits;
        gpu_e.reserve((size_t)top);
        cpu.reserve((size_t)top);
        int ssd_n = 0;
        int pf_hits = 0;
        const size_t slot_base = ssd_region_base(li);
        const int slot_end = (int)(slot_base + ssd_region_slots());
        ssd_slot_next = (int)(slot_base + ((int)li < (int)ssd_pf_n.size() ? ssd_pf_n[(size_t)li] : 0));
        for (int j = 0; j < top; ++j) {
            if (w[(size_t)j] == 0.f) continue;   // --minimax-expert-skip
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
                if (ram_inplace) {
                    // The page cache is the tier: read the expert where it is in the mapping.
                    cpu.push_back({e, j, false, L.egate.host + (size_t)e * L.expert_gu,
                                   L.eup.host + (size_t)e * L.expert_gu,
                                   L.edown.host + (size_t)e * L.expert_down});
                } else {
                    const uint8_t* p = ram_slot_ptr(const_cast<Layer&>(L), rs);
                    cpu.push_back({e, j, false, p, p + L.expert_gu, p + 2 * L.expert_gu});
                }
                ++tok_stats.ram;
                continue;
            }
            // P6 --minimax-expert-skip-ssd: this route would be read from the SSD.  A low-ranked expert that is
            // already a miss costs a whole blob; leaving it out is the one skip with no cache to fall back on.
            if (ssd_skip_rank > 0 && j >= ssd_skip_rank) {
                w[(size_t)j] = 0.f;
                ++tok_stats.skipped;
                continue;
            }
            // P6: queue the read, don't wait for it.  The bytes land in the slot's final place, and the layer's
            // other experts (VRAM on the side stream, RAM on the pool) are computed while the drive works.
            const uint8_t* g = nullptr;
            const uint8_t* u = nullptr;
            const uint8_t* d = nullptr;
            // P7: the expert's pages may already be resident (a previous token read them through the mapping, or
            // the static RAM fill touched them before they were evicted).  Then the bytes are a memory read: keep
            // them where they are and queue nothing.  Measured hit rate is reported as ssd_mapped.
            if (ssd_zero_copy) {
                const uint8_t* mg = L.egate.host + (size_t)e * L.expert_gu;
                const uint8_t* mu = L.eup.host + (size_t)e * L.expert_gu;
                const uint8_t* md = L.edown.host + (size_t)e * L.expert_down;
                if (pages_resident(mg, L.expert_gu) && pages_resident(mu, L.expert_gu) &&
                    pages_resident(md, L.expert_down)) {
                    cpu.push_back({e, j, false, mg, mu, md});
                    ++tok_stats.ssd_mapped;
                    continue;
                }
            }
            // P7 --minimax-ssd-prefetch: the read for this route was queued a few layers ago.  Match the route
            // against the predicted *set*, not rank-for-rank: a prediction that names the right experts in a
            // slightly different order is still a hit and its slot is still the one holding its bytes.
            int pf_at = -1;
            const size_t pfat_base = (size_t)li * (size_t)top;
            for (int j2 = 0; j2 < top && !ssd_pf_exp.empty(); ++j2)
                if (ssd_pf_exp[pfat_base + (size_t)j2] == e) { pf_at = j2; break; }
            if (pf_at >= 0) {
                const size_t pfat = pfat_base + (size_t)pf_at;
                const int slot = ssd_pf_slot[pfat];
                ssd_pf_exp[pfat] = -1;
                g = ssd_role_ptr(L.egate, e, ssd_slot_role(slot, 0));
                u = ssd_role_ptr(L.eup, e, ssd_slot_role(slot, 1));
                d = ssd_role_ptr(L.edown, e, ssd_slot_role(slot, 2));
                ++ssd_n;
                ++pf_hits;
            } else {
                if (ssd_slot_next >= slot_end)
                    fail("internal: the SSD staging slots are exhausted (more misses than slots in one layer)");
                ssd_submit_expert(L, e, ssd_slot_next++, g, u, d);
                ++ssd_n;
            }
            cpu.push_back({e, j, true, g, u, d});
            // Deferred: admitting (and evicting) now could overwrite a RAM slot another entry of this same
            // token is still going to read.
            ssd_admits.push_back({e, g, u, d});
        }

        // P5b: start the GPU experts on the side stream, compute the CPU ones while it runs, then join and combine.
        if (pt) {
            prof.pick += pms(t_ph);
            t_ph = std::chrono::steady_clock::now();
        }
        std::vector<float> gpu_out;
        bool gpu_async = false;
        if (!gpu_e.empty()) {
            use_device_index(L.dev);
            gpu_async = gpu_experts_begin(L, gpu_e, in);
        }
        if (pt) {
            prof.gpu_begin += pms(t_ph);
            prof.gpu_experts += (int64_t)gpu_e.size();
            t_ph = std::chrono::steady_clock::now();
        }
        // P10: the shared expert's gate/up pair goes on its own stream here: routing is done, the GPU experts are
        // already queued, and the 100+ ms CPU lane is about to run with the card otherwise idle.
        shared_async = shared_begin(L, in);
        std::vector<std::vector<float>> cpu_ye((size_t)top);
        int64_t pass_experts = 0;
        auto run_cpu_experts = [&](bool want_ssd) {
            std::vector<const CpuExpert*> cpu_of((size_t)cfg.n_expert, nullptr);
            for (const CpuExpert& c : cpu)
                if (c.ssd == want_ssd) cpu_of[(size_t)c.e] = &c;
            std::vector<CpuExpertRef> refs;
            refs.reserve((size_t)top);
            for (int j = 0; j < top; ++j) {
                if (w[(size_t)j] == 0.f) continue;
                const int e = order[(size_t)j];
                if (L.vram_of[(size_t)e] >= 0) continue;
                const CpuExpert* c = cpu_of[(size_t)e];
                if (!c) continue;   // an SSD expert that has not landed yet: its pass comes after the wait
                cpu_ye[(size_t)j].resize((size_t)cfg.n_embd);
                refs.push_back({c->e, j, c->ssd, c->gate, c->up, c->down, cpu_ye[(size_t)j].data()});
            }
            // P8: the pass is one call, so its gate/up rows are one parallel region and its down rows a second --
            // two barriers for four experts instead of twelve.
            std::vector<const CpuExpertRef*> jobs;
            jobs.reserve(refs.size());
            for (const CpuExpertRef& r : refs) jobs.push_back(&r);
            mlp_ptr_multi(L, jobs, in);
            pass_experts += (int64_t)refs.size();
        };
        run_cpu_experts(false);
        if (pt) {
            prof.cpu_ram += pms(t_ph);
            prof.ram_experts += pass_experts;
            prof.cpu_experts += pass_experts;
            prof.cpu_bytes += (double)pass_experts * (double)L.expert_bytes;
            pass_experts = 0;
            t_ph = std::chrono::steady_clock::now();
        }
        if (ssd_n > 0) {
            // The drive has had the GPU stream and this pass's RAM experts to hide behind.  A prefetched read
            // carries the same tag as a miss, so this drains it too: a route that used a predicted slot is never
            // read before its bytes are there.
            ssd_wait_all();
            if (pt) {
                prof.ssd_wait += pms(t_ph);
                t_ph = std::chrono::steady_clock::now();
            }
            run_cpu_experts(true);
            if (pt) {
                prof.cpu_ssd += pms(t_ph);
                prof.ssd_experts += pass_experts;
                prof.cpu_experts += pass_experts;
                prof.cpu_bytes += (double)pass_experts * (double)L.expert_bytes;
                pass_experts = 0;
                t_ph = std::chrono::steady_clock::now();
            }
        }
        if (!gpu_e.empty()) {
            if (gpu_async) gpu_experts_finish((int)gpu_e.size(), gpu_out);
            else gpu_experts(L, gpu_e, in, gpu_out);
            if (pt) {
                prof.gpu_finish += pms(t_ph);
                t_ph = std::chrono::steady_clock::now();
            }
        }

        std::fill(yout, yout + cfg.n_embd, 0.f);
        std::vector<float> ye((size_t)cfg.n_embd);
        size_t gi = 0;
        for (int j = 0; j < top; ++j) {
            if (w[(size_t)j] == 0.f) continue;   // --minimax-expert-skip / --minimax-expert-skip-ssd
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
        if (pt) {
            prof.combine += pms(t_ph);
            t_ph = std::chrono::steady_clock::now();
        }
        if (L.have_shared) {
            const auto t_sh = std::chrono::steady_clock::now();
            if (!(shared_async && shared_finish(L, ye)))
                mlp(L.gate, L.up, L.down, 0, in, ye.data(), (int)L.gate.ne1);
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += ye[(size_t)i];
            if (pt) {
                prof.shared += pms(t_sh);
                t_ph = std::chrono::steady_clock::now();
            }
        }
        ram_admit_many(const_cast<Layer&>(L), ssd_admits);
        if (pt) prof.admit += pms(t_ph);
        if (ssd_prefetch && (int)li < (int)ssd_pf_n.size() && ssd_pf_n[(size_t)li] > 0) {
            ssd_pf_hit += pf_hits;
            ssd_pf_waste += ssd_pf_n[(size_t)li] - pf_hits;
            ssd_pf_n[(size_t)li] = 0;
        }
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
    //
    // P7: the same lookahead queues the next layer's *SSD* misses (--minimax-ssd-prefetch).  Nothing can hide a
    // 17 MB read inside the layer that needs it; here it is in flight across the next layer's attention, its
    // router and its dense shared expert, which is what Strata-DS measures as 52.2 -> 49.6 ms/token and Maya as
    // 19.64 -> 20.22 tok/s.  The reads land in the per-route staging slots, and the route's slot is recorded, so a
    // route that the layer then skips (`--minimax-expert-skip`) never mis-maps another route's bytes.
    int prefetch_wait = -1;
    bool ssd_prefetch = false;
    int ssd_prefetch_depth = 1;     // layers between the issue and the use of a predicted read
    std::vector<int> ssd_pf_exp;    // [n_layers * top]: expert whose read is already in flight, or -1
    std::vector<int> ssd_pf_slot;   // [n_layers * top]: the staging slot it landed in
    std::vector<int> ssd_pf_n;      // [n_layers]: slots this layer's prefetch took
    int ssd_slot_next = 0;          // staging slot cursor inside one layer
    int64_t ssd_prefetched = 0;     // reads issued early (counters, not bytes: bytes count once)
    int64_t ssd_pf_hit = 0;         // predicted routes that the layer then really used
    int64_t ssd_pf_waste = 0;       // predicted routes the layer did not use (bytes read for nothing)

    // The VRAM half of the lookahead: one layer ahead, exact router, fill the card's empty slots.
    void prefetch_for(int64_t li) {
        const bool want_vram = prefetch_slots > 0 && tiers_gpu && li >= 0 && li < cfg.n_layers;
        if (!want_vram) return;
        if (li < 0 || li >= cfg.n_layers) return;
        Layer& L = layers[(size_t)li];
        if (!L.moe || !L.router.host) return;
        const bool vram_part = want_vram && !L.vram.empty();
        if (!vram_part) return;
        Work* wv = nullptr;
        if (vram_part) {
            use_layer_device(li);
            Work& w = W();
            if (!w.side) w.side = cu::stream_create();
            wv = &w;
        }
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        std::vector<float> logits((size_t)nexp), prob((size_t)nexp);
        // The router reads the post-FFN-norm input, not the raw residual (moe() applies ffn_norm first).
        std::vector<float> hnorm((size_t)cfg.n_embd);
        rms(x.data(), L.ffn_norm.data(), hnorm.data(), (int)cfg.n_embd, eps);
        gemv(L.router, hnorm.data(), logits.data());
        for (int i = 0; i < nexp; ++i) prob[(size_t)i] = sigmoid(logits[(size_t)i]);
        std::vector<int> order;
        select_routes(L, prob, order, top, false);
        int started = 0;
        for (int j = 0; j < top && started < prefetch_slots && wv; ++j) {
            const int e = order[(size_t)j];
            if (L.vram_of[(size_t)e] >= 0) continue;
            int free_slot = -1;
            for (int sl = 0; sl < (int)L.vram.size(); ++sl)
                if (L.vram[(size_t)sl].expert < 0) { free_slot = sl; break; }
            if (free_slot < 0) break;
            Layer::VramSlot& slot = L.vram[(size_t)free_slot];
            uint8_t* dst = (uint8_t*)slot.mem.p;
            cu::copy_h2d_async(dst, L.egate.host + (size_t)e * L.expert_gu, L.expert_gu, wv->side);
            cu::copy_h2d_async(dst + L.expert_gu, L.eup.host + (size_t)e * L.expert_gu, L.expert_gu, wv->side);
            cu::copy_h2d_async(dst + 2 * L.expert_gu, L.edown.host + (size_t)e * L.expert_down, L.expert_down,
                               wv->side);
            slot.expert = e;
            L.vram_of[(size_t)e] = free_slot;
            ++started;
        }
        if (started) prefetch_wait = (int)li;
    }

    // The SSD half: apply layer `li`'s router to the *post-attention, post-norm input* of the layer that is about
    // to run (`hnorm`), which is what Project Maya's device does ("the device guesses the next layer's picks from
    // the current layer's FFN input", ds4_model.cu:1173-1190).  It is called from inside layer `j`'s MoE for
    // `li = j + depth`, so the bytes are in flight across `depth` layers of expert compute and attention instead of
    // one attention block.  The router input is a prediction: the real layer `li` sees a different attention
    // output.  It only ever decides *which bytes to read early*; a route the prediction missed is read
    // synchronously exactly as it was with prefetch off, so the result cannot change.
    void ssd_prefetch_for(int64_t li, const float* hnorm) {
        if (!ssd_prefetch || !ssd_slots || !hnorm) return;
        if (li < 0 || li >= cfg.n_layers) return;
        if ((int)ssd_pf_exp.size() < (int)cfg.n_layers * (int)cfg.n_expert_used) return;
        Layer& L = layers[(size_t)li];
        if (!L.moe || !L.router.host) return;
        // This layer's router may be on the other card (--layer-split): the prediction is scored on the target
        // layer's device, not on the device of the layer whose MoE is running.  Without this, a 2-GPU split faults
        // with "gemv sync: an illegal memory access" on the first token.
        use_layer_device(li);
        // Drain every read before writing into the region.  A region can hold a read that no route ever consumed,
        // and reusing those bytes while the old read is in flight is a data race.  Draining here is a whole-layer
        // barrier, so the effective lookahead is one layer, but it is unconditional, cannot deadlock on a tag
        // that a shard never reports, and keeps the prediction safe.  (A per-region drain was tried: it deadlocked
        // after the full model's prefill when a prediction's completion was never reaped on its shard.)
        ssd_wait_all();
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        std::vector<float> logits((size_t)nexp), prob((size_t)nexp);
        gemv(L.router, hnorm, logits.data());
        for (int i = 0; i < nexp; ++i) prob[(size_t)i] = sigmoid(logits[(size_t)i]);
        std::vector<int> order;
        select_routes(L, prob, order, top, false);
        const size_t region = ssd_region_base(li);
        const size_t at = (size_t)li * (size_t)top;
        for (int j = 0; j < top; ++j) ssd_pf_exp[at + (size_t)j] = -1;
        int k = 0;
        for (int j = 0; j < top; ++j) {
            const int e = order[(size_t)j];
            if (L.vram_of[(size_t)e] >= 0) continue;
            const int rs = L.ram_of[(size_t)e];
            if (rs >= 0) continue;
            if (ssd_skip_rank > 0 && j >= ssd_skip_rank) continue;
            if (k >= top) break;
            const int slot = (int)region + k;
            const uint8_t* g = nullptr;
            const uint8_t* u = nullptr;
            const uint8_t* d = nullptr;
            ssd_submit_expert(L, e, slot, g, u, d);
            ssd_pf_exp[at + (size_t)j] = e;
            ssd_pf_slot[at + (size_t)j] = slot;
            ++k;
            ++ssd_prefetched;
        }
        ssd_pf_n[(size_t)li] = k;
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

    void gemv(const Tensor& t, const float* xin, float* yout, void* dev_out = nullptr) {
        if (t.ne2 != 1) fail(t.name + ": internal gemv is for a matrix, not a stacked expert");
        const int rows = (int)t.ne1;
        const int cols = (int)t.ne0;
        if ((int64_t)cols > max_cols || (int64_t)rows > max_rows) fail(t.name + ": gemv exceeds the workspace");
        if (t.gpu) {
            Work& w = W();
            // P5c: with `dev_out` the result stays on the device (the fused norm+rope step consumes it there).
            // The kernels then stay on the DEFAULT stream, so the norm+rope launch that follows is ordered
            // after them without a host round trip.
            if (dense_mmvq && !t.dev_f32 && mmvq_dense_fits(t.type, cols) && w.xq.p) {
                if (!dev_out && !w.ds) w.ds = cu::stream_create();
                if (dev_out && !w.dsb) w.dsb = cu::stream_create_blocking();
                void* ds = dev_out ? w.dsb : w.ds;
                h2d(w, w.x.p, xin, (size_t)cols * sizeof(float));
                strata::kernels::quantize_q8_1_rows((const float*)w.x.p, 1, cols, w.xq.p, ds);
                strata::kernels::native_mmvq(t.type, t.dev.p, w.xq.p, (float*)(dev_out ? dev_out : w.y.p), cols, rows,
                                             1, ds);
                if (dev_out) return;
                cu::copy_d2h_async(yout, w.y.p, (size_t)rows * sizeof(float), w.ds);
                cu::stream_sync(w.ds);
                return;
            }
            h2d(w, w.x.p, xin, (size_t)cols * sizeof(float));
            float* dy = (float*)(dev_out ? dev_out : w.y.p);
            if (t.dev_f32) {
                cu::gemv((const float*)t.dev.p, (const float*)w.x.p, dy, rows, cols);
            } else {
                for (int r = 0; r < rows; r += kTile) {
                    const int n = rows - r < kTile ? rows - r : kTile;
                    cu::dequant_f32(t.type, t.dev.p, r, n, cols, (float*)w.scratch.p);
                    cu::gemv((const float*)w.scratch.p, (const float*)w.x.p, dy + r, n, cols);
                }
            }
            if (dev_out) return;
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

    // P10: the same dense GEMV when the activation is already on the device (the attention output the MSA kernel just
    // wrote there).  Byte-identical to gemv(): the same q8_1 quantizer and the same native MMVQ kernel, on a stream
    // that is ordered after the default-stream kernels, with no D2H of the activation and no H2D back.
    bool gemv_dev_in_ok(const Tensor& t, int cols) const {
        return t.gpu && dense_mmvq && !t.dev_f32 && mmvq_dense_fits(t.type, cols);
    }

    void gemv_dev_in(const Tensor& t, const void* din, float* yout) {
        const int rows = (int)t.ne1;
        const int cols = (int)t.ne0;
        if (!gemv_dev_in_ok(t, cols)) {
            std::vector<float> host((size_t)cols);
            cu::copy_d2h(host.data(), din, (size_t)cols * sizeof(float));
            gemv(t, host.data(), yout);
            return;
        }
        Work& w = W();
        // The MSA kernel that wrote `din` is on the default stream, so this chain stays there too: the legacy
        // default stream's handle keeps native MMVQ (which rejects nullptr) ordered with it, and the synchronous
        // copy-out is the only wait.
        void* st = cu::stream_legacy();
        strata::kernels::quantize_q8_1_rows((const float*)din, 1, cols, w.xq.p, st);
        strata::kernels::native_mmvq(t.type, t.dev.p, w.xq.p, (float*)w.y.p, cols, rows, 1, st);
        cu::copy_d2h(yout, w.y.p, (size_t)rows * sizeof(float));
    }

    void dot_rows_base(const Tensor& t, const uint8_t* base, const float* xin, float* yout, int rows_n, int cols) {
        // One expert's rows start at `base`: yout[r] = row(r) . xin.  `base` may be the GGUF mapping, the pinned
        // RAM arena, or an SSD staging buffer; only t's type, row size and row count are used.
        if (strata::kernels::ikq::is_iqk_type(t.type)) {
            // The ported formats dot against the same q8_K activation as everything else; the weight type is
            // unknown to ggml, but the activation type is not, so its own quantizer does the work.
            if ((cols % 256) != 0) {
                fail(t.name + ": " + strata::kernels::ikq::type_name(t.type) +
                     " needs a width that is a whole number of 256-value blocks");
            }
            const ggml_type_traits_cpu* act = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
            if (!act || !act->from_float) fail(t.name + ": no from_float for the q8_K activation");
            std::vector<uint8_t> aq(ggml_row_size(GGML_TYPE_Q8_K, cols));
            act->from_float(xin, aq.data(), cols);
            pool.run(rows_n, [&](int a, int b) {
                for (int r = a; r < b; ++r)
                    yout[r] = strata::kernels::ikq::dot_row(t.type, base + (size_t) r * t.row_bytes, aq.data(),
                                                           cols);
            });
            return;
        }
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

    // ---- P8: the CPU lane's row kernel, split out of `dot_rows_base` so that a whole layer's experts can be
    // dispatched in two parallel regions instead of three regions per expert.
    //
    // `mlp_ptr` runs `dot_rows_base` for gate, up and down, and each of those is its own `pool.run`: with four routed
    // experts on the CPU lane that is twelve barriers a layer, and with the pool's spin window at its default 0 every
    // barrier after the first wakes eleven sleeping threads.  The experts of one layer share one activation, so the
    // gate and up rows of all of them are one region and the down rows a second.  Nothing about the arithmetic
    // changes: same rows, same kernel, same activation bytes, so the logits are bit-identical.
    struct RowKernel {
        int type = -1;
        bool iqk = false;
        int cols = 0;
        const ggml_type_traits_cpu* act = nullptr;   // the activation format this row kernel dots against
        ggml_vec_dot_t vec = nullptr;                // ggml's row kernel (iqk formats carry their own)
        size_t act_bytes = 0;
        bool raw = false;                            // F32/F16/BF16: no packed activation
    };

    RowKernel row_kernel(const Tensor& t, int cols) {
        RowKernel k;
        k.type = t.type;
        k.cols = cols;
        if (t.type == kF32 || t.type == kF16 || t.type == kBF16) {
            k.raw = true;
            return k;
        }
        if (strata::kernels::ikq::is_iqk_type(t.type)) {
            if ((cols % 256) != 0)
                fail(t.name + ": " + strata::kernels::ikq::type_name(t.type) +
                     " needs a width that is a whole number of 256-value blocks");
            k.iqk = true;
            k.act = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        } else {
            if (!t.cpu || !t.cpu->vec_dot)
                fail(t.name + ": ggml-cpu has no vec_dot for type " + std::to_string(t.type));
            k.vec = t.cpu->vec_dot;
            k.act = ggml_get_type_traits_cpu(t.cpu->vec_dot_type);
        }
        if (!k.act || !k.act->from_float) fail(t.name + ": no from_float for the activation type");
        k.act_bytes = ggml_row_size(k.iqk ? GGML_TYPE_Q8_K : t.cpu->vec_dot_type, cols);
        return k;
    }

    inline float row_dot(const RowKernel& k, const uint8_t* row, const uint8_t* aq) {
        if (k.iqk) return strata::kernels::ikq::dot_row(k.type, row, aq, k.cols);
        float s = 0.f;
        k.vec(k.cols, &s, 0, row, 0, aq, 0, 1);
        return s;
    }

    // Everything the CPU lane needs to run a layer's experts, reused across layers and tokens so a layer allocates
    // nothing: gate/up/hidden tiles for every routed expert on the CPU, and one activation image per matrix.
    std::vector<float> cpu_gbuf, cpu_ubuf, cpu_hbuf;
    std::vector<uint8_t> cpu_aq, cpu_hq;
    // STRATA_LANE_PF=N: software-prefetch N rows ahead in the lane's row loops (0 = off, a measured probe, not a
    // default: every thread walks its rows sequentially and the hardware prefetcher already sees that stride).
    int lane_pf = 0;
    // P14: rows per chunk in the CPU lane's two regions (~100 KB of IQ2 rows, ~25 chunks per thread a layer).
    static constexpr int kLaneChunkRows = 64;
    static inline void prefetch_row(const void* p) {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
        _mm_prefetch((const char*)p, _MM_HINT_T0);
#else
        (void)p;
#endif
    }

    // One layer's CPU experts, two parallel regions.  `jobs` are (expert, gate/up/down bytes, out[n_embd]); the
    // per-expert NVFP4 divisors are read from the layer, as `mlp_ptr` does.
    void mlp_ptr_multi(const Layer& L, const std::vector<const CpuExpertRef*>& jobs, const float* in) {
        const int n = (int)jobs.size();
        if (n == 0) return;
        const int ff = (int)cfg.n_ff_exp;
        const int embd = (int)cfg.n_embd;
        const RowKernel kg = row_kernel(L.egate, embd);
        const RowKernel ku = row_kernel(L.eup, embd);
        const RowKernel kd = row_kernel(L.edown, ff);
        if (kg.raw || ku.raw || kd.raw) {
            // F32/F16/BF16 experts: the plain per-matrix path (this is the synthetic test fixture, not the pack).
            for (const CpuExpertRef* c : jobs)
                mlp_ptr(L.egate, c->gate, L.eup, c->up, L.edown, c->down, in, c->out, ff,
                        L.egate_scale.empty() ? nullptr : &L.egate_scale[(size_t)c->e],
                        L.eup_scale.empty() ? nullptr : &L.eup_scale[(size_t)c->e],
                        L.edown_scale.empty() ? nullptr : &L.edown_scale[(size_t)c->e]);
            return;
        }
        cpu_gbuf.resize((size_t)n * (size_t)ff);
        cpu_ubuf.resize((size_t)n * (size_t)ff);
        cpu_hbuf.resize((size_t)n * (size_t)ff);
        cpu_aq.resize(kg.act_bytes);
        kg.act->from_float(in, cpu_aq.data(), embd);
        const size_t g_row = L.egate.row_bytes, u_row = L.eup.row_bytes;
        // Region 1: every gate row and every up row of the pass, one contiguous run of rows per thread.
        if ((int)L.egate.ne1 != ff || (int)L.eup.ne1 != ff)
            fail("gate/up rows do not match the expert feed-forward width");
        if (ku.act_bytes != kg.act_bytes) fail("gate and up need different activation widths");
        // P14: rows go out in chunks (run_dyn), so a late thread no longer holds the barrier; rows are independent.
        pool.run_dyn(n * 2 * ff, kLaneChunkRows, [&](int a, int b) {
            for (int i = a; i < b; ++i) {
                const int j = i / (2 * ff);
                const int k = i - j * 2 * ff;
                const bool is_up = k >= ff;
                const int r = is_up ? k - ff : k;
                const CpuExpertRef* c = jobs[(size_t)j];
                const size_t stride = is_up ? u_row : g_row;
                const uint8_t* row = (is_up ? c->up : c->gate) + (size_t)r * stride;
                if (lane_pf > 0) prefetch_row(row + stride * (size_t)lane_pf);
                const RowKernel& kk = is_up ? ku : kg;
                const float s = row_dot(kk, row, cpu_aq.data());
                (is_up ? cpu_ubuf : cpu_gbuf)[(size_t)j * (size_t)ff + (size_t)r] = s;
            }
        });
        // swigluoai and the divisors: n * ff values, ~12 K multiply-adds per expert, cheaper than a barrier.
        for (int j = 0; j < n; ++j) {
            const float* g = &cpu_gbuf[(size_t)j * (size_t)ff];
            const float* u = &cpu_ubuf[(size_t)j * (size_t)ff];
            const float fg = L.egate_scale.empty() ? 1.f : L.egate_scale[(size_t)jobs[(size_t)j]->e];
            const float fu = L.eup_scale.empty() ? 1.f : L.eup_scale[(size_t)jobs[(size_t)j]->e];
            float* h = &cpu_hbuf[(size_t)j * (size_t)ff];
            for (int i = 0; i < ff; ++i) h[i] = swigluoai(g[i] * fg, u[i] * fu, alpha, limit);
        }
        // Region 2: every down row.  Each expert's hidden state needs its own activation image; the quantizer is
        // cheap (ff values) and the rows are 6 K long.
        cpu_hq.resize((size_t)n * kd.act_bytes);
        for (int j = 0; j < n; ++j)
            kd.act->from_float(&cpu_hbuf[(size_t)j * (size_t)ff], cpu_hq.data() + (size_t)j * kd.act_bytes, ff);
        const size_t d_row = L.edown.row_bytes;
        pool.run_dyn(n * embd, kLaneChunkRows, [&](int a, int b) {
            for (int i = a; i < b; ++i) {
                const int j = i / embd;
                const int r = i - j * embd;
                const CpuExpertRef* c = jobs[(size_t)j];
                const uint8_t* row = c->down + (size_t)r * d_row;
                if (lane_pf > 0) prefetch_row(row + d_row * (size_t)lane_pf);
                c->out[r] = row_dot(kd, row, cpu_hq.data() + (size_t)j * kd.act_bytes);
            }
        });
        for (int j = 0; j < n; ++j) {
            if (L.edown_scale.empty()) continue;
            const float sd = L.edown_scale[(size_t)jobs[(size_t)j]->e];
            float* o = jobs[(size_t)j]->out;
            for (int i = 0; i < embd; ++i) o[i] *= sd;
        }
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
        std::vector<float> logits_r((size_t)nexp), prob((size_t)nexp);
        gemv(L.router, in, logits_r.data());
        for (int i = 0; i < nexp; ++i) prob[(size_t)i] = sigmoid(logits_r[(size_t)i]);
        std::vector<int> order;
        select_routes(L, prob, order, top);
        std::vector<float> w((size_t)top);
        float sum = 0.f;
        for (int j = 0; j < top; ++j) {
            w[(size_t)j] = prob[(size_t)order[(size_t)j]];
            sum += w[(size_t)j];
        }
        if (renorm && sum > 0.f)
            for (float& v : w) v /= sum;
        apply_expert_skip(w);
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
            if (w[(size_t)j] == 0.f) continue;   // --minimax-expert-skip
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

    // ---- P8: the q8_0 block codec.  Every value of a 32-wide block shares one fp16 scale, exactly as ggml's q8_0:
    // d = amax / 127, q_i = round(x_i / d) clamped to int8.  A head is 128 values = 4 blocks, so a head row is
    // self-contained and `--minimax-kv vram`'s row layout would be the same if the device tier ever takes it.
    static constexpr int kQ8Block = 32;
    static constexpr int kQ8Bytes = 34;
    static void quant_q8_block(const float* src, uint8_t* dst) {
        float amax = 0.f;
        for (int i = 0; i < kQ8Block; ++i) amax = std::fmax(amax, std::fabs(src[i]));
        const float d = amax / 127.f;
        const uint16_t dh = ggml_fp32_to_fp16(d);
        std::memcpy(dst, &dh, 2);
        // Quantize against the scale that was actually stored, so dequantize(quantize(x)) == d16 * q exactly.
        const float dq = ggml_fp16_to_fp32(dh);
        const float inv = dq > 0.f ? 1.f / dq : 0.f;
        for (int i = 0; i < kQ8Block; ++i) {
            float v = src[i] * inv;
            v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
            ((int8_t*)dst)[2 + i] = (int8_t)std::lrintf(v);
        }
    }
    static float dot_q8(const uint8_t* row, const float* q, int n) {
        float s = 0.f;
        for (int b = 0; b < n / kQ8Block; ++b) {
            const uint8_t* p = row + (size_t)b * kQ8Bytes;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const int8_t* qs = (const int8_t*)(p + 2);
            float acc = 0.f;
            for (int i = 0; i < kQ8Block; ++i) acc += (float)qs[i] * q[(size_t)b * kQ8Block + (size_t)i];
            s += ggml_fp16_to_fp32(dh) * acc;
        }
        return s;
    }
    static void deq_q8(const uint8_t* row, float* dst, int n) {
        for (int b = 0; b < n / kQ8Block; ++b) {
            const uint8_t* p = row + (size_t)b * kQ8Bytes;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const int8_t* qs = (const int8_t*)(p + 2);
            const float d = ggml_fp16_to_fp32(dh);
            for (int i = 0; i < kQ8Block; ++i) dst[(size_t)b * kQ8Block + (size_t)i] = d * (float)qs[i];
        }
    }
    // Byte offset of the q8 block that starts at value offset `off` (always a head boundary here).
    size_t q8_off(size_t off) const {
        const size_t klen = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        const size_t tok = off / klen;
        const size_t rest = off % klen;
        return tok * kv_q_row + (rest / (size_t)cfg.head_dim) * (size_t)(cfg.head_dim / kQ8Block) * (size_t)kQ8Bytes;
    }
    // Store one token's K or V (n = n_head_kv * head_dim values) into whichever layout is active.  `off` is the
    // value offset, so both layouts address the same token the same way.
    void store_kv(uint8_t* qbase, uint16_t* base, size_t off, const float* src, int n) {
        if (kv_q) {
            uint8_t* dst = qbase + q8_off(off);
            for (int i = 0; i < n; i += kQ8Block) quant_q8_block(src + i, dst + (size_t)(i / kQ8Block) * kQ8Bytes);
            return;
        }
        store_f16(base, off, src, n);
    }
    // One head's `n` values, as a dot against a fp32 query, and as fp32 values.
    float dot_head(uint8_t* qbase, uint16_t* base, size_t off, int n, const float* q) const {
        if (kv_q) return dot_q8(qbase + q8_off(off), q, n);
        return dot_f16_cache(base + off, q, n);
    }
    void load_head(uint8_t* qbase, uint16_t* base, size_t off, int n, float* dst) const {
        if (kv_q) {
            deq_q8(qbase + q8_off(off), dst, n);
            return;
        }
        for (int i = 0; i < n; ++i) dst[i] = ggml_fp16_to_fp32(base[off + (size_t)i]);
    }
    // The same store, from fp16 input (the batched path's device kernels produce fp16 K/V).
    void store_kv16(uint8_t* qbase, uint16_t* base, size_t off, const uint16_t* src, int n) {
        if (!kv_q) {
            std::memcpy(base + off, src, (size_t)n * sizeof(uint16_t));
            return;
        }
        if (kv_f32.size() < (size_t)n) kv_f32.assign((size_t)n, 0.f);
        for (int i = 0; i < n; ++i) kv_f32[(size_t)i] = ggml_fp16_to_fp32(src[i]);
        store_kv(qbase, base, off, kv_f32.data(), n);
    }

    // The MSA selection, shared by the host and device attention paths: local blocks are forced in, the rest are
    // ranked by their max-pooled indexer score, and the selected blocks expand to token positions.  The scratch is
    // reused (P8: this used to allocate five vectors per token per layer).
    void select_slots(int64_t li, int64_t pos, const float* bs_in, int nb, std::vector<int32_t>& slots,
                      std::vector<int>& off) {
        const int HI = (int)cfg.idx_heads;
        const int BLK = (int)cfg.idx_block;
        ah_bs2.assign(bs_in, bs_in + (size_t)HI * nb);
        std::vector<float>& bs = ah_bs2;
        for (int h = 0; h < HI; ++h) {
            const int qb = (int)(pos / BLK);
            for (int l = 0; l < (int)cfg.idx_local; ++l) {
                int bi = qb - l;
                if (bi < 0) bi = 0;
                if (bi < nb) bs[(size_t)h * nb + bi] = std::numeric_limits<float>::infinity();
            }
        }
        const int kk = (int)cfg.idx_top_k < nb ? (int)cfg.idx_top_k : nb;
        ah_sel.assign((size_t)HI * (size_t)nb, 0);
        std::vector<char>& sel = ah_sel;
        ah_order.resize((size_t)nb);
        std::vector<int>& order = ah_order;
        ah_ids.assign((size_t)HI * (size_t)cfg.idx_top_k, -1);   // probe dump, -1 padded
        std::vector<int64_t>& ids = ah_ids;
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
        // P7: --minimax-kv vram keeps no host K/V at all.  Calling the host path under it used to read a null
        // pointer and take an access violation tens of seconds into a batched prefill; name the combination
        // instead of dying in it.
        if ((!k_host && !k_host_q) || (!v_host && !v_host_q))
            fail("host attention was asked to run but the K/V cache is not in host memory (--minimax-kv vram "
                 "keeps it on the device). Use --minimax-kv ram or host, or put at least one dense layer on the "
                 "GPU so the batched device attention path can run.");
        const int H = (int)cfg.n_head;
        const int HKV = (int)cfg.n_head_kv;
        const int D = (int)cfg.head_dim;
        const int ID = (int)cfg.idx_dim;
        if (L.indexer && !ik_host)
            fail("host attention needs the indexer keys in host memory, and this placement keeps them on the card "
                 "only (the pinned-RAM KV tier). The batched prompt pass stages one layer's K/V on the device "
                 "instead; do not turn that off (--minimax-prefill-kv auto), or use --minimax-kv host.");
        const int64_t Tk = pos + 1;
        const int q_per_kv = H / HKV;
        const float scale = 1.f / std::sqrt((float)D);
        const size_t klen = (size_t)HKV * (size_t)D;
        const size_t lbase = (size_t)layer * (size_t)ctx * klen;   // value offset of this layer's row 0
        const bool fast = host_attn_fast;
        const bool use_q = kv_q != 0;
        if (fast && pool.nthreads > 1) {
            const size_t need = (size_t)pool.nthreads * (size_t)max_slot;
            if (ah_prob.size() < need) ah_prob.assign(need, 0.f);
            if (use_q && ah_prob_d.size() < need * (size_t)D) ah_prob_d.assign(need * (size_t)D, 0.f);
        }
        int nb = 0;
        if (L.indexer) {
            const int HI = (int)cfg.idx_heads;
            const int BLK = (int)cfg.idx_block;
            nb = (int)((Tk + BLK - 1) / BLK);
            ah_bs.assign((size_t)HI * nb, -std::numeric_limits<float>::infinity());
            if (fast) {
                // P8: one job per (indexer head, block).  The max-pool order inside a block is unchanged, so this
                // is bit-identical to the serial loop it replaces; at 32 K keys it is the difference between
                // ~1 GMAC on one thread and the same work spread over the pool.
                pool.run(HI * nb, [&](int a, int b) {
                    for (int r = a; r < b; ++r) {
                        const int h = r / nb;
                        const int bl = r % nb;
                        const float* qh = iq.data() + (size_t)h * ID;
                        const int64_t begin = (int64_t)bl * BLK;
                        int64_t end = begin + BLK < Tk ? begin + BLK : Tk;
                        float m = -std::numeric_limits<float>::infinity();
                        for (int64_t t = begin; t < end; ++t) {
                            if (t > pos) break;
                            const float sc = dot_f16_cache(ik_at(layer, t), qh, ID);
                            if (sc > m) m = sc;
                        }
                        ah_bs[(size_t)r] = m;
                    }
                });
            } else {
                for (int h = 0; h < HI; ++h) {
                    const float* qh = iq.data() + (size_t)h * ID;
                    for (int b = 0; b < nb; ++b) {
                        const int64_t begin = (int64_t)b * BLK;
                        const int64_t end = begin + BLK < Tk ? begin + BLK : Tk;
                        float m = -std::numeric_limits<float>::infinity();
                        for (int64_t t = begin; t < end; ++t) {
                            if (t > pos) break;
                            const float sc = dot_f16_cache(ik_at(layer, t), qh, ID);
                            if (sc > m) m = sc;
                        }
                        ah_bs[(size_t)h * nb + b] = m;
                    }
                }
            }
            select_slots(layer, pos, ah_bs.data(), nb, ah_slots, ah_off);
        }
        const int q_per_idx = L.indexer ? H / (int)cfg.idx_heads : H;
        // One query head is independent of every other, so splitting them across the pool keeps the arithmetic
        // identical: the same dots, in the same order, over the same slots.
        auto one_head = [&](int qh, float* prob, float* vb) {
            const int kv = qh / q_per_kv;
            const float* qrow = q.data() + (size_t)qh * D;
            float* out = att + (size_t)qh * D;
            for (int d = 0; d < D; ++d) out[d] = 0.f;
            if (!L.indexer) {
                for (int64_t t = 0; t < Tk; ++t) {
                    if (t > pos) prob[(size_t)t] = -std::numeric_limits<float>::infinity();
                    else prob[(size_t)t] = dot_head(k_host_q, k_host, lbase + (size_t)t * klen + (size_t)kv * D, D, qrow) * scale;
                }
                softmax_inplace(prob, (int)Tk);
                for (int64_t t = 0; t < Tk; ++t) {
                    const float p = prob[(size_t)t];
                    if (p == 0.f) continue;
                    if (use_q) {
                        load_head(v_host_q, v_host, lbase + (size_t)t * klen + (size_t)kv * D, D, vb);
                        for (int d = 0; d < D; ++d) out[d] += p * vb[d];
                    } else {
                        const uint16_t* vv = v_host + lbase + (size_t)t * klen + (size_t)kv * D;
                        for (int d = 0; d < D; ++d) out[d] += p * ggml_fp16_to_fp32(vv[d]);
                    }
                }
                return;
            }
            const int ih = qh / q_per_idx;
            const int a = ah_off[(size_t)ih];
            const int b = ah_off[(size_t)ih + 1];
            const int nslot = b - a;
            if (nslot <= 0) return;
            for (int i = 0; i < nslot; ++i) {
                const int32_t t = ah_slots[(size_t)(a + i)];
                prob[(size_t)i] = dot_head(k_host_q, k_host, lbase + (size_t)t * klen + (size_t)kv * D, D, qrow) * scale;
            }
            softmax_inplace(prob, nslot);
            for (int i = 0; i < nslot; ++i) {
                const float p = prob[(size_t)i];
                if (p == 0.f) continue;
                const int32_t t = ah_slots[(size_t)(a + i)];
                if (use_q) {
                    load_head(v_host_q, v_host, lbase + (size_t)t * klen + (size_t)kv * D, D, vb);
                    for (int d = 0; d < D; ++d) out[d] += p * vb[d];
                } else {
                    const uint16_t* vv = v_host + lbase + (size_t)t * klen + (size_t)kv * D;
                    for (int d = 0; d < D; ++d) out[d] += p * ggml_fp16_to_fp32(vv[d]);
                }
            }
        };
        // The parallel path needs `nslot <= max_slot`; that is guaranteed by the indexer (top_k x block) and not by
        // the dense fallback, whose row is as long as the context.
        if (fast && pool.nthreads > 1 && L.indexer && ah_prob.size() >= (size_t)max_slot) {
            pool.run(H, [&](int a, int b) {
                float* prob = ah_prob.data() + (size_t)tls_part * (size_t)max_slot;
                float* vb = use_q ? ah_prob_d.data() + (size_t)tls_part * (size_t)max_slot * (size_t)D : nullptr;
                for (int qh = a; qh < b; ++qh) one_head(qh, prob, vb);
            }, true);
        } else {
            const size_t pn = (size_t)(Tk > max_slot ? Tk : max_slot);
            if (ah_prob_s.size() < pn) ah_prob_s.assign(pn, 0.f);
            if (use_q && ah_prob_ds.size() < (size_t)D) ah_prob_ds.assign((size_t)D, 0.f);
            for (int qh = 0; qh < H; ++qh) one_head(qh, ah_prob_s.data(), use_q ? ah_prob_ds.data() : nullptr);
        }
    }

    // P4: the same attention on the owning device.  The indexer scores are computed from the device copy of the
    // indexer keys; only the selected slots of each KV head are brought into the staging buffer (a device gather
    // when the KV lives in VRAM, one H2D copy from pinned host memory in the RAM tier).
    //
    // P10: with the KV in VRAM the whole selection stays on the card (`idx_select`), so neither the block scores nor
    // the slot lists cross PCIe and there is no device sync between the indexer and the MSA kernel.  `att_dev_out`
    // keeps the attention output on the card for the output projection, removing the D2H+H2D pair around it.
    void attend_dev(int layer, const Layer& L, int64_t pos, float* att, const float* qdev = nullptr,
                    const float* iqdev = nullptr, float* att_dev_out = nullptr) {
        use_device_index(L.dev);
        Work& w = W();
        const int H = (int)cfg.n_head;
        const int HKV = (int)cfg.n_head_kv;
        const int D = (int)cfg.head_dim;
        const int HI = (int)cfg.idx_heads;
        const int BLK = (int)cfg.idx_block;
        const int64_t Tk = pos + 1;
        const int nb = (int)((Tk + BLK - 1) / BLK);
        // The device selection writes the slot lists the gather consumes; the dump needs the host arrays, and the
        // pinned-RAM tier needs the counts on the host to know how many rows to copy.
        const bool dev_slots = kv_mode == 1 && L.indexer && !dumping(layer) && w.slots.p && w.counts.p;
        std::vector<float> bs;
        const auto t_idx = std::chrono::steady_clock::now();
        if (L.indexer) {
            if (dev_slots) {
                if (iqdev) {
                    cu::idx_block_scores(L.d_ik.p, Tk, iqdev, HI, (int)cfg.idx_dim, BLK, nb, (float*)w.bs.p, nullptr);
                } else {
                    h2d(w, w.qd.p, iq.data(), (size_t)HI * (size_t)cfg.idx_dim * sizeof(float));
                    cu::idx_block_scores(L.d_ik.p, Tk, (const float*)w.qd.p, HI, (int)cfg.idx_dim, BLK, nb,
                                         (float*)w.bs.p, nullptr);
                }
                cu::idx_select((const float*)w.bs.p, HI, nb, (int)cfg.idx_top_k, BLK, pos, (int)cfg.idx_local,
                               (int)max_slot, (int32_t*)w.slots.p, (int32_t*)w.counts.p, nullptr);
            } else {
                bs.assign((size_t)HI * nb, -std::numeric_limits<float>::infinity());
                if (iqdev) {
                    cu::idx_block_scores(L.d_ik.p, Tk, iqdev, HI, (int)cfg.idx_dim, BLK, nb, (float*)w.bs.p, nullptr);
                } else {
                    h2d(w, w.qd.p, iq.data(), (size_t)HI * (size_t)cfg.idx_dim * sizeof(float));
                    cu::idx_block_scores(L.d_ik.p, Tk, (const float*)w.qd.p, HI, (int)cfg.idx_dim, BLK, nb,
                                         (float*)w.bs.p, nullptr);
                }
                cu::sync();
                cu::copy_d2h(bs.data(), w.bs.p, (size_t)HI * (size_t)nb * sizeof(float));
            }
        }
        if (opt.profile_time) prof.attn_idx_sync += pms(t_idx);
        std::vector<int32_t> slots;
        std::vector<int> off;
        std::vector<int32_t> sp;
        std::vector<int32_t> cnt;
        const int q_per_kv = H / HKV;
        if (!dev_slots) {
            select_slots(layer, pos, bs.data(), nb, slots, off);

            // Per KV head: the slot list of the indexer head that feeds it (HI == HKV is enforced at startup).
            sp.assign((size_t)HKV * (size_t)max_slot, 0);
            cnt.assign((size_t)HKV, 0);
            for (int h = 0; h < HKV; ++h) {
                const int a = off[(size_t)h], b = off[(size_t)h + 1];
                const int n = b - a;
                if (n > (int)max_slot) fail("MSA selected " + std::to_string(n) + " slots, more than --idx-top-k blocks");
                cnt[(size_t)h] = n;
                for (int i = 0; i < n; ++i) sp[(size_t)h * (size_t)max_slot + (size_t)i] = slots[(size_t)(a + i)];
            }
        }
        if (kv_mode == 1) {
            if (!dev_slots) {
                cu::copy_h2d(w.slots.p, sp.data(), sp.size() * sizeof(int32_t));
                cu::copy_h2d(w.counts.p, cnt.data(), cnt.size() * sizeof(int32_t));
            }
            cu::kv_gather(L.d_k.p, L.d_v.p, HKV, D, (const int32_t*)w.slots.p, (const int32_t*)w.counts.p,
                          (int)max_slot, w.kstage.p, w.vstage.p, nullptr);
        } else {
            cu::copy_h2d(w.slots.p, sp.data(), sp.size() * sizeof(int32_t));
            cu::copy_h2d(w.counts.p, cnt.data(), cnt.size() * sizeof(int32_t));
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
                    if (kv_q) {
                        // P8: the staged device buffer is fp16, so a quantized cache is dequantized on the way
                        // across.  It still halves the host read and the copy.
                        if (ah_stage.size() < (size_t)D * 2) ah_stage.resize((size_t)D * 2);
                        load_head(k_host_q, k_host, src, D, ah_stage.data());
                        to_f16(ah_stage.data(), w.kstage_h + dst, D);
                        load_head(v_host_q, v_host, src, D, ah_stage.data());
                        to_f16(ah_stage.data(), w.vstage_h + dst, D);
                    } else {
                        std::memcpy(w.kstage_h + dst, k_host + src, row);
                        std::memcpy(w.vstage_h + dst, v_host + src, row);
                    }
                }
                if (n > 0) {
                    const size_t off2 = (size_t)h * (size_t)max_slot * (size_t)D;
                    cu::copy_h2d((uint16_t*)w.kstage.p + off2, w.kstage_h + off2, (size_t)n * row);
                    cu::copy_h2d((uint16_t*)w.vstage.p + off2, w.vstage_h + off2, (size_t)n * row);
                }
            }
        }
        const auto t_msa = std::chrono::steady_clock::now();
        float* att_dst = att_dev_out ? att_dev_out : (float*)w.att.p;
        if (qdev) {
            cu::msa_attn(w.kstage.p, w.vstage.p, qdev, att_dst, H, q_per_kv, (int)max_slot,
                         (const int32_t*)w.counts.p, D, nullptr);
        } else {
            h2d(w, w.qd.p, q.data(), (size_t)H * (size_t)D * sizeof(float));
            cu::msa_attn(w.kstage.p, w.vstage.p, (const float*)w.qd.p, att_dst, H, q_per_kv, (int)max_slot,
                         (const int32_t*)w.counts.p, D, nullptr);
        }
        if (!att_dev_out) {
            cu::sync();
            cu::copy_d2h(att, att_dst, (size_t)H * (size_t)D * sizeof(float));
        }
        if (opt.profile_time) prof.attn_sync += pms(t_msa);
    }

    void attend(int layer, const Layer& L, int64_t pos, float* att, const float* qdev = nullptr,
                const float* iqdev = nullptr, float* att_dev_out = nullptr) {
        if (kv_mode == 0)
            attend_host(layer, L, pos, att);
        else
            attend_dev(layer, L, pos, att, qdev, iqdev, att_dev_out);
    }


    // ===================================================================== P5a: batched prompt processing

    void init_prefill() {
        prefetch_slots = opt.expert_prefetch;
        if (opt.prefill_batch <= 1) return;
        prefill_dense_f32 = opt.prefill_dense != "f16";
        prefill_expert_gemm = opt.prefill_expert_gemm.empty() ? "mmvq" : opt.prefill_expert_gemm;
        if (prefill_expert_gemm != "mmvq" && prefill_expert_gemm != "tc" && prefill_expert_gemm != "auto")
            fail("--minimax-prefill-expert-gemm: '" + prefill_expert_gemm + "' is not mmvq, tc or auto");
        prefill_direct = stream_direct;
        if (!prefill_dense_f32)
            std::fprintf(stderr,
                         "INFO minimax-prefill-dense=f16 (the down GEMM's activation is fp16; measured KL 1.7e-3 "
                         "against f32 on the real slice, +9%% prefill)\n");
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
        // P8: the RAM tier''s prompt pass runs attention on the card when one layer''s K/V fits there.  The device
        // kernel reads fp16 rows, so a q8_0 cache keeps the host path (documented in the plan line).
        prefill_kv_stage = kv_mode == 2 && !work.empty() && opt.prefill_kv != "off" &&
                           cfg.n_head_kv == cfg.idx_heads && cfg.idx_first_layer == 0;
        if (prefill_kv_stage && kv_q) {
            // Correct, just not batched: the token path''s device attention reads the q8 blocks directly.
            std::fprintf(stderr,
                         "INFO minimax-prefill-kv: the device MSA kernels read fp16 rows and the cache is q8_0; "
                         "keeping the token path for the prompt (slower, same result). Use f16 K/V, or the vram "
                         "tier, for a batched prompt at long context.\n");
            prefill_kv_stage = false;
            batch_cap = 0;
            return;
        }
        // The device dequant path (iq_mmvq) has no kernel for the expert formats ported from ik_llama.cpp.
        // In `auto` (the default) a model made of them computes its experts on the CPU, the path the G3 runs
        // use, instead of failing the first batched chunk with "iq_mmvq: type 164 is not supported".  An
        // explicit `gpu` still asks for the device and fails with that message.
        bool ikq_experts = false;
        if (prefill_gpu_experts && opt.prefill_experts != "gpu") {
            for (const Layer& L : layers) {
                if (L.moe && (strata::kernels::ikq::is_iqk_type(L.egate.type) ||
                              strata::kernels::ikq::is_iqk_type(L.eup.type) ||
                              strata::kernels::ikq::is_iqk_type(L.edown.type))) {
                    ikq_experts = true;
                    prefill_gpu_experts = false;
                    break;
                }
            }
        }
        if (max_expert_bytes == 0)
            for (const Layer& L : layers)
                if (L.moe && L.expert_bytes > max_expert_bytes) max_expert_bytes = L.expert_bytes;
        max_dense_n = cfg.n_ff_dense > cfg.n_head * cfg.head_dim ? cfg.n_ff_dense : cfg.n_head * cfg.head_dim;
        if (cfg.n_ff_exp > max_dense_n) max_dense_n = cfg.n_ff_exp;
        if (cfg.n_embd > max_dense_n) max_dense_n = cfg.n_embd;
        if (cfg.n_expert > max_dense_n) max_dense_n = cfg.n_expert;

        // Per-token device bytes.  P9: this used to size three fp32 buffers at `max_dense_n` for the whole chunk
        // (the dense MLP only ever touches one rows_cap tile of that width) plus an fp16 buffer nothing read, which
        // cost 188 KiB/token and clamped a 32 K chunk to ~20 K tokens.  The real T-wide needs are the GEMM output
        // (qlen = 8192 on M3.1, not 12288), the projection scratch, and the MoE sum (embd): 140 KiB/token.
        const size_t per_tok = (size_t)prefill_bytes_per_token();
        const size_t fixed = (size_t)224 << 20;
        for (size_t d = 0; d < devs.size(); ++d) {
            const cu::MemInfo mi = cu::mem_on(devs[d]);
            const size_t reserve = ((size_t)std::max(opt.vram_reserve_mib, 256) << 20) + fixed;
            // P12: in the swap mode the prompt pass may take the expert slots back for its workspace, so what the
            // batch may use is the free VRAM plus every slot on this card.
            const size_t reclaim = prefill_ws_swap ? slotted_bytes_on((int)d) : 0;
            size_t free_now = mi.free_bytes;
            const size_t want = reserve + per_tok * (size_t)batch_cap;
            if (!prefill_ws_swap && free_now < want && slotted_bytes_on((int)d) > 0) {
                // P14: the slot plan sets the workspace aside, but allocations made after it (lm_head, the shared
                // expert's buffers, cuBLAS) can still leave the card short, and the prompt then silently ran token by
                // token (7-10 tok/s on rig-2's 2 GPUs).  A few decode slots cost far less than that.
                const int shed = shed_slots((int)d, want, false);
                free_now = cu::mem_on(devs[d]).free_bytes;
                std::fprintf(stderr,
                             "INFO minimax-prefill-workspace dev=%d gave up %d expert slots for the %d-token batch "
                             "(%zu MiB free now)\n",
                             devs[d], shed, batch_cap, free_now >> 20);
            }
            const size_t budget = free_now + reclaim;
            if (budget <= reserve) {
                std::fprintf(stderr,
                             "strata-minimax: batched prefill needs workspace on device %d but only %zu MiB is free. "
                             "Lower --minimax-prefill-batch or --minimax-dense-layers-gpu.\n",
                             devs[d], free_now >> 20);
                batch_cap = 0;
                return;
            }
            const size_t fit = (budget - reserve) / per_tok;
            std::fprintf(stderr,
                         "INFO minimax-prefill-workspace dev=%d free=%zu MiB%s reserve=%zu MiB per_token=%.1f KiB "
                         "fit=%zu tokens\n",
                         devs[d], free_now >> 20, reclaim ? (" +slots=" + std::to_string(reclaim >> 20) + " MiB").c_str() : "",
                         reserve >> 20, (double)per_tok / 1024.0, fit);
            if ((size_t)batch_cap > fit) batch_cap = (int)fit;
        }
        if (batch_cap < 2) {
            std::fprintf(stderr, "strata-minimax: batched prefill does not fit in VRAM; keeping the token path\n");
            batch_cap = 0;
            return;
        }
        if (rows_cap > batch_cap) rows_cap = batch_cap;

        if (prefill_direct && ssd_files.empty()) {
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
        ws_taken.assign(layers.size(), 0);
        // P12: `pf_max_blk` is read by the host buffers below and by the device allocation later, so it is settled
        // here with the rest of the batch geometry.  It used to be set inside the allocation loop, which now runs at
        // the first prompt.
        pf_max_blk = (int)cfg.idx_top_k;
        // P12: the pinned expert-slot layout is host arithmetic that both the prompt ring and the decode path's
        // direct reads use, so it is computed here once, not with the (per-prompt) device buffers.
        if (prefill_gpu_experts && max_expert_bytes > 0) {
            size_t gu = 0, dn = 0;
            for (const Layer& L : layers)
                if (L.moe) {
                    if (L.expert_gu > gu) gu = L.expert_gu;
                    if (L.expert_down > dn) dn = L.expert_down;
                }
            const size_t A = strata::platform::DirectFile::alignment();
            auto up = [&](size_t x) { return (x + A - 1) & ~(A - 1); };
            for (PfBuf& p : pf) {
                p.ring_slot = max_expert_bytes;
                p.role_off[0] = A;
                p.role_off[1] = up(p.role_off[0] + gu) + A;
                p.role_off[2] = up(p.role_off[1] + gu) + A;
                p.pin_slot_bytes = up(p.role_off[2] + dn) + A;
            }
        }
        // The workspace's size on one device, from the settled batch: with the tiers off there are no slots to
        // borrow from and the allocation is just the free VRAM.
        prefill_ws_bytes = prefill_device_bytes(batch_cap);
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
        // The dense MLP layers are wider than the routed experts (12288 vs 3072 on M3.1).  These three buffers
        // are used by BOTH the dense-layer CPU MLP and the shared expert, so size them for the wider of the two:
        // with rows_cap * n_ff_exp only, one 512-row chunk of layer 0 wrote 4x past the end of the buffer
        // (G3 on IQ2_XS-D3 crashed with an access violation in gemm_cpu_base's row loop).
        const size_t pf_ff = (size_t)bigger(cfg.n_ff_exp, cfg.n_ff_dense);
        pf_gate_h.assign((size_t)rows_cap * pf_ff, 0.f);
        pf_up_h.assign((size_t)rows_cap * pf_ff, 0.f);
        pf_hid_h.assign((size_t)rows_cap * pf_ff, 0.f);
        pf_dense_h.assign((size_t)batch_cap * (size_t)(cfg.n_head * cfg.head_dim), 0.f);
        pf_moe_h.assign((size_t)batch_cap * (size_t)cfg.n_embd, 0.f);
        pf_out_tile.assign((size_t)batch_cap * (size_t)std::min<int64_t>(4096, cfg.n_vocab), 0.f);
        std::fprintf(stderr, "INFO minimax-prefill batch=%d experts=%s workspace=%.1f MiB/device reads=%s\n",
                     batch_cap, prefill_gpu_experts ? "gpu" : "cpu",
                     (double)per_tok * (double)batch_cap / 1048576.0, prefill_direct ? "direct" : "mmap");
        if (prefill_gpu_experts)
            std::fprintf(stderr, "INFO minimax-prefill-expert-gemm=%s%s\n", prefill_expert_gemm.c_str(),
                         prefill_expert_gemm == "auto" ? " (tensor cores from 16 rows per expert)" : "");
        if (ikq_experts)
            std::fprintf(stderr, "INFO minimax-prefill the expert format has no device prefill path; experts run "
                                 "on the CPU (--minimax-prefill-experts gpu to ask for the device anyway)\n");
    }

        PfBuf& P() { return pf[(size_t)cur]; }

    // =========================================================== P12: the prompt workspace and the slots share VRAM
    //
    // The batched prompt workspace is only needed while a prompt is being processed; the decode expert slots are
    // only needed after it.  Before P12 both were reserved at startup, so a server that always passes
    // `--minimax-prefill-batch 4096` gave the workspace ~700 MiB of every card out of the decode slot pool for the
    // whole session.  Now the workspace is allocated at the start of a prompt (freeing the slots it displaces) and
    // released when the prompt ends, and the slots are re-allocated, so decode runs with the decode-only count.
    // The kernels and their arguments are untouched: only the residency of the slots changes, exactly as the
    // adaptive swaps already change it during a run.

    /// Every VRAM expert slot on device `d`, in bytes: what the prompt workspace may take back.
    size_t slotted_bytes_on(int d) const {
        size_t b = 0;
        for (const Layer& L : layers)
            if (L.moe && L.dev == d) b += (size_t)L.vram.size() * L.expert_bytes;
        return b;
    }

    /// Release the device side of the batched-prompt workspace.  Idempotent.  The pinned slot *layout*
    /// (`role_off`, `pin_slot_bytes`, `ring_slot`) is host-side arithmetic that the decode path's direct reads also
    /// use, so it is computed once in init_prefill and survives this: only the buffers go away.
    void pf_free() {
        if (!ws_live) return;
        for (size_t d = 0; d < pf.size(); ++d) {
            PfBuf& p = pf[d];
            use_device_index((int)d);
            cu::dealloc(p.h16);          p.h16 = nullptr;
            cu::dealloc(p.a16);          p.a16 = nullptr;
            cu::dealloc(p.y32);          p.y32 = nullptr;
            cu::dealloc(p.p32);          p.p32 = nullptr;
            cu::dealloc(p.sh);           p.sh = nullptr;
            cu::dealloc(p.w16);          p.w16 = nullptr;
            cu::dealloc(p.w32);          p.w32 = nullptr;
            cu::dealloc(p.gx);           p.gx = nullptr;
            cu::dealloc(p.gx32);         p.gx32 = nullptr;
            cu::dealloc(p.gy);           p.gy = nullptr;
            cu::dealloc(p.gu);           p.gu = nullptr;
            cu::dealloc(p.gh);           p.gh = nullptr;
            cu::dealloc(p.gh32);         p.gh32 = nullptr;
            cu::dealloc(p.gd);           p.gd = nullptr;
            cu::dealloc(p.xq);           p.xq = nullptr;
            cu::dealloc(p.hq);           p.hq = nullptr;
            cu::dealloc(p.d_rows.p);     p.d_rows.p = nullptr;
            cu::dealloc(p.d_ent_row.p);  p.d_ent_row.p = nullptr;
            cu::dealloc(p.d_ent_w.p);    p.d_ent_w.p = nullptr;
            cu::dealloc(p.d_iq16.p);     p.d_iq16.p = nullptr;
            cu::dealloc(p.d_q16.p);      p.d_q16.p = nullptr;
            cu::dealloc(p.d_bs.p);       p.d_bs.p = nullptr;
            cu::dealloc(p.d_blk.p);      p.d_blk.p = nullptr;
            cu::dealloc(p.d_nblk.p);     p.d_nblk.p = nullptr;
            cu::dealloc(p.d_pos.p);      p.d_pos.p = nullptr;
            cu::dealloc(p.kv_k.p);       p.kv_k.p = nullptr;
            cu::dealloc(p.kv_v.p);       p.kv_v.p = nullptr;
            for (void* r : p.ring_v) cu::dealloc(r);
            for (void* h : p.pin_v) cu::host_free(h);
            for (void* e : p.ev_copy) cu::event_destroy(e);
            for (void* e : p.ev_comp) cu::event_destroy(e);
            p.ring_v.clear();
            p.pin_v.clear();
            p.ev_copy.clear();
            p.ev_comp.clear();
            p.ring = nullptr;
            p.pin = nullptr;
            p.ring_n = 0;
            p.x16 = nullptr;
            p.y32_elems = p.p32_elems = p.sh_elems = 0;
            p.w16_elems = 0;
            p.rows_cap = 0;
        }
        use_device_index(0);
        ws_live = false;
    }

    /// Allocate the device side of the batched-prompt workspace for `batch_cap` tokens.  The caller has already made
    /// room (`pf_acquire`).  A cudaMalloc that still does not fit throws, which is the honest failure: the plan
    /// above counted these bytes.
    void pf_alloc() {
        if (ws_live) return;
        // Mark it live first: a cudaMalloc that throws part-way leaves a half-filled PfBuf that pf_free() must be
        // able to release (every pointer starts null and is set one at a time).
        ws_live = true;
        for (size_t d = 0; d < devs.size(); ++d) {
            use_device_index((int)d);
            PfBuf& p = pf[d];
            const size_t T = (size_t)batch_cap;
            const size_t embd = (size_t)cfg.n_embd;
            const size_t qlen = (size_t)(cfg.n_head * cfg.head_dim);
            const size_t klen = (size_t)(cfg.n_head_kv * cfg.head_dim);
            const size_t wide = std::max(qlen, embd);
            const size_t tile_wide = (size_t)rows_cap * (size_t)max_dense_n;   // the dense MLP's rows_cap tile
            p.x16 = nullptr;   // P9: allocated and never read; 24 KiB/token of nothing
            p.h16 = (uint16_t*)cu::alloc(T * embd * 2);
            // The attention output of a chunk is [T][n_head*head_dim], wider than the hidden state; the f16 path
            // used to write it into p.h16 and run past the end of that buffer (non-finite logits on real weights).
            p.a16 = (uint16_t*)cu::alloc(T * std::max(qlen, klen) * 2);
            // P9: qlen/embd wide, not max_dense_n; the dense MLP's 12288-wide gate/up live in a rows_cap tile.
            p.y32_elems = std::max(T * wide, tile_wide);
            p.p32_elems = std::max(T * wide, tile_wide);
            p.sh_elems = std::max(T * embd, tile_wide);
            p.y32 = (float*)cu::alloc(p.y32_elems * 4);
            p.p32 = (float*)cu::alloc(p.p32_elems * 4);
            p.sh = (float*)cu::alloc(p.sh_elems * 4);
            p.w16_elems = std::min<int64_t>(4096, max_dense_n) * (int64_t)cfg.n_embd;
            p.w16 = (uint16_t*)cu::alloc((size_t)p.w16_elems * 2);
            p.w32 = (float*)cu::alloc((size_t)p.w16_elems * 4);
            p.rows_cap = rows_cap;
            const size_t rc = (size_t)rows_cap;
            p.gx = (uint16_t*)cu::alloc(rc * embd * 2);
            p.gx32 = (float*)cu::alloc(rc * embd * 4);
            p.gy = (float*)cu::alloc(rc * (size_t)cfg.n_ff_exp * 4);
            p.gu = (float*)cu::alloc(rc * (size_t)cfg.n_ff_exp * 4);
            p.gh = (uint16_t*)cu::alloc(rc * (size_t)max_dense_n * 2);
            p.gh32 = (float*)cu::alloc(rc * (size_t)max_dense_n * 4);
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
            p.d_blk = DevPtr(cu::alloc(T * (size_t)cfg.idx_heads * (size_t)pf_max_blk * sizeof(int32_t) + 4096));
            p.d_nblk = DevPtr(cu::alloc(T * (size_t)cfg.idx_heads * sizeof(int32_t) + 4096));
            p.d_pos = DevPtr(cu::alloc((size_t)T * sizeof(int32_t) + 64));
            // ---- P8: the long-context prompt path.  With the K/V in pinned RAM the batched attention used to fall
            // back to attend_host(), whose indexer score loop is scalar -- ~1 s/token at 32 K.  Instead stage one
            // layer's whole K/V prefix on the card (a single large copy per layer, ~0.06 s per 1 K tokens) and run
            // the existing device MSA kernel.  The staging buffer is [ctx][n_head_kv][head_dim] fp16, the same
            // layout the device KV tier uses, so the selected block ids are absolute positions in both.
            if (prefill_kv_stage) {
                const size_t kvb = (size_t)kv_row_bytes * (size_t)ctx;
                const cu::MemInfo mi = cu::mem_on(devs[d]);
                const size_t want = kvb * 2 + ((size_t)64 << 20);
                if (mi.free_bytes > want) {
                    p.kv_k = DevPtr(cu::alloc(kvb));
                    p.kv_v = DevPtr(cu::alloc(kvb));
                } else {
                    std::fprintf(stderr,
                                 "INFO minimax-prefill-kv: device %d cannot hold one layer's K/V (%.0f MiB "
                                 "needed, %.0f MiB free); keeping host attention for this device\n",
                                 devs[d], (double)want / 1048576.0, (double)mi.free_bytes / 1048576.0);
                }
            }
            if (prefill_gpu_experts && max_expert_bytes > 0) {
                // P10: the pinned slot's role layout (see PfBuf::role_off): gate, up and down each at a 4 KiB
                // boundary with 4 KiB of headroom, so an unbuffered read of the aligned superset can land there.
                // Four slots unless the card cannot spare them: 64 MiB of ring buys copy/compute overlap on the
                // expert stream, which is what the chunk's MoE time is actually spent on.
                size_t slot = p.ring_slot;
                int want = 4;
                const size_t per_slot = slot + (16u << 20);
                while (want > 1) {
                    const cu::MemInfo mi = cu::mem_on(devs[d]);
                    if (mi.free_bytes > per_slot * (size_t)want) break;
                    --want;
                }
                p.ring_n = want;
                for (int s = 0; s < p.ring_n; ++s) {
                    p.ring_v.push_back(cu::alloc(slot));
                    p.pin_v.push_back(cu::host_alloc(p.pin_slot_bytes));
                    p.ev_copy.push_back(cu::event_create());
                    p.ev_comp.push_back(cu::event_create());
                }
                p.ring = p.ring_v[0];
                p.pin = p.pin_v[0];
                if (!p.copy_stream) p.copy_stream = cu::stream_create();
            }
        }
        use_device_index(0);
        ws_live = true;
    }

    // `--minimax-prefill-vram swap`'s two halves.  `pf_acquire` frees expert slots until every card has room for
    // the workspace and then allocates it; `pf_release` frees the workspace and re-allocates exactly the slots it
    // took.  Slots come back empty (the experts they held are marked non-resident), which is the same state a
    // decode-only run starts in; the pinning tier and the retier loop refill them on the next tokens.
    void pf_release() {
        if (!ws_live) return;
        const auto ws_t0 = std::chrono::steady_clock::now();
        cu::sync();
        for (size_t d = 0; d < work.size(); ++d) {
            if (!work[d].side) continue;
            use_device_index((int)d);
            cu::stream_sync(work[d].side);
        }
        use_device_index(0);
        pf_free();
        int restored = 0, lost = 0;
        for (size_t li = 0; li < layers.size(); ++li) {
            Layer& L = layers[li];
            if (ws_taken[li] <= 0) continue;
            use_device_index(L.dev);
            for (int s = 0; s < ws_taken[li]; ++s) {
                // P14: the prompt pass can leave the card with less than it borrowed (cuBLAS and the allocator keep
                // some of it), and a throw here killed the server after its first prompt.  Keep what fits.
                void* p = nullptr;
                try {
                    p = cu::alloc(L.expert_bytes);
                } catch (const std::exception&) {
                    lost += ws_taken[li] - s;
                    break;
                }
                Layer::VramSlot slot;
                slot.mem = DevPtr(p);
                L.vram.push_back(std::move(slot));
                ++restored;
            }
            ws_taken[li] = 0;
        }
        use_device_index(0);
        cu::sync();
        if (lost > 0)
            std::fprintf(stderr, "WARN minimax-prefill-vram %d expert slots did not fit back after the prompt; "
                                 "decode keeps the rest\n", lost);
        if (restored > 0) {
            int lo = std::numeric_limits<int>::max(), hi = 0;
            for (const Layer& L : layers)
                if (L.moe && !L.vram.empty()) {
                    lo = std::min(lo, (int)L.vram.size());
                    hi = std::max(hi, (int)L.vram.size());
                }
            std::fprintf(stderr,
                         "INFO minimax-prefill-vram workspace released, %d expert slots re-allocated (%d..%d per "
                         "layer) in %.0f ms\n",
                         restored, lo, hi,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ws_t0).count());
        }
    }

    /// Free decode expert slots on device `d`, round-robin over its MoE layers, until it has `need` bytes free or
    /// no slot is left.  `borrow` records them in ws_taken so pf_release re-allocates them after the prompt (swap);
    /// otherwise they are gone for the run (init_prefill making room for the workspace).  Returns the count freed.
    int shed_slots(int d, size_t need, bool borrow) {
        std::vector<size_t> moe_on_d;
        for (size_t li = 0; li < layers.size(); ++li)
            if (layers[li].moe && layers[li].dev == d) moe_on_d.push_back(li);
        int freed = 0;
        size_t turn = 0;
        bool any = true;
        while (any && cu::mem_on(devs[(size_t)d]).free_bytes < need) {
            any = false;
            for (size_t k = 0; k < moe_on_d.size(); ++k) {
                const size_t li = moe_on_d[(turn + k) % moe_on_d.size()];
                Layer& L = layers[li];
                if (L.vram.empty()) continue;
                use_device_index(L.dev);
                Layer::VramSlot& sl = L.vram.back();
                if (sl.expert >= 0) L.vram_of[(size_t)sl.expert] = kNotResident;
                // `VramSlot::mem` is an RAII DevPtr: free it here and null it, or pop_back's destructor frees the
                // same pointer again (a double free that surfaces later as a bogus launch error).
                cu::dealloc(sl.mem.p);
                sl.mem.p = nullptr;
                L.vram.pop_back();
                if (borrow) ++ws_taken[li];
                ++freed;
                any = true;
                break;   // re-check the free bytes before taking another
            }
            ++turn;
        }
        use_device_index(0);
        return freed;
    }

    /// Make room for the workspace by taking slots away (swap mode only), then allocate it.  Returns false when the
    /// allocation still does not fit (the caller then keeps the token path).  `--minimax-prefill-vram reserve` and an
    /// explicit `--minimax-vram-slots-per-layer` never take a slot away: their workspace comes out of whatever
    /// free VRAM init_prefill sized the batch against, exactly as before P12.
    bool pf_acquire() {
        if (ws_live) return true;
        // cudaFree is a device-wide sync: do it before the prompt, never inside one.  The expert side streams get
        // an explicit sync too, because the adaptive retier and the lookahead copy into slots asynchronously and a
        // slot must not be freed while a copy into it is still in flight.
        cu::sync();
        for (size_t d = 0; d < work.size(); ++d) {
            if (!work[d].side) continue;
            use_device_index((int)d);
            cu::stream_sync(work[d].side);
        }
        use_device_index(0);
        const auto ws_t0 = std::chrono::steady_clock::now();
        const size_t margin = (size_t)std::max(opt.vram_reserve_mib, 256) << 20;
        const size_t need = prefill_ws_bytes + margin;
        int freed = 0;
        if (prefill_ws_swap) {
            for (size_t d = 0; d < devs.size(); ++d) {
                // Round-robin over this device's MoE layers so no single layer loses its whole pool.
                freed += shed_slots((int)d, need, true);
                if (cu::mem_on(devs[d]).free_bytes < need) {
                    use_device_index(0);
                    std::fprintf(stderr,
                                 "WARN minimax-prefill-vram device %d still has only %zu MiB free after taking %d "
                                 "slots; the batched prompt may not fit\n",
                                 devs[d], cu::mem_on(devs[d]).free_bytes >> 20, freed);
                }
            }
        }
        use_device_index(0);
        if (freed > 0)
            std::fprintf(stderr,
                         "INFO minimax-prefill-vram borrowed %d expert slots for the prompt workspace in %.0f ms\n",
                         freed, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ws_t0).count());
        try {
            pf_alloc();
        } catch (const std::exception& ex) {
            std::fprintf(stderr,
                         "WARN minimax-prefill-vram the %.0f MiB/device prompt workspace did not fit (%s); keeping "
                         "the token path for this run\n",
                         (double)prefill_ws_bytes / 1048576.0, ex.what());
            pf_release();
            batch_on = false;
            batch_cap = 0;
            return false;
        }
        return ws_live;
    }

    /// RAII: the prompt workspace exists for exactly the duration of a prompt.  Every path that runs
    /// `forward_batch` opens one; the destructor gives the slots back even when a run is cancelled.
    struct PrefillWs {
        Impl* im = nullptr;
        bool on = false;
        explicit PrefillWs(Impl* i) : im(i) {
            if (i->batch_on) on = i->pf_acquire();
        }
        ~PrefillWs() {
            if (on) im->pf_release();
        }
        PrefillWs(const PrefillWs&) = delete;
        PrefillWs& operator=(const PrefillWs&) = delete;
    };

    // P9: the batched prompt workspace, bytes per token per device.  Kept in one place because the tier sizing
    // (which must leave room for it) and the allocator below both need it.
    int64_t prefill_bytes_per_token() const {
        const int64_t embd = cfg.n_embd;
        const int64_t qlen = cfg.n_head * cfg.head_dim;
        const int64_t klen = cfg.n_head_kv * cfg.head_dim;
        const int64_t wide = bigger(qlen, embd);
        const int64_t nbmax = (ctx + cfg.idx_block - 1) / cfg.idx_block;
        return 2 * embd                              // h16
             + 2 * bigger(qlen, klen)                // a16 (attention output, K/V staging)
             + 4 * wide * 2                          // y32, p32
             + 4 * embd                              // sh (MoE combine sum)
             + 2 * qlen                              // d_q16
             + 2 * cfg.idx_heads * cfg.idx_dim       // d_iq16
             + 4 * cfg.idx_heads * nbmax             // d_bs
             + 4 * cfg.idx_heads * cfg.idx_top_k     // d_blk
             + 4 * cfg.idx_heads + 4;                // d_nblk, d_pos
    }

    // P5c: how many positions the device cos/sin table covers (all devices share the same cap).
    int64_t p_rope_cap() const { return work.empty() ? 0 : work[0].rope_pos_cap; }

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

    // One expert matrix (N rows of K) in tiles of the weight scratch, the same tiling gemm_gpu uses: the dequant
    // buffer holds min(4096, max_dense_n) x n_embd values, which is smaller than a full 6144 x 6144 expert matrix.
    void gemm_expert_gpu(int type, const void* src, size_t row_bytes, const uint16_t* X16, const float* X32, float* Y,
                         int T, int N, int K) {
        PfBuf& p = P();
        int64_t tile = p.w16_elems / K;
        if (tile > N) tile = N;
        if (tile < 1) fail("the batched-prefill weight scratch is smaller than one expert row");
        for (int r0 = 0; r0 < N; r0 += (int)tile) {
            const int n = (N - r0) < (int)tile ? (N - r0) : (int)tile;
            const uint8_t* s = (const uint8_t*)src + (size_t)r0 * row_bytes;
            if (prefill_dense_f32) {
                cu::dequant_to_f32(type, s, (int64_t)n * K, p.w32, nullptr);
                cu::gemm_f32(X32, p.w32, Y + r0, T, n, K, N, 0.f, nullptr);
            } else {
                cu::dequant_to_f16(type, s, (int64_t)n * K, p.w16, nullptr);
                cu::gemm_f16(X16, p.w16, Y + r0, T, n, K, N, 0.f, nullptr);
            }
        }
    }

    // P10: the tensor-core expert path.  One expert's weight matrix is dequantized to fp16 once (in tiles of the
    // weight scratch) and every token routed to that expert is a row of one cuBLAS HGEMM with FP32 accumulation.
    // The mmvq path decodes the same weights into int8 dots per column; the dequantization is the same amount of
    // work per weight either way, so the GEMM is free to be a tensor-core one.
    bool tc_expert_type(int t) const {
        if (t == 0 || t == 1 || t == 30) return true;   // F32 / F16 / BF16
        // iq_supported covers the i-quants and the raw ggml block types dequant_f16 reads; the ported ik_llama
        // formats are row-scaled and the flat dequantizer (deliberately) refuses them.
        return strata::kernels::iq_supported(t) && !strata::kernels::ikq::is_iqk_type(t);
    }

    void gemm_expert_f16(int type, const void* src, size_t row_bytes, const uint16_t* X16, float* Y, int T, int N,
                         int K, int ldy) {
        PfBuf& p = P();
        int64_t tile = p.w16_elems / K;
        if (tile > N) tile = N;
        if (tile < 1) fail("the batched-prefill weight scratch is smaller than one expert row");
        for (int r0 = 0; r0 < N; r0 += (int)tile) {
            const int n = (N - r0) < (int)tile ? (N - r0) : (int)tile;
            const uint8_t* s = (const uint8_t*)src + (size_t)r0 * row_bytes;
            cu::dequant_to_f16(type, s, (int64_t)n * K, p.w16, nullptr);
            cu::gemm_f16(X16, p.w16, Y + r0, T, n, K, ldy, 0.f, nullptr);
        }
    }

    // The same projection on the CPU: every weight row is read once and dotted with all T activations.
    void gemm_cpu_base(const Tensor& t, const uint8_t* base, const float* X, int T, int ldx, float* Y, int ldy) {
        const int N = (int)t.ne1, K = (int)t.ne0;
        if (strata::kernels::ikq::is_iqk_type(t.type)) {
            // The ported ik_llama.cpp formats have no ggml type id, so their batched projection uses the
            // engine's own codec, exactly as `dot_rows_base` does one row at a time.  Without this branch a
            // batched prefill of an IQ2_KL / IQ2_KT / IQ3_K / IQ4_KS model fails with "ggml-cpu has no
            // vec_dot for type 164" (type 164 is IQ2_KL) before a single token runs.
            if ((K % 256) != 0) {
                fail(t.name + ": " + strata::kernels::ikq::type_name(t.type) +
                     " needs a width that is a whole number of 256-value blocks");
            }
            const ggml_type_traits_cpu* act = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
            if (!act || !act->from_float) fail(t.name + ": no from_float for the q8_K activation");
            const size_t act_bytes = ggml_row_size(GGML_TYPE_Q8_K, K);
            std::vector<uint8_t> aq(act_bytes * (size_t)T);
            for (int c = 0; c < T; ++c) act->from_float(X + (size_t)c * ldx, aq.data() + (size_t)c * act_bytes, K);
            pool.run(N, [&](int a, int b) {
                for (int r = a; r < b; ++r) {
                    const uint8_t* row = base + (size_t)r * t.row_bytes;
                    for (int c = 0; c < T; ++c)
                        Y[(size_t)c * ldy + r] = strata::kernels::ikq::dot_row(
                            t.type, row, aq.data() + (size_t)c * act_bytes, K);
                }
            });
            return;
        }
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
        ssd_wait_ms = 0.0;
        decay_usage();
        const bool pt = opt.profile_time;
        const auto t_tok = std::chrono::steady_clock::now();
        if (pt) prof = Prof{};
        if (!seed_h.empty()) {
            if (pos >= seed_rows) fail("--minimax-seed-hidden has only " + std::to_string(seed_rows) + " rows");
            std::memcpy(x.data(), seed_h.data() + (size_t)pos * (size_t)cfg.n_embd, (size_t)cfg.n_embd * sizeof(float));
        } else {
            decode_f32(embd, embd.host + (size_t)token * embd.row_bytes, x.data(), cfg.n_embd);
        }
        if ((size_t)pos < img_row.size()) {
            const int row = img_row[(size_t)pos];
            if (row >= 0)
                std::memcpy(x.data(), img_data.data() + (size_t)row * (size_t)cfg.n_embd,
                            (size_t)cfg.n_embd * sizeof(float));
        }
        if (pt) prof.embed = pms(t_tok);

        const bool time_on = std::getenv("STRATA_MM_TIME") != nullptr;
        using tokclock = std::chrono::steady_clock;
        auto ms = [](tokclock::time_point a) {
            return std::chrono::duration<double, std::milli>(tokclock::now() - a).count();
        };
        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            if (kv_mode != 0 || L.want_gpu) use_layer_device(li);
            const auto t_layer = tokclock::now();
            const bool dbg = dumping(li);
            const std::string pfx = "L" + std::to_string(src_layer[(size_t)li]);
            if (dbg) dbuf_put(pfx + "/residual_in", x.data(), (int)cfg.n_embd);
            rms(x.data(), L.attn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            if (dbg) dbuf_put(pfx + "/attn_input", h.data(), (int)cfg.n_embd);
            // P5c: the token path's norm+rope on the device.  `dev_tok` keeps q/k/v on the card: the three
            // projections write device buffers, the fused kernel norms/rotates them there and drops K/V straight
            // into the cache, so the host sees one position update and no 16 K-float round trip per layer.
            const bool dev_tok = kv_mode != 0 && L.want_gpu && gpu_norm_rope && L.d_qn.p && cfg.n_rot > 0 &&
                                 pos < p_rope_cap();
            bool graphed = false;
            const size_t koff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
            const size_t koff_dev = (size_t)pos * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
            if (dev_tok) {
                Work& w = W();
                const int H = (int)cfg.n_head, HKV = (int)cfg.n_head_kv, D = (int)cfg.head_dim;
                const int32_t ppos = (int32_t)pos;
                // P5c: one captured graph per layer holds the whole GPU prefix of the attention block - q8_1
                // quantize, the dense MMVQs, the fused norm+rope and the fp16 conversions - so a token costs one
                // graph launch per layer instead of ~9 kernel launches.  Capture does not execute the work, so
                // the layer runs its graph (replay or first launch) right after building it.
                if (graphs_on && kv_mode == 1 && !dbg) {
                    const bool fits = dense_mmvq && mmvq_dense_fits(L.wq.type, (int)cfg.n_embd) &&
                                      mmvq_dense_fits(L.wk.type, (int)cfg.n_embd) &&
                                      mmvq_dense_fits(L.wv.type, (int)cfg.n_embd) &&
                                      (!L.indexer || (mmvq_dense_fits(L.iq.type, (int)cfg.n_embd) &&
                                                      mmvq_dense_fits(L.ik.type, (int)cfg.n_embd)));
                    if (fits) {
                        if (!w.dsb) w.dsb = cu::stream_create_blocking();
                        auto emit = [&] {
                            strata::kernels::quantize_q8_1_rows((const float*)w.x.p, 1, (int)cfg.n_embd, w.xq.p,
                                                                w.dsb);
                            strata::kernels::native_mmvq(L.wq.type, L.wq.dev.p, w.xq.p, (float*)w.qn.p,
                                                         (int)cfg.n_embd, H * D, 1, w.dsb);
                            strata::kernels::native_mmvq(L.wk.type, L.wk.dev.p, w.xq.p, (float*)w.kn.p,
                                                         (int)cfg.n_embd, HKV * D, 1, w.dsb);
                            strata::kernels::native_mmvq(L.wv.type, L.wv.dev.p, w.xq.p, (float*)w.vn.p,
                                                         (int)cfg.n_embd, HKV * D, 1, w.dsb);
                            cu::head_norm_rope((const float*)w.qn.p, (const float*)L.d_qn.p, (float*)w.qn.p,
                                               nullptr, H, H, D, (int)cfg.n_rot, (const float*)w.rope_cos.p,
                                               (const float*)w.rope_sin.p, (const int32_t*)w.pos1.p, eps, w.dsb);
                            cu::head_norm_rope((const float*)w.kn.p, (const float*)L.d_kn.p, (float*)w.kn.p,
                                               (uint16_t*)w.kn16.p, HKV, HKV, D, (int)cfg.n_rot,
                                               (const float*)w.rope_cos.p, (const float*)w.rope_sin.p,
                                               (const int32_t*)w.pos1.p, eps, w.dsb);
                            cu::f32_to_f16((const float*)w.vn.p, (uint16_t*)w.vn16.p, (int64_t)HKV * D, w.dsb);
                            if (L.indexer) {
                                const int HI = (int)cfg.idx_heads, ID = (int)cfg.idx_dim;
                                strata::kernels::native_mmvq(L.iq.type, L.iq.dev.p, w.xq.p, (float*)w.iqn.p,
                                                             (int)cfg.n_embd, HI * ID, 1, w.dsb);
                                strata::kernels::native_mmvq(L.ik.type, L.ik.dev.p, w.xq.p, (float*)w.ikn.p,
                                                             (int)cfg.n_embd, ID, 1, w.dsb);
                                cu::head_norm_rope((const float*)w.iqn.p, (const float*)L.d_iqn.p,
                                                   (float*)w.iqn.p, nullptr, (int64_t)HI, HI, ID, (int)cfg.n_rot,
                                                   (const float*)w.rope_cos.p, (const float*)w.rope_sin.p,
                                                   (const int32_t*)w.pos1.p, eps, w.dsb);
                                cu::head_norm_rope((const float*)w.ikn.p, (const float*)L.d_ikn.p,
                                                   (float*)w.ikn.p, (uint16_t*)w.ik16.p, (int64_t)1, 1, ID,
                                                   (int)cfg.n_rot, (const float*)w.rope_cos.p,
                                                   (const float*)w.rope_sin.p, (const int32_t*)w.pos1.p, eps,
                                                   w.dsb);
                            }
                        };
                        if (!tok_graph[(size_t)li]) {
                            h2d(w, w.x.p, h.data(), (size_t)cfg.n_embd * sizeof(float));
                            cu::copy_h2d(w.pos1.p, &ppos, sizeof(ppos));
                            cu::graph_capture_begin(w.dsb);
                            emit();
                            tok_graph[(size_t)li] = cu::graph_capture_end(w.dsb);
                            if (!tok_graph[(size_t)li])
                                std::fprintf(stderr, "INFO minimax-graphs layer=%lld capture failed; launching "
                                                     "kernels directly\n",
                                             (long long)li);
                        }
                        if (tok_graph[(size_t)li]) {
                            // The activation and the position are host-side inputs: copied every token, outside
                            // the graph, which only bakes in their device addresses.
                            h2d(w, w.x.p, h.data(), (size_t)cfg.n_embd * sizeof(float));
                            cu::copy_h2d(w.pos1.p, &ppos, sizeof(ppos));
                            cu::graph_launch(tok_graph[(size_t)li], w.dsb);
                            cu::copy_d2d((uint16_t*)L.d_k.p + koff_dev, w.kn16.p, (size_t)HKV * D * 2);
                            cu::copy_d2d((uint16_t*)L.d_v.p + koff_dev, w.vn16.p, (size_t)HKV * D * 2);
                            if (L.indexer)
                                cu::copy_d2d((uint16_t*)L.d_ik.p + (size_t)pos * (size_t)cfg.idx_dim, w.ik16.p,
                                             (size_t)cfg.idx_dim * 2);
                            graphed = true;
                        }
                    }
                }
                if (!graphed) {
                gemv(L.wq, h.data(), nullptr, w.qn.p);
                gemv(L.wk, h.data(), nullptr, w.kn.p);
                gemv(L.wv, h.data(), nullptr, w.vn.p);
                cu::copy_h2d(w.pos1.p, &ppos, sizeof(ppos));
                cu::head_norm_rope((const float*)w.qn.p, (const float*)L.d_qn.p, (float*)w.qn.p, nullptr, H, H, D,
                                   (int)cfg.n_rot, (const float*)w.rope_cos.p, (const float*)w.rope_sin.p,
                                   (const int32_t*)w.pos1.p, eps, nullptr);
                if (kv_mode == 1) {
                    cu::head_norm_rope((const float*)w.kn.p, (const float*)L.d_kn.p, (float*)w.kn.p,
                                       (uint16_t*)L.d_k.p + koff_dev, HKV, HKV, D, (int)cfg.n_rot,
                                       (const float*)w.rope_cos.p, (const float*)w.rope_sin.p,
                                       (const int32_t*)w.pos1.p, eps, nullptr);
                    cu::f32_to_f16((const float*)w.vn.p, (uint16_t*)L.d_v.p + koff_dev, (int64_t)HKV * D, nullptr);
                    if (dbg) {
                        cu::sync();
                        cu::copy_d2h(q.data(), w.qn.p, (size_t)H * D * sizeof(float));
                        cu::copy_d2h(k.data(), w.kn.p, (size_t)HKV * D * sizeof(float));
                        cu::copy_d2h(v.data(), w.vn.p, (size_t)HKV * D * sizeof(float));
                        dbuf_put(pfx + "/q_after_rope", q.data(), (int)(cfg.n_head * cfg.head_dim));
                        dbuf_put(pfx + "/k_after_rope", k.data(), (int)(cfg.n_head_kv * cfg.head_dim));
                    }
                } else {
                    // RAM tier: the rotated K/V are converted on the card and one small copy lands them in the
                    // page-locked host cache.
                    const int n = HKV * D;
                    cu::head_norm_rope((const float*)w.kn.p, (const float*)L.d_kn.p, (float*)w.kn.p, nullptr, HKV,
                                       HKV, D, (int)cfg.n_rot, (const float*)w.rope_cos.p,
                                       (const float*)w.rope_sin.p, (const int32_t*)w.pos1.p, eps, nullptr);
                    cu::f32_to_f16((const float*)w.kn.p, (uint16_t*)w.att.p, n, nullptr);
                    cu::copy_d2h(kv_tmp.data(), w.att.p, (size_t)n * 2);
                    store_kv16(k_host_q, k_host, koff, kv_tmp.data(), n);
                    cu::f32_to_f16((const float*)w.vn.p, (uint16_t*)w.att.p, n, nullptr);
                    cu::copy_d2h(kv_tmp.data(), w.att.p, (size_t)n * 2);
                    store_kv16(v_host_q, v_host, koff, kv_tmp.data(), n);
                }
                }
            } else {
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
            if (kv_mode == 0) {
                store_kv(k_host_q, k_host, koff, k.data(), (int)(cfg.n_head_kv * cfg.head_dim));
                store_kv(v_host_q, v_host, koff, v.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            } else {
                // The owning device's K/V (VRAM tier, one buffer per layer) or the page-locked host K/V (RAM tier).
                const int n = (int)(cfg.n_head_kv * cfg.head_dim);
                if (kv_mode == 2) {
                    store_kv(k_host_q, k_host, koff, k.data(), n);
                    store_kv(v_host_q, v_host, koff, v.data(), n);
                } else {
                    to_f16(k.data(), kv_tmp.data(), n);
                    cu::copy_h2d((uint16_t*)L.d_k.p + koff_dev, kv_tmp.data(), (size_t)n * 2);
                    to_f16(v.data(), kv_tmp.data(), n);
                    cu::copy_h2d((uint16_t*)L.d_v.p + koff_dev, kv_tmp.data(), (size_t)n * 2);
                }
            }
            }
            if (L.indexer) {
                if (dev_tok) {
                    Work& w = W();
                    const int HI = (int)cfg.idx_heads, ID = (int)cfg.idx_dim;
                    if (!graphed) {
                        gemv(L.iq, h.data(), nullptr, w.iqn.p);
                        gemv(L.ik, h.data(), nullptr, w.ikn.p);
                        cu::head_norm_rope((const float*)w.iqn.p, (const float*)L.d_iqn.p, (float*)w.iqn.p, nullptr,
                                           (int64_t)HI, HI, ID, (int)cfg.n_rot, (const float*)w.rope_cos.p,
                                           (const float*)w.rope_sin.p, (const int32_t*)w.pos1.p, eps, nullptr);
                        cu::head_norm_rope((const float*)w.ikn.p, (const float*)L.d_ikn.p, (float*)w.ikn.p,
                                           (uint16_t*)L.d_ik.p + (size_t)pos * (size_t)ID, (int64_t)1, 1, ID,
                                           (int)cfg.n_rot, (const float*)w.rope_cos.p, (const float*)w.rope_sin.p,
                                           (const int32_t*)w.pos1.p, eps, nullptr);
                    }
                    if (dbg) {
                        cu::sync();
                        cu::copy_d2h(iq.data(), w.iqn.p, (size_t)HI * ID * sizeof(float));
                        cu::copy_d2h(ik.data(), w.ikn.p, (size_t)ID * sizeof(float));
                        dbuf_put(pfx + "/index_q_after_rope", iq.data(), (int)(cfg.idx_heads * cfg.idx_dim));
                        dbuf_put(pfx + "/index_k_after_rope", ik.data(), (int)cfg.idx_dim);
                    }
                } else {
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
            }
            if (att_buf.size() < (size_t)cfg.n_head * (size_t)cfg.head_dim)
                att_buf.assign((size_t)cfg.n_head * (size_t)cfg.head_dim, 0.f);
            float* att = att_buf.data();
            // P10: keep the attention output on the card through the output projection.  Same kernels and the same
            // arithmetic; the D2H of `att` and the H2D of the projection's activation are simply not needed.
            const bool keep_att_dev = !dbg && kv_mode != 0 && L.want_gpu && L.wo.gpu &&
                                      gemv_dev_in_ok(L.wo, (int)(cfg.n_head * cfg.head_dim)) && W().att.p;
            float* att_dev = keep_att_dev ? (float*)W().att.p : nullptr;
            attend((int)li, L, pos, att, dev_tok ? (const float*)W().qn.p : nullptr,
                   dev_tok && L.indexer ? (const float*)W().iqn.p : nullptr, att_dev);
            if (dbg) dbuf_put(pfx + "/attn_core_out", att, (int)(cfg.n_head * cfg.head_dim));
            if (att_dev) gemv_dev_in(L.wo, att_dev, proj.data());
            else gemv(L.wo, att, proj.data());
            if (dbg) dbuf_put(pfx + "/attn_out", proj.data(), (int)cfg.n_embd);
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];

            rms(x.data(), L.ffn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            const auto t_ffn = tokclock::now();
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
            if (prefetch_slots > 0 && li + 1 < cfg.n_layers) {
                // The SSD lookahead is issued from inside moe() (it needs this layer's FFN input); only the VRAM
                // fill is left here, and it keeps its one-layer distance and its exact router input.
                const auto t_pf = std::chrono::steady_clock::now();
                prefetch_for(li + 1);
                if (pt) prof.prefetch += pms(t_pf);
            }
            if (dbg) dbuf_put(pfx + "/layer_out", x.data(), (int)cfg.n_embd);
            if (time_on) {
                const double f = ms(t_ffn);
                std::fprintf(stderr, "INFO token-time layer=%lld attn=%.2f ffn=%.2f moe=%d\n", (long long)li,
                             ms(t_layer) - f, f, L.moe ? 1 : 0);
            }
            if (pt)
                prof.attn += std::chrono::duration<double, std::milli>(t_ffn - t_layer).count();
        }

        const auto t_lm = std::chrono::steady_clock::now();
        rms(x.data(), out_norm.data(), h.data(), (int)cfg.n_embd, eps);
        if (output_gpu) use_device_index(layer_dev[(size_t)(cfg.n_layers - 1)]);
        gemv(output, h.data(), logit_out);
        if (pt) prof.lm_head = pms(t_lm);
        ++n_past;
        if (pt) {
            prof.total = pms(t_tok);
            prof_acc(prof_sum, prof);
            ++prof_tokens;
            const std::string line = prof_line(prof, prof_tokens);
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        if (tiers_on) {
            total_stats.vram += tok_stats.vram;
            total_stats.ram += tok_stats.ram;
            total_stats.ssd += tok_stats.ssd;
            total_stats.ssd_bytes += tok_stats.ssd_bytes;
            total_stats.skipped += tok_stats.skipped;
            total_stats.ssd_mapped += tok_stats.ssd_mapped;
            ++tokens_seen;
            if (opt.adaptive && (++rebalance_tick % 16) == 0) rebalance();
            if (opt.stats)
                std::fprintf(stderr,
                             "INFO minimax-stats token=%lld vram=%lld ram=%lld ssd=%lld ssd_bytes=%lld "
                             "ssd_wait_ms=%.1f skipped=%lld ssd_mapped=%lld swaps=%llu\n",
                             (long long)tokens_seen, (long long)tok_stats.vram, (long long)tok_stats.ram,
                             (long long)tok_stats.ssd, (long long)tok_stats.ssd_bytes, ssd_wait_ms,
                             (long long)tok_stats.skipped, (long long)tok_stats.ssd_mapped,
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
        std::vector<float> prob((size_t)nexp);
        std::vector<int> order;
        for (int t = 0; t < T; ++t) {
            const float* lg = logits + (size_t)t * nexp;
            for (int i = 0; i < nexp; ++i) prob[(size_t)i] = sigmoid(lg[i]);
            select_routes(L, prob, order, top);
            float sum = 0.f;
            for (int j = 0; j < top; ++j) {
                ids[(size_t)t * top + j] = order[(size_t)j];
                wts[(size_t)t * top + j] = prob[(size_t)order[(size_t)j]];
                sum += wts[(size_t)t * top + j];
            }
            if (renorm && sum > 0.f)
                for (int j = 0; j < top; ++j) wts[(size_t)t * top + j] /= sum;
            apply_expert_skip_range(wts.data() + (size_t)t * top, top);
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
        if (stream_direct && pf_expert_h.size() < (size_t)max_expert_bytes)
            pf_expert_h.resize((size_t)max_expert_bytes);
        for (int e = 0; e < nexp; ++e) {
            const std::vector<int>& toks = tok_of[(size_t)e];
            if (toks.empty()) continue;
            // P9: a resident expert is read from the pinned arena (P5c's rule for the device path), and a streamed
            // one through an unbuffered read instead of a fault through the 146 GB mapping.  Both hand the same
            // bytes to the same kernels, so the result is unchanged; only the page cache stops growing.
            const int rs = (!ram_inplace && L.ram_n > 0) ? L.ram_of[(size_t)e] : -1;
            const uint8_t* gb;
            const uint8_t* ub;
            const uint8_t* db;
            if (rs >= 0) {
                const uint8_t* base = ram_slot_ptr(const_cast<Layer&>(L), rs);
                gb = base;
                ub = base + L.expert_gu;
                db = base + 2 * L.expert_gu;
            } else if (stream_direct) {
                read_expert_roles(L, e, pf_expert_h.data());
                gb = pf_expert_h.data();
                ub = gb + L.expert_gu;
                db = gb + 2 * L.expert_gu;
            } else {
                gb = L.egate.host + (size_t)e * L.expert_gu;
                ub = L.eup.host + (size_t)e * L.expert_gu;
                db = L.edown.host + (size_t)e * L.expert_down;
            }
            for (size_t a = 0; a < toks.size(); a += (size_t)rows_cap) {
                const int nt = (int)std::min<size_t>(toks.size() - a, (size_t)rows_cap);
                for (int i = 0; i < nt; ++i)
                    std::memcpy(xin.data() + (size_t)i * embd, H + (size_t)toks[a + (size_t)i] * embd,
                                (size_t)embd * sizeof(float));
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

    // P16: time a pinned host -> device copy on each card (the ring slots of the live prompt workspace) and send
    // the routed prompt experts of a card whose link is 1.5x slower than the best to the best card.
    void resolve_pf_xdev() {
        const int n = (int)devs.size();
        pf_xdev.assign((size_t)n, 0);
        for (int d = 0; d < n; ++d) pf_xdev[(size_t)d] = d;
        const std::string& want = opt.prefill_expert_device;
        if (n < 2 || want == "own") return;
        const int back = cur;
        if (!want.empty() && want != "auto") {
            const int d = std::atoi(want.c_str());
            if (d < 0 || d >= n) fail("--minimax-prefill-expert-device: no device index " + want);
            for (int i = 0; i < n; ++i) pf_xdev[(size_t)i] = d;
        } else {
            std::vector<double> gbs((size_t)n, 0.0);
            for (int d = 0; d < n; ++d) {
                use_device_index(d);
                PfBuf& p = P();
                if (p.ring_v.empty() || p.pin_v.empty()) continue;
                const size_t bytes = std::min(p.ring_slot, p.pin_slot_bytes);
                cu::copy_h2d(p.ring_v[0], p.pin_v[0], bytes);   // warm-up
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < 4; ++r) cu::copy_h2d(p.ring_v[0], p.pin_v[0], bytes);
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                gbs[(size_t)d] = s > 0.0 ? 4.0 * (double)bytes / s / 1e9 : 0.0;
            }
            int best = 0;
            for (int d = 1; d < n; ++d)
                if (gbs[(size_t)d] > gbs[(size_t)best]) best = d;
            for (int d = 0; d < n; ++d)
                if (gbs[(size_t)best] >= 1.5 * gbs[(size_t)d]) pf_xdev[(size_t)d] = best;
            std::string line;
            for (int d = 0; d < n; ++d) {
                char b[96];
                std::snprintf(b, sizeof b, "%sdev=%d %.1f GB/s -> %d", d ? ", " : "", devs[(size_t)d], gbs[(size_t)d],
                              devs[(size_t)pf_xdev[(size_t)d]]);
                line += b;
            }
            std::fprintf(stderr, "INFO minimax-prefill-expert-device %s\n", line.c_str());
        }
        use_device_index(back);
    }

    void moe_batch_gpu(const Layer& L, const float* hdev, const uint16_t* h16, const float* hhost, int T,
                       float* yout_host) {
        const int nexp = (int)cfg.n_expert, top = (int)cfg.n_expert_used;
        const int embd = (int)cfg.n_embd, ff = (int)cfg.n_ff_exp;
        gemm_gpu(L.router, h16, hdev, P().y32, T, nexp);
        cu::copy_d2h(pf_router.data(), P().y32, (size_t)T * nexp * sizeof(float));
        // P16: the routed experts may run on another card (see pf_xdev); the router above and the shared expert
        // below stay on the layer's own.
        const int own = cur;
        if (pf_xdev.empty()) resolve_pf_xdev();
        const int xd = pf_xdev[(size_t)own];
        const bool remote = xd != own;
        const float* xin = hdev;   // the routed experts' input, on the card that runs them
        if (remote) {
            use_device_index(xd);
            PfBuf& r = P();
            cu::copy_h2d(r.p32, hhost, (size_t)T * embd * sizeof(float));
            xin = r.p32;
        }
        PfBuf& p = P();
        // A slot on the layer's own card is not reachable from the expert card: stream that expert too.
        auto slot_of = [&](int e) { return remote || L.vram_of.empty() ? -1 : L.vram_of[(size_t)e]; };
        std::vector<int> ids;
        std::vector<float> wts;
        pf_routes(L, pf_router.data(), T, ids, wts);

        pf_entries.clear();
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < top; ++s) {
                if (wts[(size_t)t * top + s] == 0.f) continue;   // --minimax-expert-skip
                pf_entries.push_back({t, ids[(size_t)t * top + s], wts[(size_t)t * top + s], 0});
            }
        std::stable_sort(pf_entries.begin(), pf_entries.end(),
                         [](const PfEntry& a, const PfEntry& b) { return a.expert < b.expert; });

        // Per projection, not per expert: the IQ2_XS pack has IQ2_XS gate/up and a Q2_0/IQ4_NL/F32 down, and MM-NVFP4
        // has its own type.  `gu_iq` / `d_iq` pick the int8 kernel or the dequantize-to-fp16 + cuBLAS fallback.
        const bool gu_iq = iq_expert_type(L.egate.type) && cfg.n_embd % 32 == 0;
        const bool d_iq = iq_expert_type(L.edown.type) && cfg.n_ff_exp % 32 == 0;
        const int gt = L.egate.type, dt = L.edown.type;
        // P10: the tensor-core path needs a whole 256-value block per row (the flat dequantizer's contract) on both
        // projections, and at least `tc_expert_min_rows` tokens in the group to pay for the weight dequantization.
        const bool tc_gu_ok = prefill_expert_gemm != "mmvq" && tc_expert_type(gt) && tc_expert_type(L.eup.type) &&
                              (cfg.n_embd % 256) == 0;
        const bool tc_d_ok = prefill_expert_gemm != "mmvq" && tc_expert_type(dt) && (cfg.n_ff_exp % 256) == 0;
        bool zero = true;
        std::vector<int32_t> row_t((size_t)rows_cap);
        std::vector<float> row_w((size_t)rows_cap);

        // ---- P5b: the staged expert ring.  The chunk's route entries are already sorted by expert, so each
        // expert's bytes are read once per chunk either way; what changes here is that the read is spread over the
        // CPU pool and the H2D copy of expert i+1 rides the copy stream while expert i's GEMMs run on the main
        // stream.  One slot (P5a) serialized read -> copy -> compute for all 128 experts of a chunk.
        struct Grp {
            int e;
            size_t at, end;
        };
        std::vector<Grp> grps;
        for (size_t q = 0; q < pf_entries.size();) {
            const int e = pf_entries[q].expert;
            size_t end = q;
            while (end < pf_entries.size() && pf_entries[end].expert == e) ++end;
            grps.push_back({e, q, end});
            q = end;
        }
        const int ring_n = p.ring_n > 0 ? p.ring_n : 1;
        const int lookahead = ring_n - 1;   // slots filled before the one being computed

        // One expert's [gate|up|down] into `dst`, split over the pool.  The three matrices are contiguous in the
        // destination, so this is a flat list of copies: 8 pieces of >=1 MiB each keep every core on the mmap
        // (whose page faults are what the single-threaded memcpy was waiting for) without a per-piece allocation.
        auto read_expert = [&](int e, uint8_t* dst) {
            const size_t gu = L.expert_gu, dn = L.expert_down;
            // P5c: a RAM-tier expert is read from the pinned arena, not through the mmap.  The batch path used
            // to ignore the tier and re-fault every page of every expert once per chunk.
            const int rs = L.ram_n > 0 ? L.ram_of[(size_t)e] : -1;
            const uint8_t* base = rs >= 0 && !ram_inplace ? ram_slot_ptr(const_cast<Layer&>(L), rs) : nullptr;
            const uint8_t* src[3] = {base ? base : L.egate.host + (size_t)e * gu,
                                     base ? base + gu : L.eup.host + (size_t)e * gu,
                                     base ? base + 2 * gu : L.edown.host + (size_t)e * dn};
            const size_t off[3] = {p.role_off[0], p.role_off[1], p.role_off[2]};
            const size_t len[3] = {gu, gu, dn};
            // `pool.run` runs inline when the job has fewer than 4 pieces per thread, so ask for enough pieces:
            // 4 per thread per matrix lands on ~60 KiB each at MiniMax's expert size.
            const int pieces = 4 * (threads > 0 ? threads : 4);
            pool.run(pieces * 3, [&](int a, int b) {
                for (int i = a; i < b; ++i) {
                    const int m = i / pieces, c = i % pieces;
                    const size_t p0 = len[m] * (size_t)c / (size_t)pieces;
                    const size_t p1 = len[m] * (size_t)(c + 1) / (size_t)pieces;
                    if (p1 > p0) std::memcpy(dst + off[m] + p0, src[m] + p0, p1 - p0);
                }
            });
        };
        // Stage streamed expert `si` (its index among the streamed groups) into slot si % ring_n.
        std::vector<int> streamed;   // group index of each streamed expert, in order
        for (size_t i = 0; i < grps.size(); ++i) {
            if (slot_of(grps[i].e) < 0) streamed.push_back((int)i);
        }
        // Read + queue expert number `si` into slot si % ring_n.  Called ahead of the compute loop.
        double t_disk = 0;                 // STRATA_MM_TIME: the direct-read side alone (the drive's rate)
        int64_t disk_bytes = 0;
        auto stage_si = [&](int si) {
            const int slot = si % ring_n;
            const Grp& g = grps[(size_t)streamed[(size_t)si]];
            cu::event_sync(p.ev_comp[(size_t)slot]);
            uint8_t* pin = (uint8_t*)p.pin_v[(size_t)slot];
            const uint8_t* src[3] = {nullptr, nullptr, nullptr};
            // P10: a RAM-tier expert is copied from the pinned arena; only a real miss is read unbuffered.  The old
            // `if (prefill_direct)` branch read every streamed expert from the file, including the ones the tier had
            // already pinned -- the SSD path replaced the faster RAM one for no reason.
            const int rs = (!ram_inplace && L.ram_n > 0) ? L.ram_of[(size_t)g.e] : -1;
            if (rs >= 0) {
                // The pinned tier already holds the bytes in kernel order: copy the three roles from there, no
                // staging copy.
                const uint8_t* base = ram_slot_ptr(const_cast<Layer&>(L), rs);
                src[0] = base;
                src[1] = base + L.expert_gu;
                src[2] = base + 2 * L.expert_gu;
            } else if (prefill_direct && !ssd_files.empty()) {
                const auto td0 = std::chrono::steady_clock::now();
                std::string err;
                if (!ssd_read_roles_direct(L, g.e, pin, src, err))
                    fail("batched prefill: " + err);
                t_disk += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                disk_bytes += (int64_t)L.expert_bytes;
            } else {
                read_expert(g.e, pin);
                src[0] = pin + p.role_off[0];
                src[1] = pin + p.role_off[1];
                src[2] = pin + p.role_off[2];
            }
            uint8_t* dst = (uint8_t*)p.ring_v[(size_t)slot];
            const size_t off[3] = {0, L.expert_gu, 2 * L.expert_gu};
            const size_t len[3] = {L.expert_gu, L.expert_gu, L.expert_down};
            for (int r = 0; r < 3; ++r) cu::copy_h2d_async(dst + off[r], src[r], len[r], p.copy_stream);
            cu::event_record(p.ev_copy[(size_t)slot], p.copy_stream);
        };
        int primed = 0;
        double t_read = 0, t_copywait = 0;   // STRATA_MM_TIME: the two sides of the staged ring
        auto pump = [&](int upto) {
            while (primed < upto && primed < (int)streamed.size()) {
                const auto t0 = std::chrono::steady_clock::now();
                stage_si(primed);
                t_read += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++primed;
            }
        };
        if (ring_n > 1 && !streamed.empty()) pump(std::min<int>(ring_n - 1, (int)streamed.size()));

        size_t at = 0;
        int n_vram = 0, n_ram = 0, n_disk = 0, n_tc = 0;
        while (at < pf_entries.size()) {
            const int e = pf_entries[at].expert;
            size_t end = at;
            while (end < pf_entries.size() && pf_entries[end].expert == e) ++end;
            const uint8_t* sg;
            const uint8_t* su;
            const uint8_t* sd;
            const int slot = slot_of(e);
            if (slot >= 0) ++n_vram;
            else if (!ram_inplace && L.ram_n > 0 && L.ram_of[(size_t)e] >= 0) ++n_ram;
            else ++n_disk;
            int ring_slot = -1;
            if (slot >= 0) {
                const uint8_t* base = (const uint8_t*)L.vram[(size_t)slot].mem.p;
                sg = base;
                su = base + L.expert_gu;
                sd = base + 2 * L.expert_gu;
            } else {
                if (!p.ring) fail("batched prefill: no expert streaming buffer");
                // This group's index among the streamed ones is its position in `streamed`; the pipeline keeps
                // `primed` at least one slot ahead, so waiting here only costs when the copy is the slow side.
                size_t si = 0;
                while (si < streamed.size() && grps[(size_t)streamed[si]].at != at) ++si;
                if (si >= streamed.size()) fail("internal: streamed expert is not in the order list");
                pump((int)si + 1);
                ring_slot = (int)si % ring_n;
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    cu::event_sync(p.ev_copy[(size_t)ring_slot]);
                    t_copywait +=
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
                uint8_t* dst = (uint8_t*)p.ring_v[(size_t)ring_slot];
                sg = dst;
                su = dst + L.expert_gu;
                sd = dst + 2 * L.expert_gu;
                pump((int)si + 1 + lookahead);
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
                const bool tc_gu = tc_gu_ok && (prefill_expert_gemm == "tc" || nt >= tc_expert_min_rows);
                const bool tc_d = tc_d_ok && (prefill_expert_gemm == "tc" || nt >= tc_expert_min_rows);
                if (tc_gu && a == at) ++n_tc;
                if (tc_gu) {
                    cu::gather_rows_f16(xin, (const int32_t*)p.d_rows.p, nt, embd, p.gx, nullptr);
                    gemm_expert_f16(gt, sg, L.egate.row_bytes, p.gx, p.gy, nt, ff, embd, ff);
                    gemm_expert_f16(L.eup.type, su, L.eup.row_bytes, p.gx, p.gu, nt, ff, embd, ff);
                } else if (gu_iq) {
                    cu::gather_rows_f32(xin, (const int32_t*)p.d_rows.p, nt, embd, p.gx32, nullptr);
                    strata::kernels::quantize_q8_1_rows(p.gx32, nt, embd, p.xq, nullptr);
                    strata::kernels::iq_mmvq(gt, sg, p.xq, p.gy, embd, ff, nt, nullptr);
                    strata::kernels::iq_mmvq(gt, su, p.xq, p.gu, embd, ff, nt, nullptr);
                } else if (prefill_dense_f32) {
                    cu::gather_rows_f32(xin, (const int32_t*)p.d_rows.p, nt, embd, p.gx32, nullptr);
                    gemm_expert_gpu(gt, sg, L.egate.row_bytes, nullptr, p.gx32, p.gy, nt, ff, embd);
                    gemm_expert_gpu(gt, su, L.egate.row_bytes, nullptr, p.gx32, p.gu, nt, ff, embd);
                } else {
                    cu::gather_rows_f16(xin, (const int32_t*)p.d_rows.p, nt, embd, p.gx, nullptr);
                    gemm_expert_gpu(gt, sg, L.egate.row_bytes, p.gx, nullptr, p.gy, nt, ff, embd);
                    gemm_expert_gpu(gt, su, L.egate.row_bytes, p.gx, nullptr, p.gu, nt, ff, embd);
                }
                if (fg != 1.f) cu::scale_rows(p.gy, (int64_t)nt * ff, fg, nullptr);
                if (fu != 1.f) cu::scale_rows(p.gu, (int64_t)nt * ff, fu, nullptr);
                cu::swiglu_rows_f32(p.gy, p.gu, p.gh32, nt, ff, alpha, limit, nullptr);
                if (tc_d) {
                    cu::f32_to_f16(p.gh32, p.gh, (int64_t)nt * ff, nullptr);
                    gemm_expert_f16(dt, sd, L.edown.row_bytes, p.gh, p.gd, nt, embd, ff, embd);
                } else if (d_iq) {
                    strata::kernels::quantize_q8_1_rows(p.gh32, nt, ff, p.hq, nullptr);
                    strata::kernels::iq_mmvq(dt, sd, p.hq, p.gd, ff, embd, nt, nullptr);
                } else if (prefill_dense_f32) {
                    gemm_expert_gpu(dt, sd, L.edown.row_bytes, nullptr, p.gh32, p.gd, nt, embd, ff);
                } else {
                    cu::f32_to_f16(p.gh32, p.gh, (int64_t)nt * ff, nullptr);
                    gemm_expert_gpu(dt, sd, L.edown.row_bytes, p.gh, nullptr, p.gd, nt, embd, ff);
                }
                if (fd != 1.f) cu::scale_rows(p.gd, (int64_t)nt * embd, fd, nullptr);
                cu::moe_combine_rows(p.gd, (const int32_t*)p.d_ent_row.p, (const float*)p.d_ent_w.p, nt, embd, p.sh,
                                     T, zero, nullptr);
                zero = false;
            }
            // The slot is free for a new copy once everything the main stream enqueued for it has run.
            if (ring_slot >= 0 && p.ev_comp.size() > (size_t)ring_slot)
                cu::event_record(p.ev_comp[(size_t)ring_slot], nullptr);
            at = end;
        }
        if (std::getenv("STRATA_MM_TIME"))
            std::fprintf(stderr,
                         "INFO batch-ring read=%.1f copywait=%.1f disk_ms=%.1f disk_mb=%.1f disk_gbs=%.2f ms "
                         "experts=%zu vram=%d ram=%d disk=%d tc=%d\n",
                         t_read, t_copywait, t_disk, (double)disk_bytes / 1048576.0,
                         t_disk > 0.0 ? (double)disk_bytes / (t_disk * 1.0e6) : 0.0, streamed.size(), n_vram, n_ram,
                         n_disk, n_tc);
        cu::sync();
        cu::copy_d2h(yout_host, p.sh, (size_t)T * embd * sizeof(float));
        if (remote) use_device_index(own);
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
            if ((size_t)(pos0 + t) < img_row.size()) {
                const int row = img_row[(size_t)(pos0 + t)];
                if (row >= 0)
                    std::memcpy(pf_h.data() + (size_t)t * ne,
                                img_data.data() + (size_t)row * (size_t)ne, (size_t)ne * sizeof(float));
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
        double t_idx = 0, t_topk = 0, t_msa = 0, t_prep = 0, t_kvw = 0;
        auto dur = [](pfclock::time_point a, pfclock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            const bool gpu = !devs.empty() && L.want_gpu;
            const auto t_layer = pfclock::now();
            const bool dev_attn = kv_mode == 1 || (prefill_kv_stage && !pf.empty() && P().kv_k.p && P().kv_v.p);
            if (gpu || kv_mode != 0) use_layer_device(li);
            PfBuf p_dummy;
            PfBuf& p = pf.empty() ? p_dummy : P();
            // ---- attention block
            const auto ta0 = pfclock::now();
            for (int t = 0; t < T; ++t)
                rms(pf_h.data() + (size_t)t * ne, L.attn_norm.data(), pf_q.data() + (size_t)t * ne, ne, eps);
            const float* hq = pf_q.data();   // [T][ne] normed input
            const auto ta1 = pfclock::now();
            // P5c: with the fused device path the projections never come back to the host: norm+rope runs on
            // the device and writes fp16 straight into the KV cache (and the MSA query buffer).
            const bool dev_prep = dev_attn && gpu_norm_rope && L.d_qn.p && cfg.n_rot > 0 &&
                                  (int64_t)pos0 + T <= p_rope_cap();
            if (gpu) {
                cu::copy_h2d(p.p32, hq, (size_t)T * ne * sizeof(float));
                cu::f32_to_f16(p.p32, p.h16, (size_t)T * ne, nullptr);
                cu::sync();
                gemm_gpu(L.wq, p.h16, p.p32, p.y32, T, qlen);
                if (!dev_prep) cu::copy_d2h(qf.data(), p.y32, (size_t)T * qlen * sizeof(float));
                else {
                    pf_pos.resize((size_t)T);
                    for (int t = 0; t < T; ++t) pf_pos[(size_t)t] = (int32_t)(pos0 + t);
                    cu::copy_h2d(p.d_pos.p, pf_pos.data(), (size_t)T * sizeof(int32_t));
                    cu::head_norm_rope(p.y32, (const float*)L.d_qn.p, nullptr, (uint16_t*)p.d_q16.p, (int64_t)T * H, H,
                                       D, (int)cfg.n_rot, (const float*)W().rope_cos.p,
                                       (const float*)W().rope_sin.p, (const int32_t*)p.d_pos.p, eps, nullptr);
                }
                gemm_gpu(L.wk, p.h16, p.p32, p.y32, T, klen);
                if (!dev_prep) cu::copy_d2h(pf_kv.data(), p.y32, (size_t)T * klen * sizeof(float));
                else {
                    uint16_t* kdst = (uint16_t*)p.a16;
                    cu::head_norm_rope(p.y32, (const float*)L.d_kn.p, nullptr, kdst, (int64_t)T * HKV, HKV, D,
                                       (int)cfg.n_rot, (const float*)W().rope_cos.p,
                                       (const float*)W().rope_sin.p, (const int32_t*)p.d_pos.p, eps, nullptr);
                    if (kv_mode == 1) {
                        cu::copy_d2d((uint16_t*)L.d_k.p + (size_t)pos0 * klen, kdst, (size_t)T * klen * 2);
                    } else {
                        pf_k16.resize((size_t)T * klen);
                        cu::copy_d2h(pf_k16.data(), kdst, (size_t)T * klen * 2);
                        store_kv16(k_host_q, k_host, ((size_t)li * (size_t)ctx + (size_t)pos0) * (size_t)klen,
                                   pf_k16.data(), (int)((size_t)T * (size_t)klen));
                    }
                }
                gemm_gpu(L.wv, p.h16, p.p32, p.y32, T, klen);
                if (!dev_prep) cu::copy_d2h(vf.data(), p.y32, (size_t)T * klen * sizeof(float));
                else {
                    uint16_t* vdst = (uint16_t*)p.a16;
                    cu::f32_to_f16(p.y32, vdst, (int64_t)T * klen, nullptr);
                    if (kv_mode == 1) {
                        cu::copy_d2d((uint16_t*)L.d_v.p + (size_t)pos0 * klen, vdst, (size_t)T * klen * 2);
                    } else {
                        pf_v16.resize((size_t)T * klen);
                        cu::copy_d2h(pf_v16.data(), vdst, (size_t)T * klen * 2);
                        store_kv16(v_host_q, v_host, ((size_t)li * (size_t)ctx + (size_t)pos0) * (size_t)klen,
                                   pf_v16.data(), (int)((size_t)T * (size_t)klen));
                    }
                }
                if (dev_prep) cu::sync();
            } else {
                gemm_cpu(L.wq, hq, T, ne, qf.data(), qlen);
                gemm_cpu(L.wk, hq, T, ne, pf_kv.data(), klen);
                gemm_cpu(L.wv, hq, T, ne, vf.data(), klen);
            }
            const auto ta2 = pfclock::now();
            const auto tkv0 = pfclock::now();
            if (!dev_prep)
            for (int t = 0; t < T; ++t) {
                rms_heads(qf.data() + (size_t)t * qlen, H, D, L.q_norm.data(), eps);
                rms_heads(pf_kv.data() + (size_t)t * klen, HKV, D, L.k_norm.data(), eps);
                rope(qf.data() + (size_t)t * qlen, H, D, (int)cfg.n_rot, (int)(pos0 + t), cfg.rope_theta);
                rope(pf_kv.data() + (size_t)t * klen, HKV, D, (int)cfg.n_rot, (int)(pos0 + t), cfg.rope_theta);
                const size_t koff = ((size_t)li * (size_t)ctx + (size_t)(pos0 + t)) * (size_t)klen;
                if (kv_mode != 1) {
                    store_kv(k_host_q, k_host, koff, pf_kv.data() + (size_t)t * klen, klen);
                    store_kv(v_host_q, v_host, koff, vf.data() + (size_t)t * klen, klen);
                }
            }
            if (kv_mode == 1 && !dev_prep) {
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
                const auto tix0 = pfclock::now();
                iqf.resize((size_t)T * ilen);
                ikf.resize((size_t)T * cfg.idx_dim);
                if (gpu) {
                    gemm_gpu(L.iq, p.h16, p.p32, p.y32, T, ilen);
                    if (!dev_prep) cu::copy_d2h(iqf.data(), p.y32, iqf.size() * sizeof(float));
                    else
                        cu::head_norm_rope(p.y32, (const float*)L.d_iqn.p, nullptr, (uint16_t*)p.d_iq16.p,
                                           (int64_t)T * HI, HI, (int)cfg.idx_dim, (int)cfg.n_rot,
                                           (const float*)W().rope_cos.p, (const float*)W().rope_sin.p,
                                           (const int32_t*)p.d_pos.p, eps, nullptr);
                    gemm_gpu(L.ik, p.h16, p.p32, p.y32, T, (int)cfg.idx_dim);
                    if (!dev_prep) cu::copy_d2h(ikf.data(), p.y32, ikf.size() * sizeof(float));
                    else
                        cu::head_norm_rope(p.y32, (const float*)L.d_ikn.p, nullptr,
                                           (uint16_t*)L.d_ik.p + (size_t)pos0 * cfg.idx_dim, (int64_t)T, 1,
                                           (int)cfg.idx_dim, (int)cfg.n_rot, (const float*)W().rope_cos.p,
                                           (const float*)W().rope_sin.p, (const int32_t*)p.d_pos.p, eps, nullptr);
                } else {
                    gemm_cpu(L.iq, hq, T, ne, iqf.data(), ilen);
                    gemm_cpu(L.ik, hq, T, ne, ikf.data(), (int)cfg.idx_dim);
                }
                if (!dev_prep)
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
                if (kv_mode != 0 && !dev_prep) {
                    std::vector<uint16_t> ik16((size_t)T * cfg.idx_dim);
                    for (int t = 0; t < T; ++t)
                        to_f16(ikf.data() + (size_t)t * cfg.idx_dim, ik16.data() + (size_t)t * cfg.idx_dim,
                               (int)cfg.idx_dim);
                    cu::copy_h2d((uint16_t*)L.d_ik.p + (size_t)pos0 * cfg.idx_dim, ik16.data(),
                                 (size_t)T * cfg.idx_dim * 2);
                }
                if (dev_attn && !dev_prep) {
                    iq16.resize((size_t)T * ilen);
                    for (int t = 0; t < T; ++t)
                        to_f16(iqf.data() + (size_t)t * ilen, iq16.data() + (size_t)t * ilen, ilen);
                }
                if (dev_prep) cu::sync();
                t_kvw += dur(tix0, pfclock::now());
            }
            if (dev_attn) {
                // Device attention: block scores on the device, top-k on the host (as the decode path does).
                int nb = 0;
                if (L.indexer) {
                    if (!dev_prep) cu::copy_h2d(p.d_iq16.p, iq16.data(), iq16.size() * 2);
                    const auto ti0 = pfclock::now();
                    nb = (int)((pos0 + T + BLK - 1) / BLK);
                    cu::idx_block_scores_batch(L.d_ik.p, (const uint16_t*)p.d_iq16.p, HI, (int)cfg.idx_dim, BLK, nb, T,
                                               pos0, (float*)p.d_bs.p, nullptr);
                    cu::sync();
                    const auto ti1 = pfclock::now();
                    host_bs.resize((size_t)T * HI * (size_t)nb);
                    cu::copy_d2h(host_bs.data(), p.d_bs.p, host_bs.size() * sizeof(float));
                    for (int t = 0; t < T; ++t) {
                        int nblk = 0;
                        pf_select_blocks(li, pos0 + t, host_bs.data() + (size_t)t * HI * (size_t)nb, nb,
                                         pf_blk.data() + (size_t)t * HI * pf_max_blk, nblk);
                        for (int h = 0; h < HI; ++h) pf_nblk[(size_t)t * HI + h] = nblk;
                    }
                    const auto ti2 = pfclock::now();
                    t_idx += dur(ti0, ti1);
                    t_topk += dur(ti1, ti2);
                    cu::copy_h2d(p.d_blk.p, pf_blk.data(), pf_blk.size() * sizeof(int32_t));
                    cu::copy_h2d(p.d_nblk.p, pf_nblk.data(), pf_nblk.size() * sizeof(int32_t));
                }
                if (!dev_prep) {
                    cu::copy_h2d(p.y32, qf.data(), (size_t)T * qlen * sizeof(float));
                    cu::f32_to_f16(p.y32, (uint16_t*)p.d_q16.p, (size_t)T * qlen, nullptr);
                }
                cu::sync();
                // P8: with the K/V in pinned RAM, stage this layer's used prefix on the card once and let the
                // device MSA kernel read it.  One copy of [pos0+T] rows per layer, not one gather per query.
                const void* kk = L.d_k.p;
                const void* vv = L.d_v.p;
                if (kv_mode == 2) {
                    const size_t kvb = (size_t)klen * sizeof(uint16_t);
                    const size_t used = (size_t)(pos0 + T) * kvb;
                    cu::copy_h2d(p.kv_k.p, (const uint8_t*)k_host + (size_t)li * (size_t)ctx * kvb, used);
                    cu::copy_h2d(p.kv_v.p, (const uint8_t*)v_host + (size_t)li * (size_t)ctx * kvb, used);
                    kk = p.kv_k.p;
                    vv = p.kv_v.p;
                }
                const auto tm0 = pfclock::now();
                cu::msa_attn_batch(kk, vv, (const uint16_t*)p.d_q16.p, (const int32_t*)p.d_blk.p,
                                   (const int32_t*)p.d_nblk.p, H, H / HKV, L.indexer ? H / HI : 1, D, BLK,
                                   L.indexer ? 0 : 1, pos0, T, pf_max_blk, p.p32, nullptr);
                cu::sync();
                t_msa += dur(tm0, pfclock::now());
                const auto tw0 = pfclock::now();
                if (!gpu) {
                    // P7: the KV is on the device but this layer's dense weights are not (--minimax-dense-layers-gpu
                    // 0, or a split that ends before this layer).  The device attention kernel still has to run --
                    // the host K/V does not exist -- but its output projection is the CPU's.
                    pf_att_h.resize((size_t)T * (size_t)qlen);
                    cu::copy_d2h(pf_att_h.data(), p.p32, pf_att_h.size() * sizeof(float));
                    gemm_cpu(L.wo, pf_att_h.data(), T, qlen, pf_moe_h.data(), ne);
                    t_prep += dur(tw0, pfclock::now());
                } else {
                    if (!prefill_dense_f32) cu::f32_to_f16(p.p32, p.a16, (size_t)T * qlen, nullptr);
                    cu::sync();
                    gemm_gpu(L.wo, prefill_dense_f32 ? p.h16 : p.a16, p.p32, p.y32, T, ne);
                    cu::copy_d2h(pf_moe_h.data(), p.y32, (size_t)T * ne * sizeof(float));
                    t_prep += dur(tw0, pfclock::now());
                }
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
                moe_batch_gpu(L, p.p32, p.h16, hn, T, pf_moe_h.data());
            } else {
                moe_batch_cpu(L, hn, T, pf_moe_h.data());
            }
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < ne; ++c) pf_h[(size_t)t * ne + c] += pf_moe_h[(size_t)t * ne + c];
            if (time_on) {
                const double f = ms(t_ffn);
                std::fprintf(stderr,
                             "INFO batch-time layer=%lld attn=%.1f ffn=%.1f moe=%d norm=%.1f qkv=%.1f rest=%.1f\n",
                             (long long)li, ms(t_layer) - f, f, L.moe ? 1 : 0, dur(ta0, ta1), dur(ta1, ta2),
                             (ms(t_layer) - f) - dur(ta0, ta2));
            }
            // P9: the layer read through the mapping (a dense weight on the CPU, the embedding table) leaves clean
            // pages behind; hand them back so the available figure stays where the budget put it.
            if (stream_direct && (li % 8) == 7) trim_mapped_pages();
        }
        if (time_on) {
            std::fprintf(stderr, "INFO batch-marks idx=%.1f topk=%.1f msa=%.1f prep=%.1f kvw=%.1f\n", t_idx, t_topk,
                         t_msa, t_prep, t_kvw);
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
                if ((int64_t)rows * nt > (int64_t)p.y32_elems) fail("output tile exceeds the workspace");
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
            total_stats.skipped += tok_stats.skipped;
            total_stats.ssd_mapped += tok_stats.ssd_mapped;
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
    o << " kv=" << (impl_->kv_q ? "q8_0" : "f16") << " norm_add1=" << (impl_->norm_add1 ? 1 : 0)
      << " threads=" << impl_->threads << " ctx=" << impl_->ctx
      << " host_attn=" << (impl_->host_attn_fast ? "fast" : "legacy")
      << " prefill_kv=" << (impl_->prefill_kv_stage ? "stage" : "host");    if (impl_->devs.size() > 1) {
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
    if (!impl_->tiers_on && impl_->total_stats.skipped == 0 && impl_->prof_tokens == 0) return std::string();
    std::ostringstream o;
    o << "tiers=on vram_slots_per_layer=" << impl_->vram_per_layer << " ram_experts_per_layer="
      << impl_->ram_per_layer << " tokens=" << impl_->tokens_seen << " vram_hits=" << impl_->total_stats.vram
      << " ram_hits=" << impl_->total_stats.ram << " ssd_reads=" << impl_->total_stats.ssd
      << " ssd_bytes=" << impl_->total_stats.ssd_bytes << " swaps=" << impl_->swaps
      << " skipped=" << impl_->total_stats.skipped
      << " ssd_mapped=" << impl_->total_stats.ssd_mapped
      << " ssd_pf_hit=" << impl_->ssd_pf_hit << " ssd_pf_waste=" << impl_->ssd_pf_waste;
    if (impl_->rb_picks > 0)
        o << " route_swapped=" << (100.0 * (double)impl_->rb_swapped / (double)impl_->rb_picks)
          << "% route_weight_kept="
          << (impl_->rb_wden > 0 ? 100.0 * impl_->rb_wnum / impl_->rb_wden : 100.0) << "%";
    if (impl_->prof_tokens > 0)
        o << " | mean of " << impl_->prof_tokens << " tokens: "
          << Impl::prof_line(Impl::prof_mean(impl_->prof_sum, impl_->prof_tokens), impl_->prof_tokens).substr(5);
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
    // P12: the batched workspace (and the slots it borrows) lives for the teacher-forced pass only.
    Impl::PrefillWs ws(impl_);
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
                                const std::function<void(int, int, double)>& on_pp,
                                const std::vector<std::string>& images) {
    GenerateResult gr;
    if (prompt.empty()) fail("prompt is empty");
    if ((int64_t)prompt.size() > impl_->ctx) fail("prompt is longer than the context");
    if (max_new < 0) max_new = 0;
    if (sp.have_seed) impl_->rng.seed(sp.seed);
    else {
        std::random_device rd;
        impl_->rng.seed(((uint64_t)rd() << 32) ^ rd());
    }

    // P6: encode every picture, then line its rows up with the `<]image[>` placeholders in prompt order.
    // The server expands one placeholder per token; a caller that leaves a single token per picture is refused
    // rather than silently reading the wrong rows.
    impl_->img_row.clear();
    impl_->img_data.clear();
    impl_->img_rows = 0;
    if (!images.empty()) {
        if (!impl_->vision) fail("an image was sent but no vision encoder is loaded (pass --mmproj)");
        std::vector<int64_t> per;
        int64_t total = 0;
        for (const std::string& path : images) {
            ImageEmbeddings e = impl_->vision->encode_file(path, impl_->opt.image_max_tokens);
            if (e.dim != impl_->cfg.n_embd)
                fail("the mmproj projects to " + std::to_string(e.dim) + " but the model needs " +
                     std::to_string(impl_->cfg.n_embd));
            per.push_back(e.tokens);
            total += e.tokens;
            impl_->img_data.insert(impl_->img_data.end(), e.data.begin(), e.data.end());
        }
        impl_->img_rows = total;
        impl_->img_row.assign(prompt.size(), -1);
        int64_t row = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            if (prompt[i] == impl_->opt.image_token_id) {
                if (row >= total) fail("the prompt has more image tokens than the pictures supply");
                impl_->img_row[i] = (int)row++;
            }
        }
        if (row != total)
            fail("the prompt has " + std::to_string(row) + " image tokens but " + std::to_string(images.size()) +
                 " picture(s) produce " + std::to_string(total) + " (each picture is expanded to its own tokens)");
        // A picture's rows are request data, not cache: always prefill this prompt again.
        impl_->n_past = 0;
        impl_->held.clear();
        impl_->have_logits = false;
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
    // P12: the batched-prompt workspace borrows the decode expert slots for the prompt and returns them when it
    // ends, so decode starts with the decode-only slot count.  The scope is the prompt: the brace closes before the
    // decode loop, and the guard releases on every early return.
    {
        Impl::PrefillWs ws(impl_);
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
    }   // the prompt workspace's scope
    gr.prompt_ms = ms_since(t0);
    if (!impl_->have_logits) {
        gr.finish = "length";
        return gr;    }

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
