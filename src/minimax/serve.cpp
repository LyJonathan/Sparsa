// src/minimax/serve.cpp - MiniMax-M3.1-Flash inspect, teacher-forced eval, and the serve line protocol.
//
// Inspect never constructs the engine (no CUDA). Serve and eval load the GGUF, refuse a non-M3.1 file unless
// --minimax-allow-nonstandard is set, then build the forward pass. GEN streams T / PP / DONE. GENI and BGEN
// answer with ERR and no DONE. Unknown --flags are ignored only in --serve, matching the shared server config.
// An unasked context is min(model, 32768); only --max-context raises it.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <errno.h>
#include <poll.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"

#include "strata/kernels/cpu/iqk_quants.hpp"
#include "strata/kernels/iqk_types.hpp"
#include "strata/minimax/engine.hpp"
#include "strata/minimax/forward.hpp"
#include "strata/minimax/gguf.hpp"
#include "strata/minimax/vision.hpp"

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace strata::minimax {
namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: strata-minimax --model GGUF --minimax-inspect\n"
                 "       strata-minimax --model GGUF --serve [--max-context N] [--minimax-dense-layers-gpu N]\n"
                 "                       (default context is min(model, 32768); --max-context is the only way up)\n"
                 "       strata-minimax --model GGUF --minimax-eval IDS --minimax-dump-logits OUT.npy\n"
                 "                       [--minimax-last-k K] [--minimax-dense-layers-gpu N]\n"
                 "       (--minimax-allow-nonstandard serves a reduced test GGUF; --serve otherwise refuses it)\n"
                 "Prefill is token-by-token through the decode path. Dense layers 0..N-1 go to the GPU;\n"
                 "N=0 keeps every dense matrix on the CPU. Routed experts stay on the CPU unless the P3 tiers\n"
                 "are enabled: --minimax-tiers (auto sizes), --minimax-profile PATH, or an explicit slot budget.\n"
                 "  --minimax-vram-slots-per-layer N  VRAM hot-expert slots per layer (-1 auto, 0 none)\n"
                 "  --minimax-ram-experts-per-layer N pinned-RAM experts per layer (-1 auto, 0 none)\n"
                 "  --vram-reserve-mib N              VRAM kept free (default 700)\n"
                 "  --memory-limit-mib N              pinned-tier budget (default: sized from the box)\n"
                 "  --minimax-ram-headroom-mib N      RAM kept out of the pinned tier (default: 1/16 of physical, 3-10 GiB)\n"
                 "  --minimax-ram-mode pinned|inplace pinned: copy the ranked experts into locked RAM (default);\n"
                 "                                    inplace: no copy, the OS page cache is the tier (Maya)\n"
                 "  --minimax-profile PATH            STRP profile for the static fill\n"
                 "  --minimax-profile-save PATH       record routing and write an STRP profile on exit\n"
                 "  --minimax-routing-counts PATH     also write a text 'layer expert count' table\n"
                 "  --minimax-stats [--minimax-stats-per-layer]   tier counters on stderr\n"
                 "  --minimax-profile-time            one line per decode token: the wall time of every phase\n"
                 "  --no-minimax-adaptive             disable decayed VRAM swaps and the RAM-tier LRU\n"
                 "  --minimax-dump-dir DIR            write per-layer activations as npy (parity probes)\n"
                 "  --minimax-dump-layers LIST|all    which source layers to dump (default all)\n"
                 "  --minimax-seed-hidden FILE.npy    feed row t as token t's input instead of the embedding\n"
                 "  --gpus 0,1                       CUDA devices to use (default: 0)\n"
                 "  --layer-split auto|K[,K2]         layer K is the first one on the next GPU (auto: by free VRAM)\n"
                 "  --minimax-kv host|vram|ram|auto   where K/V live; vram|ram run MSA on the owning device\n"
                 "  --minimax-kv-quant f16|q8_0|auto  fp16 (default) or q8_0 blocks for the host/pinned K/V;\n"
                 "                                    auto picks q8_0 only when f16 does not fit\n"
                 "  --minimax-host-attn fast|legacy   multithreaded host attention (default fast; same arithmetic)\n"
                 "  --minimax-prefill-kv auto|off     with the KV in RAM, stage one layer's K/V on the card for the\n"
                 "                                    batched prompt pass (default auto, the long-context path)\n"
                 "  --minimax-prefill-batch N         process prompt tokens N at a time in batched passes (0 = per token)\n"
                 "  --minimax-prefill-experts auto|cpu|gpu  where a batched chunk's routed experts are computed\n"
                 "  --minimax-prefill-expert-gemm mmvq|tc|auto\n"
                 "                                    routed experts: int8 iq_mmvq dots (default), dequantize-once +\n"
                 "                                    FP16 tensor-core GEMM per expert, or auto (tc from 16 rows)\n"
                 "  --minimax-prefill-read mmap|direct    cached expert reads (default) or unbuffered 16 MiB reads\n"
                 "  --minimax-expert-prefetch N       decode: copy the next layer's N hottest missed experts ahead\n"
                 "  --minimax-ssd-chunks N            pieces one SSD expert read is split into, in flight together\n"
                 "                                    (default 1, 1-16); try 4-8 for a drive that prefers\n"
                 "                                    many small requests\n"
                 "  --minimax-expert-skip-ssd N       drop a route at or past rank N (0 = best) when its expert would\n"
                 "                                    be read from the SSD; 0 (default) keeps every route\n"
                 "  --minimax-ssd-zerocopy on|off     an SSD-tier expert whose pages are already resident is computed\n"
                 "                                    from the mapping with no read (default on)\n"
                 "  --minimax-page-probe N            pages sampled per expert for that check (default 32; 0 = all)\n"
                 "  --minimax-ssd-prefetch on|off     queue a future layer's SSD misses while this one computes\n"
                 "                                    (default off; pair with --minimax-ssd-prefetch-depth)\n"
                 "  --minimax-ssd-prefetch-depth N    layers between issuing a predicted read and using it\n"
                 "                                    (default 1, 1-8); the read overwinters depth layers of compute\n"
                 "  --minimax-route-bias on|off       rank the router's top-k on score + a residency bonus / file\n"
                 "                                    penalty (Xolotl); weights stay unbiased; default off\n"
                 "  --minimax-route-bias-in A         bonus for a resident expert (default 0.25)\n"
                 "  --minimax-route-bias-out C        penalty for an expert in neither tier (default 0.15)\n"
                 "  --minimax-route-protect P         the exact top-P experts are never displaced (default 2)\n"
                 "  --minimax-route-pool M            a pick must stay inside the exact top-M (default 12)\n"
                 "  --minimax-route-margin D          nudge only experts within D of the cutoff (default 0 = all)\n"
                 "  --minimax-pool-spin-us N          CPU pool workers spin N us before sleeping (default 0)\n"
                 "  --mmproj PATH                     vision encoder (llama.cpp clip mmproj, minimax_m3)\n"
                 "  --minimax-vision IMAGE            encode IMAGE and write --minimax-vision-out (no LM loaded)\n"
                 "  --minimax-vision-pixels PV.npy    encode preprocessed pixel_values [n,1176] instead of a picture\n"
                 "  --minimax-vision-size WxH         the resized size that goes with --minimax-vision-pixels\n");
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

/// A float32 .npy with a 2-D shape (the parity harness's pixel_values).  Header-only subset of the format.
bool read_npy_2d(const std::string& path, std::vector<float>& out, int64_t& rows, int64_t& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6] = {};
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    char ver[2] = {};
    f.read(ver, 2);
    uint32_t hlen = 0;
    if ((unsigned char)ver[0] == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header((size_t)hlen, '\0');
    f.read(header.data(), (std::streamsize)hlen);
    const auto lp = header.find("(");
    const auto rp = header.find(")", lp == std::string::npos ? 0 : lp);
    if (lp == std::string::npos || rp == std::string::npos) return false;
    std::string dims = header.substr(lp + 1, rp - lp - 1);
    for (char& c : dims)
        if (c == ',') c = ' ';
    std::istringstream is(dims);
    if (!(is >> rows >> cols)) return false;
    if (rows <= 0 || cols <= 0) return false;
    out.resize((size_t)rows * (size_t)cols);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(out.size() * sizeof(float)));
    return f.good() || f.eof();
}

int parse_chunk(const std::string& s) {
    if (s == "auto" || s == "AUTO") return 16;
    return std::atoi(s.c_str());
}

struct LineQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;

    void push(std::string s) {
        {
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(std::move(s));
        }
        cv.notify_one();
    }
    // false on EOF with nothing left.
    bool pop(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return eof || !q.empty(); });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_id_token(const std::string& tok, std::vector<int32_t>& ids, std::string& err) {
    std::string cur;
    auto flush = [&](const std::string& piece) -> bool {
        if (piece.empty()) return true;
        char* end = nullptr;
        const long v = std::strtol(piece.c_str(), &end, 10);
        if (end == piece.c_str() || *end != '\0') {
            err = "not a token id: " + piece;
            return false;
        }
        ids.push_back((int32_t)v);
        return true;
    };
    for (char c : tok) {
        if (c == ',') {
            if (!flush(cur)) return false;
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    return flush(cur);
}

// STDIN bytes already pulled and not yet split into lines. Only the reader thread touches this.
struct StdinBuf {
    std::string pending;
    bool eof = false;
};

// Append bytes the OS already has. A pipe drain does not block. A console has no byte count, so this is a no-op
// there and the poll loop below waits for the next key. false means the handle is closed.
bool slurp_ready(StdinBuf& in) {
    if (in.eof) return false;
#if defined(_WIN32)
    const HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    if (h == nullptr || h == INVALID_HANDLE_VALUE) {
        in.eof = true;
        return false;
    }
    DWORD avail = 0;
    if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr)) {
        const DWORD err = GetLastError();
        if (err == ERROR_INVALID_FUNCTION) return true;
        in.eof = true;
        return false;
    }
    if (avail == 0) return true;
    const size_t at = in.pending.size();
    in.pending.resize(at + (size_t)avail);
    DWORD n = 0;
    if (!ReadFile(h, in.pending.data() + at, avail, &n, nullptr)) {
        in.pending.resize(at);
        in.eof = true;
        return false;
    }
    in.pending.resize(at + (size_t)n);
    if (n == 0) in.eof = true;
    return n > 0;
#else
    pollfd pfd{};
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    const int r = ::poll(&pfd, 1, 0);
    if (r < 0) {
        if (errno == EINTR) return true;
        in.eof = true;
        return false;
    }
    if (r == 0) return true;
    char buf[4096];
    const ssize_t n = ::read(STDIN_FILENO, buf, sizeof buf);
    if (n < 0) {
        if (errno == EINTR) return true;
        in.eof = true;
        return false;
    }
    if (n == 0) {
        in.eof = true;
        return false;
    }
    in.pending.append(buf, buf + n);
    return true;
#endif
}

bool take_line(StdinBuf& in, std::string& line) {
    const size_t p = in.pending.find('\n');
    if (p == std::string::npos) {
        if (!in.eof || in.pending.empty()) return false;
        line = std::move(in.pending);
        in.pending.clear();
    } else {
        line.assign(in.pending, 0, p);
        in.pending.erase(0, p + 1);
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

// Block until a line, shutdown, or a sweep request. A sweep means "the bytes already in the pipe are part of
// the script that was written before this GEN"; the caller ignores a STOP among them.
bool wait_input(StdinBuf& in, const std::atomic<bool>& shutdown, const std::atomic<int>& sweep) {
    for (;;) {
        if (shutdown.load(std::memory_order_relaxed) || sweep.load(std::memory_order_relaxed) == 1) return false;
        if (!slurp_ready(in)) return false;
        if (in.pending.find('\n') != std::string::npos || (in.eof && !in.pending.empty())) return true;
        if (in.eof) return false;
#if defined(_WIN32)
        const HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
        DWORD avail = 0;
        if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr)) {
            if (GetLastError() == ERROR_INVALID_FUNCTION) {
                const DWORD w = WaitForSingleObject(h, 50);
                if (w == WAIT_TIMEOUT) continue;
                if (w != WAIT_OBJECT_0) {
                    in.eof = true;
                    return false;
                }
                char ch = 0;
                DWORD n = 0;
                if (!ReadFile(h, &ch, 1, &n, nullptr) || n == 0) {
                    in.eof = true;
                    return false;
                }
                in.pending.push_back(ch);
                continue;
            }
            in.eof = true;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
#else
        pollfd pfd{};
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;
        const int r = ::poll(&pfd, 1, 50);
        if (r < 0 && errno != EINTR) {
            in.eof = true;
            return false;
        }
#endif
    }
}

// gguf-py can dequantize Q4_K and Q6_K but does not quantize them, and IQ2_XS has no Python quantizer at all.
// This writes one ggml_quantize_chunk blob. IQ2_XS gets a uniform importance row of 1s; K-quants do not need one.
// The formats ported from ik_llama.cpp (IQ2_KT, IQ3_K, IQ4_KS, IQ3_KT, IQ2_KL) go through the engine's own codec
// (src/kernels/cpu/iqk_quants.cpp) with a uniform importance row, which is also what the synthetic fixtures use.
bool quant_type_from_name(const std::string& name, ggml_type& type) {
    if (name == "q4_k") type = GGML_TYPE_Q4_K;
    else if (name == "q6_k") type = GGML_TYPE_Q6_K;
    else if (name == "q8_0") type = GGML_TYPE_Q8_0;
    else if (name == "iq2_xs") type = GGML_TYPE_IQ2_XS;
    else if (name == "iq2_kt") type = (ggml_type) strata::kernels::ikq::kIQ2KT;
    else if (name == "iq3_k") type = (ggml_type) strata::kernels::ikq::kIQ3K;
    else if (name == "iq4_ks") type = (ggml_type) strata::kernels::ikq::kIQ4KS;
    else if (name == "iq3_kt") type = (ggml_type) strata::kernels::ikq::kIQ3KT;
    else if (name == "iq2_kl") type = (ggml_type) strata::kernels::ikq::kIQ2KL;
    else return false;
    return true;
}

int pack_quant(const std::string& type_name, int64_t rows, int64_t cols, const std::string& in_path,
               const std::string& out_path) {
    ggml_type type = GGML_TYPE_F32;
    if (!quant_type_from_name(type_name, type)) {
        std::fprintf(stderr, "strata-minimax: quant pack type must be q4_k, q6_k, q8_0, iq2_xs, iq2_kt, iq3_k, "
                             "iq4_ks, iq3_kt or iq2_kl\n");
        return 2;
    }
    const bool ikq = strata::kernels::ikq::is_iqk_type((int) type);
    const int64_t block = ikq ? 256 : ggml_blck_size(type);
    if (rows < 1 || cols < 1 || block < 1 || cols % block != 0) {
        std::fprintf(stderr, "strata-minimax: %s pack needs rows > 0 and cols a positive multiple of %lld\n",
                     type_name.c_str(), (long long)block);
        return 2;
    }
    std::ifstream in(in_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "strata-minimax: cannot read %s\n", in_path.c_str());
        return 1;
    }
    std::vector<float> src((size_t)rows * (size_t)cols);
    in.read(reinterpret_cast<char*>(src.data()), (std::streamsize)(src.size() * sizeof(float)));
    if (in.gcount() != (std::streamsize)(src.size() * sizeof(float))) {
        std::fprintf(stderr, "strata-minimax: short read of f32 weights for quant pack\n");
        return 1;
    }
    std::vector<float> imatrix;
    const float* im = nullptr;
    if (ikq || ggml_quantize_requires_imatrix(type)) {
        imatrix.assign((size_t)cols, 1.f);
        im = imatrix.data();
    }
    std::vector<uint8_t> dst;
    size_t row_bytes = 0;
    if (ikq) {
        row_bytes = strata::kernels::ikq::row_bytes((int) type, cols);
        if (row_bytes == 0) {
            std::fprintf(stderr, "strata-minimax: %s cannot hold %lld values in a row\n", type_name.c_str(),
                         (long long)cols);
            return 2;
        }
        dst.resize(row_bytes * (size_t)rows);
        try {
            strata::kernels::ikq::quantize_rows((int) type, src.data(), dst.data(), rows, cols, im, 1);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-minimax: %s quantize failed: %s\n", type_name.c_str(), e.what());
            return 1;
        }
    } else {
        ggml_quantize_init(type);
        row_bytes = ggml_row_size(type, cols);
        dst.resize(row_bytes * (size_t)rows);
        const size_t got = ggml_quantize_chunk(type, src.data(), dst.data(), 0, rows, cols, im);
        if (got != dst.size()) {
            std::fprintf(stderr, "strata-minimax: %s quantize wrote %zu bytes, expected %zu\n", type_name.c_str(),
                         got, dst.size());
            return 1;
        }
    }
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "strata-minimax: cannot write %s\n", out_path.c_str());
        return 1;
    }
    out.write(reinterpret_cast<const char*>(dst.data()), (std::streamsize)dst.size());
    if (!out) {
        std::fprintf(stderr, "strata-minimax: short write of packed quant weights\n");
        return 1;
    }
    return 0;
}

// The reference side of the synthetic tests: packed ikq rows -> f32, one row at a time through the engine's own
// dequantizer.  A Python reference cannot read these types (gguf-py has no codec for them).
int unpack_dequant(const std::string& type_name, int64_t rows, int64_t cols, const std::string& in_path,
                   const std::string& out_path) {
    ggml_type type = GGML_TYPE_F32;
    if (!quant_type_from_name(type_name, type) || !strata::kernels::ikq::is_iqk_type((int) type)) {
        std::fprintf(stderr, "strata-minimax: dequant type must be one of the ik_llama.cpp formats\n");
        return 2;
    }
    const size_t row_bytes = strata::kernels::ikq::row_bytes((int) type, cols);
    if (rows < 1 || cols < 1 || row_bytes == 0) {
        std::fprintf(stderr, "strata-minimax: bad dequant shape\n");
        return 2;
    }
    std::ifstream in(in_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "strata-minimax: cannot read %s\n", in_path.c_str());
        return 1;
    }
    std::vector<uint8_t> raw(row_bytes * (size_t)rows);
    in.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)raw.size());
    if (in.gcount() != (std::streamsize)raw.size()) {
        std::fprintf(stderr, "strata-minimax: short read of packed quant weights\n");
        return 1;
    }
    std::vector<float> out((size_t)rows * (size_t)cols);
    try {
        for (int64_t r = 0; r < rows; ++r)
            strata::kernels::ikq::dequantize_row((int) type, raw.data() + (size_t)r * row_bytes,
                                                out.data() + (size_t)r * (size_t)cols, cols);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-minimax: %s dequantize failed: %s\n", type_name.c_str(), e.what());
        return 1;
    }
    std::ofstream f(out_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "strata-minimax: cannot write %s\n", out_path.c_str());
        return 1;
    }
    f.write(reinterpret_cast<const char*>(out.data()), (std::streamsize)(out.size() * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "strata-minimax: short write of f32 weights\n");
        return 1;
    }
    return 0;
}

// sweep: 0 idle, 1 the reader should fold already-buffered stdin into the queue, 2 that fold is done.
struct GenGate {
    std::atomic<bool>* stop = nullptr;
    std::atomic<int>* sweep = nullptr;
    std::atomic<bool>* reader_done = nullptr;
    std::mutex* mu = nullptr;
    std::condition_variable* cv = nullptr;

    // A STOP already sitting in the pipe (the test script writes the whole session before READY) is applied
    // by the reader during the sweep and then cleared here. A STOP that arrives after the sweep still cancels.
    // If the reader already drained stdin and exited, that STOP was applied too, so clearing it is enough.
    void ignore_buffered_stop() {
        sweep->store(1, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lk(*mu);
        cv->wait(lk, [&] {
            return sweep->load(std::memory_order_relaxed) == 2 || reader_done->load(std::memory_order_relaxed);
        });
        sweep->store(0, std::memory_order_relaxed);
        stop->store(false, std::memory_order_relaxed);
    }
};

void handle_gen(Engine& engine, const std::string& line, GenGate& gate, bool greedy_default, bool print_stats) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    int max_new = 0;
    if (!(in >> max_new) || max_new < 0) {
        std::printf("ERR minimax engine: GEN needs a non-negative max_new\n");
        std::fflush(stdout);
        return;
    }
    SampleParams sp;
    if (greedy_default) sp.temperature = 0.f;
    std::vector<int32_t> ids;
    std::vector<std::string> images;
    bool ids_started = false;
    std::string tok;
    while (in >> tok) {
        const auto eq = tok.find('=');
        if (!ids_started && eq != std::string::npos) {
            const std::string key = tok.substr(0, eq);
            const std::string val = tok.substr(eq + 1);
            if (key == "temperature") sp.temperature = std::strtof(val.c_str(), nullptr);
            else if (key == "top_p") sp.top_p = std::strtof(val.c_str(), nullptr);
            else if (key == "min_p") sp.min_p = std::strtof(val.c_str(), nullptr);
            else if (key == "top_k") sp.top_k = std::atoi(val.c_str());
            else if (key == "penalty_last_n") sp.penalty_last_n = std::atoi(val.c_str());
            else if (key == "penalty_repeat") sp.penalty_repeat = std::strtof(val.c_str(), nullptr);
            else if (key == "penalty_freq") sp.penalty_freq = std::strtof(val.c_str(), nullptr);
            else if (key == "penalty_present") sp.penalty_present = std::strtof(val.c_str(), nullptr);
            else if (key == "seed") {
                sp.seed = std::strtoull(val.c_str(), nullptr, 10);
                sp.have_seed = true;
            } else if (key == "img") {
                size_t at = 0;
                while (at <= val.size()) {
                    const size_t comma = val.find(',', at);
                    const std::string one = val.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                    if (!one.empty()) images.push_back(one);
                    if (comma == std::string::npos) break;
                    at = comma + 1;
                }
            }
            continue;
        }
        ids_started = true;
        std::string err;
        if (!parse_id_token(tok, ids, err)) {
            std::printf("ERR minimax engine: %s\n", err.c_str());
            std::fflush(stdout);
            return;
        }
    }
    if (ids.empty()) {
        std::printf("ERR minimax engine: GEN has an empty prompt\n");
        std::fflush(stdout);
        return;
    }
    gate.ignore_buffered_stop();
    try {
        const GenerateResult gr = engine.generate(
            ids, max_new, sp, *gate.stop,
            [](int32_t id) {
                std::printf("T %d\n", (int)id);
                std::fflush(stdout);
            },
            [](int done, int total, double ms) {
                const double tps = ms > 0.0 ? (done * 1000.0 / ms) : 0.0;
                std::printf("PP %d %d %.3f %.3f\n", done, total, ms, tps);
                std::fflush(stdout);
            },
            images);
        std::printf("DONE %d %d %.3f %.3f %s 0 0 %d\n", gr.generated, gr.prompt, gr.prompt_ms, gr.decode_ms,
                    gr.finish.c_str(), gr.reused);
        std::fflush(stdout);
        if (print_stats) {
            const std::string st = engine.stats();
            if (!st.empty()) std::fprintf(stderr, "INFO minimax-stats total %s\n", st.c_str());
        }
    } catch (const std::exception& e) {
        std::printf("ERR minimax engine: %s\n", e.what());
        std::fflush(stdout);
    }
}

}  // namespace

int run(int argc, char** argv) {
    std::string model, eval_path, dump_path;
    int64_t ctx_asked = 0;
    int dense_gpu = -1;
    int cpu_threads = 0;
    int pool_workers = 0;
    int chunk = 16;
    int last_k = 0;
    bool serve_mode = false, inspect = false, allow_nonstandard = false, greedy = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--serve") serve_mode = true;
    bool tiers = false;
    std::string profile, profile_save, routing_counts;
    int vram_slots = -1, ram_experts = -1, vram_reserve_mib = 700, memory_limit_mib = 0, ram_headroom_mib = 0;
    int pool_spin_us = 0;
    bool stats = false, stats_per_layer = false, profile_time = false, adaptive = true;
    bool pack = false, unpack = false;
    int64_t pack_rows = 0, pack_cols = 0;
    std::string pack_type, pack_in, pack_out;
    std::string dump_dir, dump_layers, seed_hidden;
    std::string layer_split, gpus, kv_space;
    std::string kv_quant, host_attn, prefill_kv;
    int prefill_batch = 0, expert_prefetch = 0;
    std::string prefill_experts, prefill_dense, prefill_expert_gemm, prefill_read, dense_mmvq;
    std::string gpu_normrope, expert_resident;
    float expert_skip = 0.f;
    int ssd_chunks = 1, expert_skip_ssd = 0;
    std::string ssd_zerocopy;
    int page_probe = 32;
    std::string ssd_prefetch;
    int ssd_prefetch_depth = 1;
    std::string ram_mode;
    std::string route_bias;
    float route_bias_in = 0.25f, route_bias_out = 0.15f, route_margin = 0.f;
    int route_protect = 2, route_pool = 12;
    std::string graphs;
    std::string mmproj, vision_img, vision_out, vision_pixels, vision_size, vision_pixels_out;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-minimax: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--minimax-inspect" || a == "--inspect") inspect = true;
        else if (a == "--minimax-allow-nonstandard") allow_nonstandard = true;
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx_asked = std::atoll(next().c_str());
        else if (a == "--minimax-dense-layers-gpu") dense_gpu = std::atoi(next().c_str());
        else if (a == "--minimax-eval") eval_path = next();
        else if (a == "--minimax-dump-logits") dump_path = next();
        else if (a == "--minimax-last-k") last_k = std::atoi(next().c_str());
        else if (a == "--minimax-dump-dir") dump_dir = next();
        else if (a == "--minimax-dump-layers") dump_layers = next();
        else if (a == "--minimax-seed-hidden") seed_hidden = next();
        else if (a == "--layer-split") layer_split = next();
        else if (a == "--gpus" || a == "--gpu") gpus = next();
        else if (a == "--minimax-kv" || a == "--kv-space") kv_space = next();
        else if (a == "--minimax-kv-quant" || a == "--kv-quant") kv_quant = next();
        else if (a == "--minimax-host-attn" || a == "--host-attn") host_attn = next();
        else if (a == "--minimax-prefill-kv") prefill_kv = next();
        else if (a == "--minimax-prefill-batch" || a == "--minimax-batch") prefill_batch = std::atoi(next().c_str());
        else if (a == "--minimax-prefill-experts") prefill_experts = next();
        else if (a == "--minimax-prefill-dense") prefill_dense = next();
        else if (a == "--minimax-prefill-expert-gemm") prefill_expert_gemm = next();
        else if (a == "--minimax-prefill-read") prefill_read = next();
        else if (a == "--minimax-expert-prefetch") expert_prefetch = std::atoi(next().c_str());
        else if (a == "--minimax-dense-mmvq") dense_mmvq = next();
        else if (a == "--minimax-gpu-normrope") gpu_normrope = next();
        else if (a == "--minimax-expert-skip") expert_skip = (float)std::atof(next().c_str());
        else if (a == "--minimax-ssd-chunks") ssd_chunks = std::atoi(next().c_str());
        else if (a == "--minimax-expert-skip-ssd") expert_skip_ssd = std::atoi(next().c_str());
        else if (a == "--minimax-ssd-zerocopy") ssd_zerocopy = next();
        else if (a == "--minimax-page-probe") page_probe = std::atoi(next().c_str());
        else if (a == "--minimax-ssd-prefetch") ssd_prefetch = next();
        else if (a == "--minimax-ssd-prefetch-depth") ssd_prefetch_depth = std::atoi(next().c_str());
        else if (a == "--minimax-route-bias") route_bias = next();
        else if (a == "--minimax-route-bias-in") route_bias_in = (float)std::atof(next().c_str());
        else if (a == "--minimax-route-bias-out") route_bias_out = (float)std::atof(next().c_str());
        else if (a == "--minimax-route-protect") route_protect = std::atoi(next().c_str());
        else if (a == "--minimax-route-pool") route_pool = std::atoi(next().c_str());
        else if (a == "--minimax-route-margin") route_margin = (float)std::atof(next().c_str());
        else if (a == "--minimax-graphs") graphs = next();
        else if (a == "--minimax-expert-resident") expert_resident = next();
        else if (a == "--minimax-tiers") tiers = true;
        else if (a == "--minimax-profile") { profile = next(); tiers = true; }
        else if (a == "--minimax-profile-save") profile_save = next();
        else if (a == "--minimax-routing-counts") routing_counts = next();
        else if (a == "--minimax-vram-slots-per-layer") { vram_slots = std::atoi(next().c_str()); tiers = true; }
        else if (a == "--minimax-ram-experts-per-layer") { ram_experts = std::atoi(next().c_str()); tiers = true; }
        else if (a == "--vram-reserve-mib" || a == "--minimax-vram-reserve-mib")
            vram_reserve_mib = std::atoi(next().c_str());
        else if (a == "--memory-limit-mib") memory_limit_mib = std::atoi(next().c_str());
        else if (a == "--minimax-ram-headroom-mib") ram_headroom_mib = std::atoi(next().c_str());
        else if (a == "--minimax-ram-mode") ram_mode = next();
        else if (a == "--minimax-stats") stats = true;
        else if (a == "--minimax-stats-per-layer") { stats_per_layer = true; stats = true; }
        else if (a == "--minimax-profile-time") profile_time = true;
        else if (a == "--no-minimax-adaptive" || a == "--minimax-no-adaptive") adaptive = false;
        else if (a == "--minimax-pack-quant" || a == "--minimax-pack-iq2xs") {
            pack = true;
            pack_type = a == "--minimax-pack-iq2xs" ? "iq2_xs" : next();
            pack_rows = std::atoll(next().c_str());
            pack_cols = std::atoll(next().c_str());
            pack_in = next();
            pack_out = next();
        }
        else if (a == "--minimax-pack-dequant") {
            unpack = true;
            pack_type = next();
            pack_rows = std::atoll(next().c_str());
            pack_cols = std::atoll(next().c_str());
            pack_in = next();
            pack_out = next();
        }
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--minimax-vision") vision_img = next();
        else if (a == "--minimax-vision-out") vision_out = next();
        else if (a == "--minimax-vision-pixels") vision_pixels = next();
        else if (a == "--minimax-vision-size") vision_size = next();
        else if (a == "--minimax-vision-pixels-out") vision_pixels_out = next();
        else if (a == "--prefill" || a == "--chunk") chunk = parse_chunk(next());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--minimax-pool-spin-us") pool_spin_us = std::atoi(next().c_str());
        else if (a == "--pool-workers") pool_workers = std::atoi(next().c_str());
        else if (a == "--gpu") (void)next();
        else if (a == "--max-new" || a == "--n" || a == "--tokens" || a == "--force" || a == "--mtp" ||
                 a == "--batch" || a == "--slots" || a == "--expert-vram" || a == "--expert-slots" ||
                 a == "--hybrid-rows" || a == "--draft-vocab" || a == "--bench-pp" || a == "--bench-tg" ||
                 a == "--emb" || a == "--dump-full-logits")
            (void)next();
        else if (a == "--vision") {
            // Bare `--vision` and `--vision off|on|gpu|cpu` are both accepted. The value is not used.
            if (i + 1 < argc && !starts_with(argv[i + 1], "--")) ++i;
        }
        else if (a == "--no-warmup" || a == "--no-token-graph" || a == "--no-graph" ||
                 a == "--batch-mtp" || a == "--no-fit-max-tokens")
            continue;
        else if (a == "--greedy") greedy = true;
        else if (serve_mode && starts_with(a, "--")) {
            const bool val = i + 1 < argc && !starts_with(argv[i + 1], "--");
            std::fprintf(stderr, "strata-minimax: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else if (serve_mode) {
            std::fprintf(stderr, "strata-minimax: ignoring '%s' (not an option of this engine)\n", a.c_str());
        } else {
            usage();
            return 2;
        }
    }
    if (pack) return pack_quant(pack_type, pack_rows, pack_cols, pack_in, pack_out);
    if (unpack) return unpack_dequant(pack_type, pack_rows, pack_cols, pack_in, pack_out);
    if (!vision_img.empty() || !vision_pixels.empty()) {
        if (mmproj.empty()) {
            std::fprintf(stderr, "strata-minimax: --minimax-vision needs --mmproj\n");
            return 2;
        }
        if (vision_out.empty() && vision_pixels_out.empty()) {
            std::fprintf(stderr, "strata-minimax: --minimax-vision needs --minimax-vision-out\n");
            return 2;
        }
        try {
            const auto t0 = std::chrono::steady_clock::now();
            Vision vis(mmproj, cpu_threads > 0 ? cpu_threads : pool_workers);
            ImageEmbeddings emb;
            if (!vision_pixels_out.empty()) {
                if (vision_img.empty()) {
                    std::fprintf(stderr, "strata-minimax: --minimax-vision-pixels-out needs --minimax-vision\n");
                    return 2;
                }
                std::vector<float> pv;
                int64_t n = 0, rw = 0, rh = 0;
                vis.preprocess_file(vision_img, pv, n, rw, rh);
                write_npy_f32(vision_pixels_out, pv.data(), (int)n, 1176);
                std::printf("PIXELS %lld %lld %lld %s\n", (long long)n, (long long)rw, (long long)rh,
                            vision_pixels_out.c_str());
                std::fflush(stdout);
                return 0;
            }
            if (!vision_pixels.empty()) {
                std::vector<float> pv;
                int64_t rows = 0, cols = 0;
                if (!read_npy_2d(vision_pixels, pv, rows, cols)) {
                    std::fprintf(stderr, "strata-minimax: cannot read %s as a 2-D float32 .npy\n",
                                 vision_pixels.c_str());
                    return 2;
                }
                int64_t w = 0, h = 0;
                if (!vision_size.empty()) {
                    const auto x = vision_size.find('x');
                    if (x == std::string::npos) {
                        std::fprintf(stderr, "strata-minimax: --minimax-vision-size wants WxH\n");
                        return 2;
                    }
                    w = std::atoll(vision_size.substr(0, x).c_str());
                    h = std::atoll(vision_size.substr(x + 1).c_str());
                }
                if (w <= 0 || h <= 0) {
                    std::fprintf(stderr, "strata-minimax: --minimax-vision-pixels needs --minimax-vision-size WxH\n");
                    return 2;
                }
                if (cols != 1176) {
                    std::fprintf(stderr, "strata-minimax: pixel_values must be [n, 1176], got [%lld, %lld]\n",
                                 (long long)rows, (long long)cols);
                    return 2;
                }
                emb = vis.encode_pixels(pv.data(), rows, w, h);
            } else {
                emb = vis.encode_file(vision_img);
            }
            write_npy_f32(vision_out, emb.data.data(), (int)emb.tokens, (int)emb.dim);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("VISION %lld %lld %lld %lld %.1f %s\n", (long long)emb.tokens, (long long)emb.dim,
                        (long long)emb.width, (long long)emb.height, ms, vision_out.c_str());
            std::fflush(stdout);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-minimax: %s\n", e.what());
            return 1;
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "strata-minimax: --model is required\n");
        return 2;
    }
    if (serve_mode && !eval_path.empty()) {
        std::fprintf(stderr, "strata-minimax: --minimax-eval and --serve are exclusive\n");
        return 2;
    }
    if (!eval_path.empty() && dump_path.empty()) {
        std::fprintf(stderr, "strata-minimax: --minimax-eval needs --minimax-dump-logits\n");
        return 2;
    }
    if (chunk <= 0) chunk = 16;

    if (inspect) {
        try {
            const Model m = load_model(model, false);
            std::cout << report(m);
            std::cout.flush();
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-minimax: %s\n", e.what());
            return 1;
        }
    }

    // Fingerprint before any CUDA allocation. A non-M3.1 file stops here, with no READY.
    Model meta;
    try {
        meta = load_model(model, !allow_nonstandard);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-minimax: %s\n", e.what());
        return 1;
    }
    if (!serve_mode && eval_path.empty()) {
        std::printf("%s\n", summary(meta.config).c_str());
        std::fprintf(stderr, "strata-minimax: pass --serve, --minimax-inspect, or --minimax-eval\n");
        return 0;
    }

    // An unasked context stays at most 32k. --max-context is what raises it, still clamped to the model.
    constexpr int64_t kUnaskedContext = 32768;
    int64_t ctx = meta.config.context;
    if (ctx_asked > 0) ctx = (std::min)(ctx, ctx_asked);
    else ctx = (std::min)(ctx, kUnaskedContext);
    if (ctx < 1) ctx = kUnaskedContext;

    EngineOptions opt;
    opt.max_context = ctx;
    opt.dense_layers_gpu = dense_gpu;
    opt.cpu_threads = pool_workers > 0 ? pool_workers : cpu_threads;
    opt.chunk = chunk;
    opt.allow_nonstandard = allow_nonstandard;
    opt.tiers = tiers;
    opt.profile_path = profile;
    opt.profile_save_path = profile_save;
    opt.routing_counts_path = routing_counts;
    opt.vram_slots_per_layer = vram_slots;
    opt.ram_experts_per_layer = ram_experts;
    opt.vram_reserve_mib = vram_reserve_mib;
    opt.memory_limit_mib = memory_limit_mib;
    opt.ram_headroom_mib = ram_headroom_mib;
    opt.ram_mode = ram_mode;
    opt.pool_spin_us = pool_spin_us;
    opt.adaptive = adaptive;
    opt.stats = stats;
    opt.stats_per_layer = stats_per_layer;
    opt.profile_time = profile_time;
    opt.dump_dir = dump_dir;
    opt.dump_layers = dump_layers;
    opt.seed_hidden = seed_hidden;
    opt.mmproj = mmproj;
    opt.layer_split = layer_split;
    opt.gpus = gpus;
    opt.kv_space = kv_space;
    opt.kv_quant = kv_quant;
    opt.host_attn = host_attn;
    opt.prefill_kv = prefill_kv;
    opt.prefill_batch = prefill_batch;
    opt.prefill_experts = prefill_experts;
    opt.prefill_dense = prefill_dense;
    opt.prefill_expert_gemm = prefill_expert_gemm;
    opt.prefill_read = prefill_read;
    opt.expert_prefetch = expert_prefetch;
    opt.dense_mmvq = dense_mmvq;
    opt.gpu_normrope = gpu_normrope;
    opt.expert_resident = expert_resident;
    opt.expert_skip = expert_skip;
    opt.ssd_chunks = ssd_chunks;
    opt.expert_skip_ssd = expert_skip_ssd;
    opt.ssd_zerocopy = ssd_zerocopy;
    opt.page_probe = page_probe;
    opt.ssd_prefetch = ssd_prefetch;
    opt.ssd_prefetch_depth = ssd_prefetch_depth;
    opt.route_bias = route_bias;
    opt.route_bias_in = route_bias_in;
    opt.route_bias_out = route_bias_out;
    opt.route_protect = route_protect;
    opt.route_pool = route_pool;
    opt.route_margin = route_margin;
    opt.graphs = graphs;

    Engine* engine = nullptr;
    try {
        engine = new Engine(model, opt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-minimax: %s\n", e.what());
        return 1;
    }

    if (!eval_path.empty()) {
        try {
            const std::vector<int32_t> ids = read_token_file(eval_path);
            std::vector<float> logits;
            int rows = 0;
            engine->teacher_forced(ids, last_k, logits, rows);
            const int cols = (int)engine->config().n_vocab;
            std::fprintf(stderr, "INFO %s\n", engine->describe().c_str());
            write_npy_f32(dump_path, logits.data(), rows, cols);
            std::printf("LOGITS %d %d %s\n", rows, cols, dump_path.c_str());
            std::fflush(stdout);
            engine->save_profile();
            if (stats) {
                const std::string st = engine->stats();
                if (!st.empty()) std::fprintf(stderr, "INFO minimax-stats total %s\n", st.c_str());
            }
            delete engine;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-minimax: %s\n", e.what());
            delete engine;
            return 1;
        }
    }

    std::printf("INFO engine=minimax-m3 version=%s forward=token%s\n", STRATA_VERSION,
                allow_nonstandard ? " geometry=nonstandard" : "");
    std::printf("INFO ctx=%lld layers=%lld embd=%lld experts=%lld\n", (long long)engine->context(),
                (long long)engine->config().n_layers, (long long)engine->config().n_embd,
                (long long)engine->config().n_expert);
    std::printf("INFO %s\n", engine->describe().c_str());
    std::printf("READY %lld stop\n", (long long)engine->context());
    std::fflush(stdout);

    LineQueue lines;
    std::atomic<bool> stop{false};
    std::atomic<bool> shutdown{false};
    std::atomic<bool> reader_done{false};
    std::atomic<int> sweep{0};
    std::mutex sweep_mu;
    std::condition_variable sweep_cv;
    GenGate gate{&stop, &sweep, &reader_done, &sweep_mu, &sweep_cv};
    std::thread reader([&] {
        StdinBuf in;
        auto take_ready = [&] {
            if (sweep.load(std::memory_order_relaxed) != 1) return;
            slurp_ready(in);
            std::string got;
            while (take_line(in, got)) {
                if (got.empty()) continue;
                if (got == "STOP") stop.store(true, std::memory_order_relaxed);
                else lines.push(std::move(got));
            }
            {
                std::lock_guard<std::mutex> lk(sweep_mu);
                sweep.store(2, std::memory_order_relaxed);
            }
            sweep_cv.notify_all();
        };
        while (!shutdown.load(std::memory_order_relaxed)) {
            take_ready();
            if (!wait_input(in, shutdown, sweep)) {
                take_ready();
                if (shutdown.load(std::memory_order_relaxed)) break;
                if (sweep.load(std::memory_order_relaxed) == 1) continue;
                if (in.eof) break;
                continue;
            }
            std::string got;
            while (take_line(in, got)) {
                if (got.empty()) continue;
                if (got == "STOP") stop.store(true, std::memory_order_relaxed);
                else lines.push(std::move(got));
            }
        }
        if (!in.pending.empty()) {
            std::string got = std::move(in.pending);
            if (!got.empty() && got.back() == '\r') got.pop_back();
            if (!got.empty()) {
                if (got == "STOP") stop.store(true, std::memory_order_relaxed);
                else lines.push(std::move(got));
            }
        }
        {
            std::lock_guard<std::mutex> lk(lines.mu);
            lines.eof = true;
        }
        lines.cv.notify_one();
        {
            std::lock_guard<std::mutex> lk(sweep_mu);
            reader_done.store(true, std::memory_order_relaxed);
            if (sweep.load(std::memory_order_relaxed) == 1) sweep.store(2, std::memory_order_relaxed);
        }
        sweep_cv.notify_all();
    });

    std::string line;
    while (lines.pop(line)) {
        if (line == "QUIT") break;
        if (starts_with(line, "GEN ")) {
            handle_gen(*engine, line, gate, greedy, stats || profile_time);
            continue;
        }
        if (starts_with(line, "GENI ") || starts_with(line, "GENI")) {
            handle_gen(*engine, "GEN " + line.substr(4), gate, greedy, stats || profile_time);
            continue;
        }
        if (starts_with(line, "BGEN ") || starts_with(line, "BGENI ") || starts_with(line, "BGENI")) {
            std::printf("ERR minimax engine: batch generate is not implemented\n");
            std::fflush(stdout);
            continue;
        }
        if (line == "BSTOP" || starts_with(line, "BSTOP ") || starts_with(line, "BYIELD")) continue;
        std::printf("ERR minimax engine: '%s' is not implemented\n", line.c_str());
        std::fflush(stdout);
    }
    shutdown.store(true, std::memory_order_relaxed);
    reader.join();
    delete engine;
    return 0;
}

}  // namespace strata::minimax
