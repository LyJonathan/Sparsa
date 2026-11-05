// src/minimax/lane_pool.hpp - R9: a worker pool for the MiniMax CPU lane's two expert regions, CCD-aware.
//
// The engine's `Pool` (engine.cpp) hands out the lane's rows from ONE atomic counter to threads the OS places
// anywhere.  On a two-CCD Zen part (Ryzen 9 5900X/5950X/7900X...) that counter's cache line bounces between the CCDs
// on every chunk, SMT siblings compete for one core's load ports, and the last chunks of a region straggle.  This
// pool changes only WHO computes a row, never what a row computes, so the lane's numbers are bit-identical:
//
//   STRATA_POOL_PIN=1     one worker per PHYSICAL core, pinned, ordered CCD by CCD (L3 domain); the calling
//                         thread then only waits (its affinity is never changed).
//   STRATA_POOL_CCD=1     one row counter per CCD (each CCD starts on its own share of the rows, sized by its thread
//                         count); a CCD that runs dry steals from the others' counters.
//   STRATA_POOL_TAIL=N    guided tail: a chunk shrinks to max(N, remaining / (2 x threads)) near the end of a region
//                         (0 = off, the fixed chunk).
//   STRATA_POOL_SPIN_US=N the idle spin before a worker sleeps (the engine's --pool-spin-us knob, same meaning).
//
// Turned on as a whole by STRATA_LANE_POOL=1 (engine.cpp); everything defaults off.  Linux reads the topology from
// /sys/devices/system/cpu (core_id, cache/index3/shared_cpu_list); Windows from GetLogicalProcessorInformationEx
// (RelationProcessorCore, RelationCache L3).  Anything it cannot read degrades to "one CCD, logical order, no pinning".
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace strata::minimax {

/// One worker slot: the logical CPU to pin to (-1 = no pinning) and its CCD (L3 domain) index.
struct LaneCpu {
    int cpu = -1;
    int ccd = 0;
};

struct LanePoolOptions {
    bool pin = false;
    bool ccd = false;
    int tail = 0;
    int spin_us = 0;

    static LanePoolOptions from_env(int spin_default = 0) {
        auto num = [](const char* n, int d) {
            const char* v = std::getenv(n);
            return v && *v ? std::atoi(v) : d;
        };
        LanePoolOptions o;
        o.pin = num("STRATA_POOL_PIN", 0) == 1;
        o.ccd = num("STRATA_POOL_CCD", 0) == 1;
        o.tail = std::max(0, num("STRATA_POOL_TAIL", 0));
        o.spin_us = std::max(0, num("STRATA_POOL_SPIN_US", spin_default));
        return o;
    }
};

namespace lane_topo {

/// "0-5,12-17" -> {0..5, 12..17}
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        const size_t dash = part.find('-');
        try {
            if (dash == std::string::npos) {
                if (!part.empty() && part.find_first_not_of(" \n\t") != std::string::npos) out.push_back(std::stoi(part));
            } else {
                const int a = std::stoi(part.substr(0, dash)), b = std::stoi(part.substr(dash + 1));
                for (int c = a; c <= b; ++c) out.push_back(c);
            }
        } catch (...) {
            return {};
        }
    }
    return out;
}

inline bool read_line(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::getline(f, out);
    return true;
}

/// The physical cores of this machine, one logical CPU each (the lowest-numbered sibling), ordered CCD by CCD, with
/// their CCD index.  `root` is the sysfs root ("" = the real one; the test points it at a fake tree).
inline std::vector<LaneCpu> physical_cores_by_ccd(const std::string& root = "") {
    std::vector<LaneCpu> out;
#if defined(_WIN32)
    (void)root;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    if (len == 0) return out;
    std::vector<uint8_t> buf(len);
    auto* base = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data();
    if (!GetLogicalProcessorInformationEx(RelationAll, base, &len)) return out;
    std::vector<std::pair<int, KAFFINITY>> cores;   // (group, mask) of each physical core
    std::vector<std::pair<int, KAFFINITY>> l3s;
    for (DWORD off = 0; off < len;) {
        auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1)
            cores.push_back({(int)e->Processor.GroupMask[0].Group, e->Processor.GroupMask[0].Mask});
        else if (e->Relationship == RelationCache && e->Cache.Level == 3)
            l3s.push_back({(int)e->Cache.GroupMask.Group, e->Cache.GroupMask.Mask});
        off += e->Size;
    }
    for (const auto& c : cores) {
        if (c.second == 0) continue;
        int bit = 0;
        while (!((c.second >> bit) & 1)) ++bit;
        int ccd = 0;
        for (size_t k = 0; k < l3s.size(); ++k)
            if (l3s[k].first == c.first && (l3s[k].second & c.second)) { ccd = (int)k; break; }
        out.push_back({c.first * 64 + bit, ccd});
    }
#elif defined(__linux__)
    const std::string sys = root + "/sys/devices/system/cpu/";
    const unsigned n = std::max(1u, std::thread::hardware_concurrency());
    std::set<std::pair<int, int>> seen_core;   // (package, core_id)
    std::vector<std::string> l3_keys;
    for (unsigned c = 0; c < 4096 && (c < n || !root.empty()); ++c) {
        const std::string d = sys + "cpu" + std::to_string(c) + "/";
        std::string core, pkg, l3;
        if (!read_line(d + "topology/core_id", core)) {
            if (c >= n) break;
            continue;
        }
        if (!read_line(d + "topology/physical_package_id", pkg)) pkg = "0";
        if (!read_line(d + "cache/index3/shared_cpu_list", l3)) l3 = "pkg" + pkg;
        int core_i = 0, pkg_i = 0;
        try {
            core_i = std::stoi(core);
            pkg_i = std::stoi(pkg);
        } catch (...) {
            continue;
        }
        if (!seen_core.insert({pkg_i, core_i}).second) continue;   // an SMT sibling of a core already listed
        int ccd = -1;
        for (size_t k = 0; k < l3_keys.size(); ++k)
            if (l3_keys[k] == l3) ccd = (int)k;
        if (ccd < 0) {
            ccd = (int)l3_keys.size();
            l3_keys.push_back(l3);
        }
        out.push_back({(int)c, ccd});
    }
#else
    (void)root;
#endif
    std::stable_sort(out.begin(), out.end(), [](const LaneCpu& a, const LaneCpu& b) { return a.ccd < b.ccd; });
    return out;
}

inline bool pin_this_thread(int cpu) {
    if (cpu < 0) return false;
#if defined(_WIN32)
    GROUP_AFFINITY ga{};
    ga.Group = (WORD)(cpu / 64);
    ga.Mask = (KAFFINITY)1 << (cpu % 64);
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    return false;
#endif
}

}  // namespace lane_topo

class LanePool {
public:
    using Job = std::function<void(int, int)>;

    /// `threads` workers (unpinned: the caller is worker 0).  With opt.pin the count is capped at the physical cores and the
    /// workers are placed per `slots` (default: the machine's cores by CCD); without it `slots` only gives the CCDs.
    LanePool(int threads, const LanePoolOptions& opt, std::vector<LaneCpu> slots = {}) : opt_(opt) {
        if (slots.empty() && (opt.pin || opt.ccd)) slots = lane_topo::physical_cores_by_ccd();
        if (threads < 1) threads = 1;
        if (opt.pin && !slots.empty() && threads > (int)slots.size()) threads = (int)slots.size();
        n_ = threads;
        slot_.assign((size_t)n_, LaneCpu{});
        for (int p = 0; p < n_; ++p)
            if (p < (int)slots.size()) slot_[(size_t)p] = slots[(size_t)p];
        if (!opt.pin)
            for (auto& s : slot_) s.cpu = -1;
        n_ccd_ = 1;
        if (opt.ccd)
            for (const auto& s : slot_) n_ccd_ = std::max(n_ccd_, s.ccd + 1);
        else
            for (auto& s : slot_) s.ccd = 0;
        ccd_.reset(new Ccd[(size_t)n_ccd_]);
        for (const auto& s : slot_) ccd_[(size_t)s.ccd].threads++;
        // Pinned: every slot is a pinned worker and the caller (the engine's main thread, which also issues the GPU
        // work) only waits, so its own affinity is never touched.  Unpinned: the caller is worker 0, as in `Pool`.
        first_ = opt.pin ? 0 : 1;
        for (int p = first_; p < n_; ++p) workers_.emplace_back([this, p] { worker(p); });
    }

    ~LanePool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            epoch_.fetch_add(1);
        }
        cv_work_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    LanePool(const LanePool&) = delete;
    LanePool& operator=(const LanePool&) = delete;

    int threads() const { return n_; }
    int ccds() const { return n_ccd_; }
    const std::vector<LaneCpu>& slots() const { return slot_; }

    /// Every row in [0, nrows) is passed to `job` exactly once, in chunks of at most `chunk_rows`.  Rows must be
    /// independent (the lane's rows are); which thread computes a row is unspecified.
    void run_dyn(int nrows, int chunk_rows, const Job& job) {
        if (nrows <= 0) return;
        chunk_ = std::max(1, chunk_rows);
        if (n_ <= 1 || nrows < 2 * chunk_) {
            for (int a = 0; a < nrows; a += chunk_) job(a, std::min(nrows, a + chunk_));
            return;
        }
        rows_ = nrows;
        // each CCD's share of the rows, sized by its thread count
        int at = 0;
        for (int k = 0; k < n_ccd_; ++k) {
            Ccd& c = ccd_[(size_t)k];
            const int share = k + 1 == n_ccd_ ? nrows - at : (int)((int64_t)nrows * c.threads / n_);
            c.begin = at;
            c.end = at + share;
            c.next.store(at, std::memory_order_relaxed);
            at += share;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &job;
            pending_ = n_ - first_;
            epoch_.fetch_add(1, std::memory_order_release);
        }
        cv_work_.notify_all();
        if (first_ == 1) work(0, job);
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&] { return pending_ == 0; });
        fn_ = nullptr;
    }

private:
    struct alignas(64) Ccd {
        std::atomic<int> next{0};
        int begin = 0, end = 0;
        int threads = 0;
        char pad[64];
    };

    static inline void relax() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#endif
    }

    int take_chunk(Ccd& c) const {
        if (opt_.tail <= 0) return chunk_;
        const int left = c.end - c.next.load(std::memory_order_relaxed);
        return std::max(1, std::min(chunk_, std::max(opt_.tail, left / (2 * std::max(1, c.threads)))));
    }

    bool drain(Ccd& c, const Job& job) {
        bool any = false;
        for (;;) {
            const int ch = take_chunk(c);
            const int a = c.next.fetch_add(ch, std::memory_order_relaxed);
            if (a >= c.end) return any;
            job(a, std::min(c.end, a + ch));
            any = true;
        }
    }

    void work(int p, const Job& job) {
        const int home = slot_[(size_t)p].ccd;
        drain(ccd_[(size_t)home], job);
        for (int k = 1; k < n_ccd_; ++k) drain(ccd_[(size_t)((home + k) % n_ccd_)], job);   // steal
    }

    void worker(int p) {
        if (opt_.pin) lane_topo::pin_this_thread(slot_[(size_t)p].cpu);
        int seen = 0;
        for (;;) {
            if (opt_.spin_us > 0) {
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(opt_.spin_us);
                int k = 0;
                while (!stop_.load(std::memory_order_relaxed) && epoch_.load(std::memory_order_acquire) == seen) {
                    relax();
                    if ((++k & 255) == 0 && std::chrono::steady_clock::now() >= until) break;
                }
            }
            const Job* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [&] { return stop_.load() || epoch_.load() != seen; });
                if (stop_) return;
                seen = epoch_.load();
                job = fn_;
            }
            if (job) work(p, *job);
            std::lock_guard<std::mutex> lk(mu_);
            if (--pending_ == 0) cv_done_.notify_one();
        }
    }

    LanePoolOptions opt_;
    int n_ = 1, n_ccd_ = 1;
    std::vector<LaneCpu> slot_;
    std::unique_ptr<Ccd[]> ccd_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    const Job* fn_ = nullptr;
    int rows_ = 0, chunk_ = 1, pending_ = 0;
    std::atomic<int> epoch_{0};
    std::atomic<bool> stop_{false};
    int first_ = 1;
};

}  // namespace strata::minimax
