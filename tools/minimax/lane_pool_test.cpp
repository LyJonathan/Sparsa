// tools/minimax/lane_pool_test.cpp - R9: src/minimax/lane_pool.hpp.
//   1. the cpu-list parser and the sysfs topology reader on a fake two-CCD tree (6 cores, SMT2): one CPU per
//      physical core, CCD by CCD;
//   2. run_dyn hands every row out exactly once, for thread counts 1..12, one or two CCDs (stealing), the guided
//      tail on and off, pinned and not, many rounds (a lost or doubled row fails);
//   3. the real machine's topology is printed (informational).
// Exit 0 = pass.
#include "../../src/minimax/lane_pool.hpp"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using strata::minimax::LaneCpu;
using strata::minimax::LanePool;
using strata::minimax::LanePoolOptions;
namespace topo = strata::minimax::lane_topo;

static int g_fail = 0;
#define CHECK(c, ...)                                                       \
    do {                                                                    \
        if (!(c)) {                                                         \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                \
            std::printf(__VA_ARGS__);                                       \
            std::printf("\n");                                              \
            ++g_fail;                                                       \
        }                                                                   \
    } while (0)

#if defined(__linux__)
static void write(const std::filesystem::path& p, const std::string& s) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p) << s << "\n";
}
#endif

static void test_parse() {
    const auto v = topo::parse_cpu_list("0-2,6,8-9");
    CHECK((v == std::vector<int>{0, 1, 2, 6, 8, 9}), "parse_cpu_list");
    CHECK(topo::parse_cpu_list("x").empty(), "parse_cpu_list garbage");
}

static void test_fake_sysfs() {
#if defined(__linux__)
    const auto root = std::filesystem::temp_directory_path() / "lane_pool_test_sysfs";
    std::filesystem::remove_all(root);
    // cpu0-5: cores 0-5 (CCD0 = cores 0-2, CCD1 = cores 3-5); cpu6-11: their SMT siblings
    for (int c = 0; c < 12; ++c) {
        const int core = c % 6;
        const auto d = root / "sys/devices/system/cpu" / ("cpu" + std::to_string(c));
        write(d / "topology/core_id", std::to_string(core));
        write(d / "topology/physical_package_id", "0");
        write(d / "cache/index3/shared_cpu_list", core < 3 ? "0-2,6-8" : "3-5,9-11");
    }
    const auto cores = topo::physical_cores_by_ccd(root.string());
    CHECK(cores.size() == 6, "6 physical cores, got %zu", cores.size());
    if (cores.size() == 6) {
        for (int k = 0; k < 6; ++k) {
            CHECK(cores[(size_t)k].cpu == k, "core %d -> cpu %d", k, cores[(size_t)k].cpu);
            CHECK(cores[(size_t)k].ccd == (k < 3 ? 0 : 1), "core %d ccd %d", k, cores[(size_t)k].ccd);
        }
    }
    std::filesystem::remove_all(root);
#endif
}

static void test_coverage() {
    const std::vector<LaneCpu> two_ccd = {{-1, 0}, {-1, 0}, {-1, 0}, {-1, 1}, {-1, 1}, {-1, 1},
                                          {-1, 0}, {-1, 0}, {-1, 0}, {-1, 1}, {-1, 1}, {-1, 1}};
    for (int threads : {1, 2, 3, 5, 12}) {
        for (int ccd = 0; ccd < 2; ++ccd) {
            for (int tail : {0, 4}) {
                LanePoolOptions o;
                o.ccd = ccd == 1;
                o.tail = tail;
                o.spin_us = (threads & 1) ? 50 : 0;
                LanePool pool(threads, o, two_ccd);
                CHECK(pool.ccds() == (ccd ? (threads > 3 ? 2 : 1) : 1), "ccds %d threads %d", pool.ccds(), threads);
                for (int nrows : {1, 7, 100, 1536, 12345}) {
                    for (int chunk : {1, 64}) {
                        std::vector<std::atomic<int>> hits((size_t)nrows);
                        for (int round = 0; round < 20; ++round) {
                            for (auto& h : hits) h.store(0);
                            pool.run_dyn(nrows, chunk, [&](int a, int b) {
                                CHECK(a >= 0 && b <= nrows && a < b && b - a <= chunk, "bad chunk [%d,%d)", a, b);
                                for (int i = a; i < b; ++i) hits[(size_t)i].fetch_add(1);
                            });
                            int bad = 0;
                            for (auto& h : hits) bad += h.load() != 1;
                            CHECK(bad == 0, "threads %d ccd %d tail %d nrows %d chunk %d: %d rows not hit once", threads,
                                  ccd, tail, nrows, chunk, bad);
                        }
                    }
                }
            }
        }
    }
}

static void test_pinned_real() {
    LanePoolOptions o;
    o.pin = true;
    o.ccd = true;
    LanePool pool(64, o);
    std::printf("real machine: %d pinned workers over %d CCD(s):", pool.threads(), pool.ccds());
    for (const auto& s : pool.slots()) std::printf(" %d/%d", s.cpu, s.ccd);
    std::printf("\n");
    std::vector<std::atomic<int>> hits(100000);
    pool.run_dyn(100000, 64, [&](int a, int b) {
        for (int i = a; i < b; ++i) hits[(size_t)i].fetch_add(1);
    });
    int bad = 0;
    for (auto& h : hits) bad += h.load() != 1;
    CHECK(bad == 0, "pinned: %d rows not hit once", bad);
}

int main() {
    test_parse();
    test_fake_sysfs();
    test_coverage();
    test_pinned_real();
    std::printf("lane_pool_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
