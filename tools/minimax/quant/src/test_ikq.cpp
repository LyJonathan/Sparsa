// tools/minimax/quant/src/test_ikq.cpp - the ported ik_llama.cpp formats, against ik's own output.
//
// Three checks per format:
//   1. quantize: our bytes against ik's `ggml_quantize_chunk` for the same tensor, imatrix and thread
//      count.  Bit-exact is the only pass.
//   2. dequantize: our values against ik's `to_float` times the run-time factor ik applies in its matmul
//      (1.05 for IQ2_KT, 1.01 for IQ3_KT; ik's `to_float` is inconsistent between the two).  Bit-exact.
//   3. dot: our `dot_row` against a double-precision dot of our own dequantized values.  Relative error
//      below 1e-5, which is what float accumulation of 256 products costs.
//
// The reference bytes come from `_ref/research/iqk/iqk-dump`, which links ik_llama.cpp (MIT) unmodified.
// Without that binary the test reports the port's own consistency and skips checks 1 and 2; it says so.
//
//   test_ikq [--ref <iqk-dump.exe>] [--work <dir>] [--bench]

#include "strata/kernels/cpu/iqk_quants.hpp"

#include "strata/kernels/iqk_types.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <process.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::kernels::ikq;

namespace {

int g_failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL %s\n", what.c_str());
    ++g_failures;
}

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float uniform(uint64_t& s) { return (float) ((splitmix(s) >> 40) & 0xffffff) / (float) 0x7fffff - 1.0f; }

/// A tensor with the shape of a real expert row: mostly small values, a few heavy ones.
std::vector<float> make_tensor(int64_t nrows, int64_t n, uint64_t seed) {
    std::vector<float> x((size_t) (nrows * n));
    for (int64_t r = 0; r < nrows; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            // roughly the shape of an expert row: Gaussian body, a thin tail of larger values
            const float u1 = 0.5f * (uniform(seed) + 1.f) + 1e-7f;
            const float u2 = 0.5f * (uniform(seed) + 1.f);
            float v = 0.02f * std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
            if (j % 101 == 0) v *= 4.f;
            x[(size_t) (r * n + j)] = v;
        }
    }
    return x;
}

std::vector<float> make_imatrix(int64_t n, uint64_t seed) {
    std::vector<float> w((size_t) n);
    for (int64_t j = 0; j < n; ++j) w[(size_t) j] = 0.05f + std::fabs(uniform(seed));
    return w;
}

bool read_file(const fs::path& p, std::vector<char>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize((size_t) f.tellg());
    f.seekg(0, std::ios::beg);
    f.read(out.data(), (std::streamsize) out.size());
    return (bool) f;
}

void write_file(const fs::path& p, const void* data, size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char*) data, (std::streamsize) bytes);
}

int ik_id(int type) {
    switch (type) {
    case kIQ2KT: return 153;
    case kIQ3KT: return 154;
    case kIQ3K: return 138;
    case kIQ4KS: return 144;
    case kIQ2KL: return 157;
    default: return -1;
    }
}

std::string mbs(double bytes, double secs) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f MB/s", secs > 0 ? bytes / 1e6 / secs : 0.0);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    fs::path ref_exe;
    fs::path work = fs::temp_directory_path() / "strata-ikq-test";
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_exe = argv[++i];
        else if (!std::strcmp(argv[i], "--work") && i + 1 < argc) work = argv[++i];
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
        else {
            std::fprintf(stderr, "usage: %s [--ref <iqk-dump.exe>] [--work <dir>] [--bench]\n", argv[0]);
            return 2;
        }
    }
    if (ref_exe.empty()) {
        const char* env = std::getenv("IQK_DUMP");
        if (env) ref_exe = env;
    }
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) {
        std::fprintf(stderr, "cannot create %s\n", work.string().c_str());
        return 2;
    }

    const bool have_ref = !ref_exe.empty() && fs::exists(ref_exe);
    std::printf("ik_llama.cpp format port: %s\n", have_ref ? "with the ik reference binary" : "no reference binary");

    // Build both codebooks before any timing: ik's cluster search is a one-off cost of about a second and a
    // half, and charging it to the first case would read as a broken quantizer.
    {
        const auto warm = make_tensor(1, 256, 0x5eedull);
        const auto ww = make_imatrix(256, 0x5eedull);
        for (int t : {kIQ2KT, kIQ3KT, kIQ3K, kIQ4KS, kIQ2KL}) {
            std::vector<char> out(strata::kernels::ikq::row_bytes(t, 256));
            quantize_rows(t, warm.data(), out.data(), 1, 256, ww.data(), 1);
        }
    }

    struct Case {
        const char* label;
        int type;
        int64_t rows;
        int64_t cols;
        bool imatrix;
    };
    const Case cases[] = {
        {"IQ2_KT", kIQ2KT, 32, 512, true},
        {"IQ2_KT", kIQ2KT, 16, 544, true},    // a 32-value tail block
        {"IQ2_KT", kIQ2KT, 16, 512, false},   // no imatrix
        {"IQ3_KT", kIQ3KT, 32, 512, true},
        // no 544-wide IQ3_KT row: ik's own tail path aborts on the all-zero importance pad, so the format
        // is defined for whole super-blocks only (see quantize_rows)
        {"IQ3_KT", kIQ3KT, 16, 768, true},
        {"IQ3_KT", kIQ3KT, 16, 512, false},
        // IQ3_K, IQ4_KS and IQ2_KL have no tail layout in ik either (GGML_ASSERT(n_per_row%QK_K == 0)).
        {"IQ3_K", kIQ3K, 32, 512, true},
        {"IQ3_K", kIQ3K, 16, 768, true},
        {"IQ3_K", kIQ3K, 16, 512, false},
        {"IQ3_K", kIQ3K, 16, 1024, true},
        {"IQ3_K", kIQ3K, 8, 256, true},
        {"IQ3_K", kIQ3K, 12, 3072, true},
        {"IQ4_KS", kIQ4KS, 32, 512, true},
        {"IQ4_KS", kIQ4KS, 16, 768, true},
        {"IQ4_KS", kIQ4KS, 16, 512, false},
        {"IQ4_KS", kIQ4KS, 16, 1024, true},
        {"IQ4_KS", kIQ4KS, 8, 256, true},
        {"IQ4_KS", kIQ4KS, 12, 3072, true},
        {"IQ2_KL", kIQ2KL, 32, 512, true},
        {"IQ2_KL", kIQ2KL, 16, 768, true},
        {"IQ2_KL", kIQ2KL, 16, 512, false},
        {"IQ2_KL", kIQ2KL, 16, 1024, true},
        {"IQ2_KL", kIQ2KL, 8, 256, true},
        {"IQ2_KL", kIQ2KL, 12, 3072, true},
    };

    for (const Case& c : cases) {
        std::printf("\n%s %lldx%lld%s\n", c.label, (long long) c.rows, (long long) c.cols,
                    c.imatrix ? " + imatrix" : "");
        const auto x = make_tensor(c.rows, c.cols, 0x1234567ull + (uint64_t) c.cols);
        const auto imx = make_imatrix(c.cols, 0x89abcdefull);
        const fs::path in_p = work / "in.f32";
        const fs::path imx_p = work / "imx.f32";
        write_file(in_p, x.data(), x.size() * sizeof(float));
        write_file(imx_p, imx.data(), imx.size() * sizeof(float));

        const size_t row_bytes = strata::kernels::ikq::row_bytes(c.type, c.cols);
        std::vector<char> ours((size_t) c.rows * row_bytes);
        auto t0 = std::chrono::steady_clock::now();
        size_t got = 0;
        try {
            got = quantize_rows(c.type, x.data(), ours.data(), c.rows, c.cols,
                                c.imatrix ? imx.data() : nullptr, 6);
        } catch (const std::exception& e) {
            fail(std::string("quantize_rows threw: ") + e.what());
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (got != ours.size()) {
            fail("quantize_rows returned " + std::to_string(got) + ", expected " + std::to_string(ours.size()));
            continue;
        }
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        if (bench) {
            std::printf("  quantize %.2f s, %s (%.0f MB/s f32 logical)\n", secs,
                        mbs((double) x.size() * sizeof(float), secs).c_str(),
                        (double) x.size() * sizeof(float) / 1e6 / secs);
        }

        // deploy: the reader must see the same values back
        std::vector<float> deq((size_t) (c.rows * c.cols));
        for (int64_t r = 0; r < c.rows; ++r)
            dequantize_row(c.type, ours.data() + (size_t) r * row_bytes, deq.data() + (size_t) r * c.cols, c.cols);
        double se = 0, sq = 0, smax = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double e = (double) x[i] - deq[i];
            se += e * e;
            sq += (double) x[i] * x[i];
            smax = std::max(smax, std::fabs(e));
        }
        std::printf("  round trip: rel %.4f, max abs %.3g\n", std::sqrt(se / sq), smax);
        if (!(std::sqrt(se / sq) < 0.40)) fail("round-trip error out of range");

        // dot against a double reference over our own dequantized values (whole super-blocks only)
        if ((c.cols % 256) == 0) {
        std::vector<float> y((size_t) c.cols);
        uint64_t seed = 0xd1ce;
        for (int64_t j = 0; j < c.cols; ++j) y[(size_t) j] = 0.5f * uniform(seed);
        // q8_K quantize, the activation format the engine hands the kernels
        std::vector<char> q8;
        {
            const int nb = (int) (c.cols / 256);
            q8.resize((size_t) nb * (4 + 256 + 32));
            for (int ib = 0; ib < nb; ++ib) {
                const float* yb = y.data() + ib * 256;
                float amax = 0;
                for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(yb[j]));
                const float d = amax / 127.f;
                char* blk = q8.data() + (size_t) ib * (4 + 256 + 32);
                std::memcpy(blk, &d, 4);
                int8_t* qs = (int8_t*) (blk + 4);
                for (int j = 0; j < 256; ++j) qs[j] = (int8_t) std::lround(yb[j] / d);
            }
        }
        double worst = 0;
        for (int64_t r = 0; r < c.rows; ++r) {
            const float s = dot_row(c.type, ours.data() + (size_t) r * row_bytes, q8.data(), (int) c.cols);
            double ref = 0;
            for (int64_t j = 0; j < c.cols; ++j) {
                double yq = 0;
                {
                    const int ib = (int) (j / 256);
                    const float d = *(const float*) (q8.data() + (size_t) ib * (4 + 256 + 32));
                    yq = (double) ((const int8_t*) (q8.data() + (size_t) ib * (4 + 256 + 32) + 4))[j % 256] * d;
                }
                ref += (double) deq[(size_t) (r * c.cols + j)] * yq;
            }
            double mag = 0;
            for (int64_t j = 0; j < c.cols; ++j) mag += std::fabs((double) deq[(size_t) (r * c.cols + j)]);
            const double rel = std::fabs(s - ref) / (mag + 1e-30);
            worst = std::max(worst, rel);
        }
        std::printf("  dot vs double: worst rel %.2e\n", worst);
        if (!(worst < 1e-6)) fail("dot differs from the double reference");
        } else {
            std::printf("  dot: skipped (a %lld-wide row has a tail block; the engine rounds to 256)\n",
                        (long long) c.cols);
        }

        if (!have_ref) continue;

        // 1. bytes, 2. dequant, against ik
        const fs::path ref_bin = work / ("ref-" + std::to_string(c.type) + "-" + std::to_string(c.cols) +
                                         (c.imatrix ? "w" : "n") + ".bin");
        const fs::path ref_f32 = work / ("ref-" + std::to_string(c.type) + "-" + std::to_string(c.cols) +
                                         (c.imatrix ? "w" : "n") + ".f32");
        // _spawnv: no shell, so no quoting to get wrong.  The strings outlive the call: the temporary
        // `p.string()` results would be freed before the child runs.
        const std::string type_s = std::to_string(ik_id(c.type));
        const std::string rows_s = std::to_string(c.rows);
        const std::string cols_s = std::to_string(c.cols);
        const std::string imx_s = c.imatrix ? imx_p.string() : std::string("-");
        const std::string in_s = in_p.string();
        const std::string bin_s = ref_bin.string();
        const std::string f32_s = ref_f32.string();
        const std::string exe_s = ref_exe.string();
        // argv[0] is the name the child sees, not the file spawned: a full path with spaces would be split
        // again by the child's parser (three extra argv entries, observed on this machine).
        const std::string argv0_s = ref_exe.filename().string();
        const char* args[] = {argv0_s.c_str(),    type_s.c_str(), rows_s.c_str(), cols_s.c_str(),
                              in_s.c_str(),       imx_s.c_str(),  bin_s.c_str(),  f32_s.c_str(),
                              nullptr};
        const intptr_t rc = _spawnv(_P_WAIT, exe_s.c_str(), args);
        if (rc != 0) {
            fail("the ik reference binary failed (" + std::to_string((long long) rc) + ")");
            continue;
        }
        std::vector<char> ref;
        if (!read_file(ref_bin, ref) || ref.size() != ours.size()) {
            fail("reference byte count differs (" + std::to_string(ref.size()) + " vs " +
                 std::to_string(ours.size()) + ")");
            continue;
        }
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] != ours[i]) {
                if (!bad) first = i;
                ++bad;
            }
        }
        if (bad) {
            fail("quantize is not bit-exact: " + std::to_string(bad) + " of " + std::to_string(ref.size()) +
                 " bytes differ, first at " + std::to_string(first));
        } else {
            std::printf("  quantize: bit-exact vs ik (%zu bytes)\n", ref.size());
        }

        std::vector<char> reff32;
        if (!read_file(ref_f32, reff32) || reff32.size() != deq.size() * sizeof(float)) {
            fail("reference dequant size");
            continue;
        }
        const float* rf = (const float*) reff32.data();
        const float scale = dequant_factor(c.type);
        size_t dbad = 0;
        double dmax = 0;
        for (size_t i = 0; i < deq.size(); ++i) {
            const float want = rf[i] * scale;
            if (std::memcmp(&want, &deq[i], 4) != 0) {
                ++dbad;
                dmax = std::max(dmax, (double) std::fabs(want - deq[i]));
            }
        }
        if (dbad) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "dequant is not bit-exact: %zu of %zu values, max diff %.3g",
                          dbad, deq.size(), dmax);
            fail(buf);
        } else {
            std::printf("  dequant: bit-exact vs ik to_float x %.2f\n", scale);
        }
    }

    if (bench) {
        constexpr int64_t rows = 64, cols = 3072;
        std::printf("\nquantize speed (%lldx%lld f32 expert rows, 8 threads)\n", (long long) rows, (long long) cols);
        const auto big = make_tensor(rows, cols, 0xfeedfaceull);
        const auto w = make_imatrix(cols, 0xdeadbeefull);
        for (int type : {kIQ2KT, kIQ3KT, kIQ3K, kIQ4KS, kIQ2KL}) {
            std::vector<char> out((size_t) rows * strata::kernels::ikq::row_bytes(type, cols));
            quantize_rows(type, big.data(), out.data(), rows, cols, w.data(), 8);   // warm: codebook
            auto t0 = std::chrono::steady_clock::now();
            quantize_rows(type, big.data(), out.data(), rows, cols, w.data(), 8);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  %-7s %.2f s, %s of f32 input, %.4f bpw\n", type_name(type), secs,
                        mbs((double) big.size() * sizeof(float), secs).c_str(),
                        8.0 * (double) out.size() / (double) ((int64_t) rows * cols));
        }
    }

    std::printf("\n%s\n", g_failures ? "FAILED" : "ok");
    return g_failures ? 1 : 0;
}
