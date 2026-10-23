// src/minimax/serve.cpp - MiniMax-M3.1-Flash inspect, teacher-forced eval, and the serve line protocol.
//
// Inspect never constructs the engine (no CUDA). Serve and eval load the GGUF, refuse a non-M3.1 file unless
// --minimax-allow-nonstandard is set, then build the forward pass. GEN streams T / PP / DONE. GENI and BGEN
// answer with ERR and no DONE. Unknown --flags are ignored only in --serve, matching the shared server config.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "strata/minimax/engine.hpp"
#include "strata/minimax/forward.hpp"
#include "strata/minimax/gguf.hpp"

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace strata::minimax {
namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: strata-minimax --model GGUF --minimax-inspect\n"
                 "       strata-minimax --model GGUF --serve [--max-context N] [--minimax-dense-layers-gpu N]\n"
                 "       strata-minimax --model GGUF --minimax-eval IDS --minimax-dump-logits OUT.npy\n"
                 "                       [--minimax-last-k K] [--minimax-dense-layers-gpu N]\n"
                 "       (--minimax-allow-nonstandard serves a reduced test GGUF; --serve otherwise refuses it)\n"
                 "Prefill is token-by-token through the decode path. Dense layers 0..N-1 go to the GPU;\n"
                 "N=0 keeps every dense matrix on the CPU. Routed experts stay on the CPU.\n");
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

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

void handle_gen(Engine& engine, const std::string& line, std::atomic<bool>& stop, bool greedy_default) {
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
    // A STOP already sitting in the pipe ahead of this GEN (the test script writes the whole session at
    // once) must not cancel it. A STOP that arrives while tokens are being produced still will.
    stop.store(false, std::memory_order_relaxed);
    try {
        const GenerateResult gr = engine.generate(
            ids, max_new, sp, stop,
            [](int32_t id) {
                std::printf("T %d\n", (int)id);
                std::fflush(stdout);
            },
            [](int done, int total, double ms) {
                const double tps = ms > 0.0 ? (done * 1000.0 / ms) : 0.0;
                std::printf("PP %d %d %.3f %.3f\n", done, total, ms, tps);
                std::fflush(stdout);
            });
        std::printf("DONE %d %d %.3f %.3f %s 0 0 %d\n", gr.generated, gr.prompt, gr.prompt_ms, gr.decode_ms,
                    gr.finish.c_str(), gr.reused);
        std::fflush(stdout);
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
        else if (a == "--prefill" || a == "--chunk") chunk = parse_chunk(next());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--pool-workers") pool_workers = std::atoi(next().c_str());
        else if (a == "--gpu") (void)next();
        else if (a == "--max-new" || a == "--n" || a == "--tokens" || a == "--force" || a == "--mtp" ||
                 a == "--batch" || a == "--slots" || a == "--expert-vram" || a == "--expert-slots" ||
                 a == "--hybrid-rows" || a == "--draft-vocab" || a == "--bench-pp" || a == "--bench-tg" ||
                 a == "--emb" || a == "--dump-full-logits")
            (void)next();
        else if (a == "--no-warmup" || a == "--no-token-graph" || a == "--no-graph" || a == "--vision" ||
                 a == "--batch-mtp" || a == "--no-fit-max-tokens")
            continue;
        else if (a == "--greedy") greedy = true;
        else if (serve_mode && starts_with(a, "--")) {
            const bool val = i + 1 < argc && !starts_with(argv[i + 1], "--");
            std::fprintf(stderr, "strata-minimax: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            usage();
            return 2;
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

    int64_t ctx = meta.config.context;
    if (ctx_asked > 0) ctx = (std::min)(ctx, ctx_asked);
    if (ctx < 1) ctx = 32768;

    EngineOptions opt;
    opt.max_context = ctx;
    opt.dense_layers_gpu = dense_gpu;
    opt.cpu_threads = pool_workers > 0 ? pool_workers : cpu_threads;
    opt.chunk = chunk;
    opt.allow_nonstandard = allow_nonstandard;

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
            write_npy_f32(dump_path, logits.data(), rows, cols);
            std::printf("LOGITS %d %d %s\n", rows, cols, dump_path.c_str());
            std::fflush(stdout);
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
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line == "STOP") {
                stop.store(true, std::memory_order_relaxed);
                continue;
            }
            lines.push(std::move(line));
        }
        {
            std::lock_guard<std::mutex> lk(lines.mu);
            lines.eof = true;
        }
        lines.cv.notify_one();
    });

    std::string line;
    while (lines.pop(line)) {
        if (line == "QUIT") break;
        if (starts_with(line, "GEN ")) {
            handle_gen(*engine, line, stop, greedy);
            continue;
        }
        if (starts_with(line, "GENI ") || starts_with(line, "GENI")) {
            std::printf("ERR minimax engine: GENI is not implemented (no vision encoder)\n");
            std::fflush(stdout);
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
    reader.detach();
    delete engine;
    return 0;
}

}  // namespace strata::minimax
