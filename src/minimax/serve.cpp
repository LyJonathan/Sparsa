// src/minimax/serve.cpp - the P2a MiniMax engine skeleton: GGUF inspection and the server's line protocol.
//
// There is no forward pass yet (P2b): this accepts the main engine's command line (and, in --serve mode, ignores
// everything it does not know, because the server's config is shared with the Qwen families), prints the INFO /
// READY handshake, and answers GEN/GENI with one ERR line and no DONE.  That is enough for serve/server.py to
// launch it, see READY, and get a clean refusal instead of a hang.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "strata/minimax/engine.hpp"
#include "strata/minimax/gguf.hpp"

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"   // the CMake build defines the real version globally
#endif

namespace strata::minimax {
namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: strata-minimax --model GGUF --minimax-inspect\n"
                 "       strata-minimax --model GGUF --serve [--max-context N] [strata's other flags]\n"
                 "       (--minimax-allow-nonstandard serves a reduced test GGUF; --serve otherwise refuses it)\n"
                 "P2a scaffold: loads and validates the GGUF's tensor directory; the forward pass is not implemented.\n");
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

}  // namespace

int run(int argc, char** argv) {
    std::string model;
    int64_t ctx_asked = 0;
    bool serve_mode = false, inspect = false, allow_nonstandard = false;

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
        else if (a == "--minimax-allow-nonstandard") allow_nonstandard = true;   // development: serve a reduced test file
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx_asked = std::atoll(next().c_str());
        // The main engine's names where they mean the same thing here.  A forward pass will use them; the scaffold
        // records the ones that affect the handshake and consumes the rest.
        else if (a == "--prefill" || a == "--chunk") (void) next();
        else if (a == "--gpu") (void) next();   // already applied by strata before the family switch
        else if (a == "--max-new" || a == "--n" || a == "--tokens" || a == "--force" || a == "--pool-workers" ||
                 a == "--mtp" || a == "--batch" || a == "--slots" || a == "--cpu-threads" || a == "--expert-vram" ||
                 a == "--expert-slots" || a == "--hybrid-rows" || a == "--draft-vocab" || a == "--bench-pp" ||
                 a == "--bench-tg" || a == "--emb" || a == "--dump-full-logits")
            (void) next();
        else if (a == "--no-warmup" || a == "--no-token-graph" || a == "--no-graph" || a == "--greedy" ||
                 a == "--vision" || a == "--batch-mtp" || a == "--no-fit-max-tokens")
            continue;
        else if (serve_mode && starts_with(a, "--")) {
            // serve passes the config's args plus a few of its own; an option this engine does not have is said and
            // skipped (with its value), never fatal.  This is what q36 does and what makes one config usable.
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

    Model m;
    try {
        m = load_model(model, !allow_nonstandard);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-minimax: %s\n", e.what());
        return 1;
    }
    if (!serve_mode) {
        std::printf("%s\n", summary(m.config).c_str());
        std::fprintf(stderr, "strata-minimax: only --serve and --minimax-inspect are implemented in this build\n");
        return 0;
    }

    int64_t ctx = m.config.context;
    if (ctx_asked > 0) ctx = (std::min)(ctx, ctx_asked);
    if (ctx < 1) ctx = 32768;

    std::printf("INFO engine=minimax-m3 version=%s state=scaffold forward=none%s\n", STRATA_VERSION,
                allow_nonstandard ? " geometry=nonstandard" : "");
    std::printf("INFO ctx=%lld layers=%lld embd=%lld experts=%lld\n", (long long) ctx, (long long) m.config.n_layers,
                (long long) m.config.n_embd, (long long) m.config.n_expert);
    std::printf("READY %lld stop\n", (long long) ctx);
    std::fflush(stdout);
    std::fprintf(stderr, "strata serve: minimax-m3 scaffold ready (ctx %lld); GEN/GENI refuse until the forward pass "
                         "lands\n", (long long) ctx);

    bool stop = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "STOP") {
            stop = true;
            continue;
        }
        if (line == "QUIT") return 0;
        if (starts_with(line, "GEN ") || starts_with(line, "GENI ") || starts_with(line, "BGEN ") ||
            starts_with(line, "BGENI ")) {
            (void) stop;
            std::printf("ERR minimax engine: forward pass not implemented yet\n");
            std::fflush(stdout);
            continue;
        }
        if (line == "BSTOP" || starts_with(line, "BSTOP ") || starts_with(line, "BYIELD")) continue;
        // One answer line per control command; anything else the server may send (VRAM, SAVE, RESTORE) is refused.
        std::printf("ERR minimax engine: '%s' is not implemented yet\n", line.c_str());
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace strata::minimax
