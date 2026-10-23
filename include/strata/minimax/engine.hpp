// include/strata/minimax/engine.hpp - the MiniMax-M3.1-Flash (GGUF architecture minimax-m3) engine path as a library:
// `strata` hands a minimax-m3 model to it at the top of main() (the family switch), and strata-minimax is a thin
// wrapper around the same entry point.  See src/minimax/.
#pragma once

namespace strata::minimax {

/// The engine's command line: strata's flag names where they mean the same thing (--native / --model,
/// --max-context, --prefill / --chunk, --serve, --gpu, --batch, ...) plus the family's own --minimax-inspect.
/// Unknown options are ignored in --serve mode (the server's config is shared with the Qwen families).
/// Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::minimax
