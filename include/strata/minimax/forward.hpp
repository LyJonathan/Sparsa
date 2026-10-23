// include/strata/minimax/forward.hpp - MiniMax-M3.1-Flash forward pass (P2b).
//
// One GPU plus the CPU. Dense tensors for the first `--minimax-dense-layers-gpu` layers live on the device in
// their GGUF bytes (F32/F16/BF16 expanded to fp32, quantized types dequantized per tile). Every other dense
// matrix, and every routed expert, runs on the CPU. Routed experts use ggml-cpu's `from_float` + `vec_dot` and
// the swigluoai activation from the reference spec. KV (K, V, indexer K) is fp16 on the host. A device copy is
// not allocated until a GPU attention kernel reads it. `--minimax-dense-layers-gpu 0` makes no CUDA calls.
//
// Prefill walks the same per-token path as decode. Norm weights follow the llama.cpp GGUF convention: `+1` is
// baked into the tensor unless `minimax-m3.norm.add1` is true, and a file whose norms sit near 0 is refused
// unless `--minimax-allow-nonstandard` is set.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "strata/minimax/config.hpp"

namespace strata::minimax {

struct EngineOptions {
    int64_t max_context = 0;     ///< 0: the model's context, still clamped by what was asked
    int dense_layers_gpu = -1;   ///< -1: fit as many leading layers as VRAM allows; 0: all dense on the CPU
    int cpu_threads = 0;         ///< 0: the machine's hardware concurrency, capped
    int chunk = 16;              ///< prompt tokens between PP lines
    bool allow_nonstandard = false;
};

struct SampleParams {
    float temperature = 1.f;
    float top_p = 1.f;
    float min_p = 0.f;
    int top_k = 0;               ///< 0: no top-k cap
    uint64_t seed = 0;
    bool have_seed = false;
    int penalty_last_n = 64;
    float penalty_repeat = 1.f;
    float penalty_freq = 0.f;
    float penalty_present = 0.f;
};

struct GenerateResult {
    std::string finish;          ///< stop | length | cancel | nonfinite
    int generated = 0;
    int prompt = 0;
    double prompt_ms = 0;
    double decode_ms = 0;
    int reused = 0;
};

class Engine {
public:
    Engine(const std::string& path, const EngineOptions& opt);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const Config& config() const;
    int64_t context() const;
    int dense_layers_gpu() const;
    bool norm_add1() const;
    /// One human line: placement, norm convention, prefill mode.
    std::string describe() const;

    /// Teacher-forced logits, fp32, row-major. `last_k <= 0` keeps every position; otherwise the last K.
    void teacher_forced(const std::vector<int32_t>& tokens, int last_k, std::vector<float>& logits, int& rows);

    GenerateResult generate(const std::vector<int32_t>& prompt, int max_new, const SampleParams& sp,
                            std::atomic<bool>& stop, const std::function<void(int32_t)>& on_token,
                            const std::function<void(int, int, double)>& on_pp);

private:
    struct Impl;
    Impl* impl_;
};

void write_npy_f32(const std::string& path, const float* data, int rows, int cols);
std::vector<int32_t> read_token_file(const std::string& path);

}  // namespace strata::minimax
