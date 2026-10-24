// include/strata/minimax/forward.hpp - MiniMax-M3.1-Flash forward pass (P2b).
//
// One GPU plus the CPU. Dense tensors for the first `--minimax-dense-layers-gpu` layers live on the device in
// their GGUF bytes (F32/F16/BF16 expanded to fp32, quantized types dequantized per tile). Every other dense
// matrix, and every routed expert, runs on the CPU. Routed experts use ggml-cpu's `from_float` + `vec_dot` and
// the swigluoai activation from the reference spec. KV (K, V, indexer K) is fp16 on the host. A device copy is
// not allocated until a GPU attention kernel reads it. `--minimax-dense-layers-gpu 0` makes no CUDA calls while
// the expert tiers are off; `--minimax-tiers` (or an explicit tier budget) starts CUDA for the VRAM expert slots
// even then, and keeps every dense matrix on the CPU.
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
    int pool_spin_us = 0;        ///< P7b: micro-seconds an idle pool worker spins before sleeping (0 = condvar only)
    int chunk = 16;              ///< prompt tokens between PP lines
    bool allow_nonstandard = false;

    // ---- P3: the routed-expert tiers (VRAM / pinned RAM / SSD).  Off by default: every routed expert then runs on
    // the CPU from the mapped GGUF, exactly as P2 did.  `--minimax-tiers` (or any explicit slot budget, or a
    // profile) turns them on.
    bool tiers = false;
    std::string profile_path;       ///< STRP profile to fill the static tiers from (57 x 128 for M3.1)
    std::string profile_save_path;  ///< record routing counts and write an STRP profile when the engine closes
    std::string routing_counts_path;///< also write a text "layer expert count" table beside the profile
    int vram_slots_per_layer = -1;  ///< -1: auto from free VRAM after dense/KV/workspace and the reserve; 0: none
    int ram_experts_per_layer = -1; ///< -1: auto from RAM minus the reserve; 0: no pinned tier
    int vram_reserve_mib = 700;     ///< VRAM kept free for the driver / display after dense weights and workspace
    int memory_limit_mib = 0;       ///< 0: auto from available RAM; else the pinned tier's byte budget
    int ram_headroom_mib = 0;       ///< 0: 1/16 of physical RAM clamped to 3-10 GiB; else RAM kept unpinned
    std::string ram_mode;           ///< "" / "pinned": the RAM tier is a pinned copy; "inplace": the OS page cache
                                    ///< is the tier and the CPU lane reads the mapping (nothing is copied)
    bool adaptive = true;           ///< decayed-count VRAM swaps and the SSD-miss LRU in the RAM tier
    bool stats = false;             ///< per-token tier counters on stderr
    bool stats_per_layer = false;   ///< one counters line per layer as a token runs

    // ---- parity probes (P4).  Both are opt-in and add nothing to a normal run.
    std::string dump_dir;           ///< write per-layer activations as npy files here (empty: off)
    std::string dump_layers;        ///< "all" (default when dump_dir is set) or source layer ids "0,3,40"
    std::string seed_hidden;        ///< npy fp32 [seq, n_embd]: use row t as token t's input instead of the embedding

    // ---- P4: more than one GPU, and where the KV cache lives.
    std::string gpus;               ///< CUDA ordinals to use, comma separated ("0,1"); empty: device 0
    std::string layer_split;        ///< "auto" or K[,K2]: layer K is the first one on the next GPU
    std::string kv_space;           ///< "host" (CPU attention, the P2 default) | "vram" | "ram" | "auto"
    // ---- P8: long context.  `kv_quant` halves the host/pinned KV (q8_0 blocks, 34 B / 32 values); the indexer
    // keys stay fp16 because they only rank blocks and a flipped rank changes the output discretely.
    std::string kv_quant;           ///< "" / "f16" (default, byte-identical) | "q8_0" | "auto" (q8 only when f16
                                    ///< does not fit)
    std::string host_attn;          ///< "" / "fast": the multithreaded, allocation-free host attention (same
                                    ///< arithmetic order, bit-identical); "legacy": the P2 loop
    std::string prefill_kv;         ///< "" / "auto": with the KV in pinned RAM, stage one layer's K/V on the card
                                    ///< for the batched prompt pass instead of running attention on the host;
                                    ///< "off": keep the host attention in prefill (P7 behaviour)

    // ---- P5a: batched prefill.  0 keeps the token path (byte-identical to P4); a positive value is the maximum
    // prompt tokens processed per layer in one batched pass.  The prompt is then walked in chunks of that size.
    int prefill_batch = 0;
    std::string prefill_experts;    ///< "auto" (chunk size decides), "cpu" or "gpu": where a batched chunk's routed
                                    ///< experts are computed
    std::string prefill_dense;      ///< "f32" (default: fp32 cuBLAS, matches the per-token GPU GEMV) or "f16"
    std::string prefill_read;       ///< "" / "auto": the OS page cache, except when --memory-limit-mib is set (the
                                    ///< budget cannot bound the page cache, so streamed experts are read unbuffered);
                                    ///< "mmap": always the page cache; "direct": always unbuffered 16 MiB reads
    // ---- P5b: decode.  Copy the next layer's routed experts into free VRAM slots while this layer computes.
    int expert_prefetch = 0;        ///< experts per layer to prefetch ahead (0 = off)
    std::string dense_mmvq;         ///< "" / "on": packed dense GEMV through CUDA MMVQ (q8_1 activation);
                                    ///< "off": the fp32 dequantize + SGEMM path, bit-closer to the CPU dots
    // ---- P5c: the fused device norm+rope step and the resident expert arena.
    std::string gpu_normrope;       ///< "" / "on": per-head RMSNorm + RoPE + the fp16 cache write on the device;
                                    ///< "off": the host rms_heads/rope/ggml_fp32_to_fp16 loop (the P5b path)
    float expert_skip = 0.f;        ///< --minimax-expert-skip: drop a routed expert whose renormalized weight is
                                    ///< below this; the survivors are renormalized. 0 (default) keeps every route.
    std::string graphs;             ///< "" / "on": CUDA-graph the token path's per-layer GPU prefix; "off": launch
                                    ///< the same kernels one at a time
    std::string expert_resident;    ///< "" / "auto": pin the RAM tier at startup when it fits; "all": also pin the
                                    ///< mmap-only experts (everything the RAM budget can hold); "off": on demand
    // ---- P6: the SSD tier's read path (a model larger than RAM).
    int ssd_chunks = 1;             ///< pieces one expert's read is split into, in flight together (1-16).  1 = one
                                    ///< read a part (the default: on this box's NVMe the whole 5.5 MiB role read is
                                    ///< 1.9x faster than eight 700 KiB pieces, 261 MB/token of misses: 46 ms of
                                    ///< waiting against 110 ms).  A drive that prefers many small requests gains
                                    ///< from 2-8 (Project Maya measured +15% at 8 on a Windows laptop's RAID).
    int expert_skip_ssd = 0;        ///< drop a route at or below this rank (0 = best) when its expert is neither in
                                    ///< VRAM nor in the pinned RAM tier, i.e. it would be read from the SSD.  0 keeps
                                    ///< every route; 4 of MiniMax's top-4 keeps only the top three misses.
    // ---- P7: the SSD tier's page-cache path.
    std::string ssd_zerocopy;       ///< "" / "on": an expert whose pages are already resident is computed from the
                                    ///< mapping with no read; "off": always issue the read (P6 behaviour)
    std::string ssd_prefetch;       ///< "" / off: "on" queues the next layer's SSD misses right after this layer's
                                    ///< MoE, so they are in flight across that layer's attention
    // ---- P7b: the route nudge (Xolotl's selection bias toward resident experts).
    std::string route_bias;         ///< "" / off: "on" ranks the router's top-k on the expert's score plus a
                                    ///< residency bonus / file penalty; the weights stay unbiased
    float route_bias_in = 0.25f;    ///< bonus for an expert in the VRAM or pinned-RAM tier
    float route_bias_out = 0.15f;   ///< penalty for an expert in neither (it would be read from the file)
    int route_protect = 2;          ///< the exact top-N experts are never displaced
    int route_pool = 12;            ///< a pick must stay inside the exact top-N
    float route_margin = 0.f;       ///< 0: nudge every expert in the pool; else only within this of the cutoff
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
    /// One human line: placement, norm convention, prefill mode, tier sizes.
    std::string describe() const;
    /// Tier counters as one "key=value ..." line (empty when the tiers are off).
    std::string stats() const;
    /// Write the profile the run collected, if --minimax-profile-save was given (also called by the destructor).
    void save_profile();

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
