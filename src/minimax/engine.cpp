// src/minimax/engine.cpp - MiniMax-M3.1-Flash forward pass.
//
// Token-by-token, prefill and decode share one path (P2b; a batched prefill can replace it later without changing
// the math). Dense GEMVs are fp32. Quantized routed experts are ggml-cpu vec_dot (activation quantized with the
// type's vec_dot_type) plus swigluoai. KV is fp16. See include/strata/minimax/forward.hpp.
#include "strata/minimax/forward.hpp"

#include "cuda_ops.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dequant_bf16.hpp"

namespace strata::minimax {
namespace {

constexpr int kF32 = 0;
constexpr int kF16 = 1;
constexpr int kBF16 = 30;
constexpr int kTile = 128;

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
};

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

int64_t bigger(int64_t a, int64_t b) { return a > b ? a : b; }

void parallel_for(int threads, int rows, const std::function<void(int, int)>& fn) {
    if (rows <= 0) return;
    if (threads <= 1 || rows < threads * 4) {
        fn(0, rows);
        return;
    }
    const int parts = threads;
    std::vector<std::thread> th;
    th.reserve((size_t)parts - 1);
    for (int p = 1; p < parts; ++p) {
        const int a = (int)((int64_t)rows * p / parts);
        const int b = (int)((int64_t)rows * (p + 1) / parts);
        th.emplace_back([&, a, b] {
            if (a < b) fn(a, b);
        });
    }
    const int b0 = (int)((int64_t)rows / parts);
    if (b0 > 0) fn(0, b0);
    for (auto& t : th) t.join();
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

    std::vector<uint16_t> k_host, v_host, ik_host;
    DevPtr k_dev, v_dev, ik_dev;
    DevPtr d_x, d_y, d_scratch;
    int64_t max_cols = 1, max_rows = 1;

    std::vector<float> x, h, q, k, v, proj, ff_g, ff_u, ff_h, ff_y, iq, ik, logits, last_logits;
    std::vector<int32_t> held;
    int64_t n_past = 0;
    bool have_logits = false;
    std::mt19937_64 rng{0};

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
        ctx = cfg.context;
        if (opt.max_context > 0 && opt.max_context < ctx) ctx = opt.max_context;
        if (ctx < 1) fail("context length is 0");

        ggml_cpu_init();
        load_tensors();
        decide_norm();
        build_layers();
        read_eos();

        threads = opt.cpu_threads;
        if (threads <= 0) threads = (int)std::thread::hardware_concurrency();
        if (threads <= 0) threads = 4;
        if (threads > 24) threads = 24;

        cu::init();
        alloc_cache();
        alloc_workspace();
        place_weights();

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
        logits.assign((size_t)cfg.n_vocab, 0.f);
        last_logits.assign((size_t)cfg.n_vocab, 0.f);
    }

    Tensor must(const std::string& name) {
        size_t shard = 0;
        const TensorInfo* info = model.find(name, &shard);
        if (!info) fail("missing tensor " + name);
        Tensor t;
        t.name = name;
        t.type = (int)info->type;
        t.ne0 = (int64_t)info->shape[0];
        t.ne1 = info->shape.size() > 1 ? (int64_t)info->shape[1] : 1;
        t.ne2 = info->shape.size() > 2 ? (int64_t)info->shape[2] : 1;
        if (info->shape.size() > 3) fail(name + " has more than 3 dimensions");
        t.host = model.shard(shard).tensor_data(*info);
        t.bytes = tensor_payload_bytes(*info);
        t.row_bytes = ggml_row_size((ggml_type)t.type, t.ne0);
        if (t.row_bytes == 0 || t.row_bytes * (size_t)t.ne1 * (size_t)t.ne2 != t.bytes)
            fail(name + ": row size does not match the GGUF payload (type " + std::to_string(t.type) + ")");
        t.base = ggml_get_type_traits((ggml_type)t.type);
        t.cpu = ggml_get_type_traits_cpu((ggml_type)t.type);
        if (!t.base) fail(name + ": ggml has no type traits for type " + std::to_string(t.type));
        if (t.type != kF32 && !t.base->to_float)
            fail(name + ": no ggml to_float for type " + std::to_string(t.type));
        ggml_quantize_init((ggml_type)t.type);
        if (t.cpu && t.cpu->vec_dot_type != (ggml_type)t.type) ggml_quantize_init(t.cpu->vec_dot_type);
        return t;
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

    void alloc_cache() {
        const size_t kp = (size_t)cfg.n_layers * (size_t)ctx * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        const size_t ip = (size_t)cfg.n_layers * (size_t)ctx * (size_t)cfg.idx_dim;
        try {
            k_host.assign(kp, 0);
            v_host.assign(kp, 0);
            ik_host.assign(ip, 0);
            k_dev = DevPtr(cu::alloc(kp * sizeof(uint16_t)));
            v_dev = DevPtr(cu::alloc(kp * sizeof(uint16_t)));
            ik_dev = DevPtr(cu::alloc(ip * sizeof(uint16_t)));
        } catch (const std::exception& e) {
            const double mib = (double)((kp + kp + ip) * sizeof(uint16_t)) / (1024.0 * 1024.0);
            fail(std::string(e.what()) + " (fp16 KV for context " + std::to_string(ctx) + " needs about " +
                 std::to_string((int)mib) + " MiB on the GPU and the same on the host; lower --max-context)");
        }
    }

    void alloc_workspace() {
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
        d_x = DevPtr(cu::alloc((size_t)max_cols * sizeof(float)));
        d_y = DevPtr(cu::alloc((size_t)max_rows * sizeof(float)));
        d_scratch = DevPtr(cu::alloc((size_t)kTile * (size_t)max_cols * sizeof(float)));
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
        std::vector<size_t> cost((size_t)cfg.n_layers);
        for (int64_t i = 0; i < cfg.n_layers; ++i) cost[(size_t)i] = layer_cost(layers[(size_t)i]);
        const size_t out_b = gpu_bytes(output);
        const cu::MemInfo mi = cu::mem();
        const size_t reserve = (size_t)512 << 20;
        const size_t budget = mi.free_bytes > reserve ? mi.free_bytes - reserve : 0;
        const bool automatic = opt.dense_layers_gpu < 0;
        int N = automatic ? 0 : opt.dense_layers_gpu;
        if (N > (int)cfg.n_layers) N = (int)cfg.n_layers;
        if (automatic) {
            size_t used = 0;
            while (N < (int)cfg.n_layers && used + cost[(size_t)N] <= budget) {
                used += cost[(size_t)N];
                ++N;
            }
        } else if (N > 0) {
            size_t used = 0;
            for (int i = 0; i < N; ++i) used += cost[(size_t)i];
            if (used > budget) {
                fail("the first " + std::to_string(N) + " dense layers need " +
                     std::to_string(used / (1024 * 1024)) + " MiB of GPU weight storage but only " +
                     std::to_string(budget / (1024 * 1024)) +
                     " MiB is free after KV and a 512 MiB reserve. Lower --minimax-dense-layers-gpu.");
            }
        }
        dense_gpu = N;
        for (int i = 0; i < N; ++i) {
            Layer& L = layers[(size_t)i];
            L.want_gpu = true;
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
        }
        if (N > 0 && out_b != std::numeric_limits<size_t>::max()) {
            const cu::MemInfo now = cu::mem();
            if (now.free_bytes > out_b + ((size_t)256 << 20)) {
                upload(output);
                output_gpu = true;
            }
        }
    }

    void gemv(const Tensor& t, const float* xin, float* yout) {
        if (t.ne2 != 1) fail(t.name + ": internal gemv is for a matrix, not a stacked expert");
        const int rows = (int)t.ne1;
        const int cols = (int)t.ne0;
        if ((int64_t)cols > max_cols || (int64_t)rows > max_rows) fail(t.name + ": gemv exceeds the workspace");
        if (t.gpu) {
            cu::copy_h2d(d_x.p, xin, (size_t)cols * sizeof(float));
            float* dy = (float*)d_y.p;
            if (t.dev_f32) {
                cu::gemv((const float*)t.dev.p, (const float*)d_x.p, dy, rows, cols);
            } else {
                for (int r = 0; r < rows; r += kTile) {
                    const int n = rows - r < kTile ? rows - r : kTile;
                    cu::dequant_f32(t.type, t.dev.p, r, n, cols, (float*)d_scratch.p);
                    cu::gemv((const float*)d_scratch.p, (const float*)d_x.p, dy + r, n, cols);
                }
            }
            cu::copy_d2h(yout, dy, (size_t)rows * sizeof(float));
            return;
        }
        if (t.type == kF32 || t.type == kF16 || t.type == kBF16) {
            parallel_for(threads, rows, [&](int a, int b) {
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
        parallel_for(threads, rows, [&](int a, int b) {
            for (int r = a; r < b; ++r) {
                float s = 0.f;
                dot(cols, &s, 0, t.host + (size_t)r * t.row_bytes, 0, aq.data(), 0, 1);
                yout[r] = s;
            }
        });
    }

    void dot_rows(const Tensor& t, int64_t expert, const float* xin, float* yout, int rows_n, int cols) {
        // One expert (or a plain matrix when expert == 0 and ne2 == 1): yout[r] = row(expert, r) · xin.
        if (t.type == kF32 || t.type == kF16 || t.type == kBF16) {
            parallel_for(threads, rows_n, [&](int a, int b) {
                for (int r = a; r < b; ++r) {
                    const uint8_t* row = row_ptr(t, expert, r);
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
        parallel_for(threads, rows_n, [&](int a, int b) {
            for (int r = a; r < b; ++r) {
                float s = 0.f;
                dot(cols, &s, 0, row_ptr(t, expert, r), 0, aq.data(), 0, 1);
                yout[r] = s;
            }
        });
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
             int ff) {
        if ((int)ff_g.size() < ff) fail("MLP wider than the workspace");
        apply(gate, expert, in, ff_g.data(), ff, (int)gate.ne0);
        apply(up, expert, in, ff_u.data(), ff, (int)up.ne0);
        for (int i = 0; i < ff; ++i) ff_h[i] = swigluoai(ff_g[i], ff_u[i], alpha, limit);
        apply(down, expert, ff_h.data(), yout, (int)down.ne1, ff);
    }

    void moe(const Layer& L, const float* in, float* yout) {
        const int nexp = (int)cfg.n_expert;
        const int top = (int)cfg.n_expert_used;
        std::vector<float> logits_r((size_t)nexp), prob((size_t)nexp), score((size_t)nexp);
        gemv(L.router, in, logits_r.data());
        for (int i = 0; i < nexp; ++i) {
            prob[(size_t)i] = sigmoid(logits_r[(size_t)i]);
            score[(size_t)i] = prob[(size_t)i] + L.bias[(size_t)i];
        }
        std::vector<int> order(nexp);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](int a, int b) {
            if (score[(size_t)a] != score[(size_t)b]) return score[(size_t)a] > score[(size_t)b];
            return a < b;
        });
        std::vector<float> w((size_t)top);
        float sum = 0.f;
        for (int j = 0; j < top; ++j) {
            w[(size_t)j] = prob[(size_t)order[(size_t)j]];
            sum += w[(size_t)j];
        }
        if (renorm && sum > 0.f)
            for (float& v : w) v /= sum;
        std::fill(yout, yout + cfg.n_embd, 0.f);
        std::vector<float> ye((size_t)cfg.n_embd);
        const int ff = (int)cfg.n_ff_exp;
        for (int j = 0; j < top; ++j) {
            mlp(L.egate, L.eup, L.edown, order[(size_t)j], in, ye.data(), ff);
            const float WJ = w[(size_t)j];
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += WJ * ye[(size_t)i];
        }
        const float s = scale_routed;
        for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] *= s;
        if (L.have_shared) {
            mlp(L.gate, L.up, L.down, 0, in, ye.data(), (int)L.gate.ne1);
            for (int64_t i = 0; i < cfg.n_embd; ++i) yout[i] += ye[(size_t)i];
        }
    }

    const uint16_t* k_at(int layer, int64_t pos, int kv) const {
        const size_t stride = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        return k_host.data() + ((size_t)layer * (size_t)ctx + (size_t)pos) * stride + (size_t)kv * (size_t)cfg.head_dim;
    }
    const uint16_t* v_at(int layer, int64_t pos, int kv) const {
        const size_t stride = (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
        return v_host.data() + ((size_t)layer * (size_t)ctx + (size_t)pos) * stride + (size_t)kv * (size_t)cfg.head_dim;
    }
    const uint16_t* ik_at(int layer, int64_t pos) const {
        return ik_host.data() + ((size_t)layer * (size_t)ctx + (size_t)pos) * (size_t)cfg.idx_dim;
    }

    void store_f16(std::vector<uint16_t>& host, uint16_t* dev, size_t off, const float* src, int n) {
        for (int i = 0; i < n; ++i) host[off + (size_t)i] = ggml_fp32_to_fp16(src[i]);
        cu::copy_h2d(dev + off, host.data() + off, (size_t)n * sizeof(uint16_t));
    }

    void attend(int layer, const Layer& L, int64_t pos, const float* qq, float* att) {
        const int H = (int)cfg.n_head;
        const int HKV = (int)cfg.n_head_kv;
        const int D = (int)cfg.head_dim;
        const int64_t Tk = pos + 1;
        const int q_per_kv = H / HKV;
        const float scale = 1.f / std::sqrt((float)D);
        std::vector<char> sel;
        int nb = 0;
        if (L.indexer) {
            const int HI = (int)cfg.idx_heads;
            const int BLK = (int)cfg.idx_block;
            nb = (int)((Tk + BLK - 1) / BLK);
            std::vector<float> bs((size_t)HI * nb, -std::numeric_limits<float>::infinity());
            for (int h = 0; h < HI; ++h) {
                const float* qh = iq.data() + (size_t)h * cfg.idx_dim;
                for (int b = 0; b < nb; ++b) {
                    const int64_t begin = (int64_t)b * BLK;
                    const int64_t end = begin + BLK < Tk ? begin + BLK : Tk;
                    float m = -std::numeric_limits<float>::infinity();
                    for (int64_t t = begin; t < end; ++t) {
                        if (t > pos) break;
                        const float sc = dot_f16_cache(ik_at(layer, t), qh, (int)cfg.idx_dim);
                        if (sc > m) m = sc;
                    }
                    bs[(size_t)h * nb + b] = m;
                }
                const int qb = (int)(pos / BLK);
                for (int l = 0; l < (int)cfg.idx_local; ++l) {
                    int bi = qb - l;
                    if (bi < 0) bi = 0;
                    if (bi < nb) bs[(size_t)h * nb + bi] = std::numeric_limits<float>::infinity();
                }
            }
            const int kk = (int)cfg.idx_top_k < nb ? (int)cfg.idx_top_k : nb;
            sel.assign((size_t)HI * nb, 0);
            for (int h = 0; h < HI; ++h) {
                std::vector<int> order(nb);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int a, int b) {
                    const float sa = bs[(size_t)h * nb + a];
                    const float sb = bs[(size_t)h * nb + b];
                    if (sa != sb) return sa > sb;
                    return a < b;
                });
                for (int i = 0; i < kk; ++i) {
                    const int b = order[(size_t)i];
                    if (bs[(size_t)h * nb + b] == -std::numeric_limits<float>::infinity()) continue;
                    sel[(size_t)h * nb + b] = 1;
                }
            }
        }
        const int q_per_idx = L.indexer ? H / (int)cfg.idx_heads : H;
        std::vector<float> prob((size_t)Tk);
        for (int qh = 0; qh < H; ++qh) {
            const int kv = qh / q_per_kv;
            const int ih = L.indexer ? qh / q_per_idx : 0;
            const float* qq = q.data() + (size_t)qh * D;
            for (int64_t t = 0; t < Tk; ++t) {
                bool keep = t <= pos;
                if (keep && L.indexer) keep = sel[(size_t)ih * nb + (size_t)(t / cfg.idx_block)] != 0;
                if (!keep) prob[(size_t)t] = -std::numeric_limits<float>::infinity();
                else prob[(size_t)t] = dot_f16_cache(k_at(layer, t, kv), qq, D) * scale;
            }
            softmax_inplace(prob.data(), (int)Tk);
            float* out = att + (size_t)qh * D;
            for (int d = 0; d < D; ++d) out[d] = 0.f;
            for (int64_t t = 0; t < Tk; ++t) {
                const float p = prob[(size_t)t];
                if (p == 0.f) continue;
                const uint16_t* vv = v_at(layer, t, kv);
                for (int d = 0; d < D; ++d) out[d] += p * ggml_fp16_to_fp32(vv[d]);
            }
        }
        (void)qq;
    }

    void forward_token(int32_t token, float* logit_out) {
        if (token < 0 || (int64_t)token >= cfg.n_vocab) fail("token id out of range");
        if (n_past >= ctx) fail("context is full (" + std::to_string(ctx) + ")");
        const int64_t pos = n_past;
        decode_f32(embd, embd.host + (size_t)token * embd.row_bytes, x.data(), cfg.n_embd);

        for (int64_t li = 0; li < cfg.n_layers; ++li) {
            Layer& L = layers[(size_t)li];
            rms(x.data(), L.attn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            gemv(L.wq, h.data(), q.data());
            gemv(L.wk, h.data(), k.data());
            gemv(L.wv, h.data(), v.data());
            rms_heads(q.data(), (int)cfg.n_head, (int)cfg.head_dim, L.q_norm.data(), eps);
            rms_heads(k.data(), (int)cfg.n_head_kv, (int)cfg.head_dim, L.k_norm.data(), eps);
            rope(q.data(), (int)cfg.n_head, (int)cfg.head_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
            rope(k.data(), (int)cfg.n_head_kv, (int)cfg.head_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
            const size_t koff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.n_head_kv * (size_t)cfg.head_dim;
            store_f16(k_host, (uint16_t*)k_dev.p, koff, k.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            store_f16(v_host, (uint16_t*)v_dev.p, koff, v.data(), (int)(cfg.n_head_kv * cfg.head_dim));
            if (L.indexer) {
                gemv(L.iq, h.data(), iq.data());
                gemv(L.ik, h.data(), ik.data());
                rms_heads(iq.data(), (int)cfg.idx_heads, (int)cfg.idx_dim, L.iq_norm.data(), eps);
                rms(ik.data(), L.ik_norm.data(), ik.data(), (int)cfg.idx_dim, eps);
                rope(iq.data(), (int)cfg.idx_heads, (int)cfg.idx_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
                rope(ik.data(), 1, (int)cfg.idx_dim, (int)cfg.n_rot, (int)pos, cfg.rope_theta);
                const size_t ioff = ((size_t)li * (size_t)ctx + (size_t)pos) * (size_t)cfg.idx_dim;
                store_f16(ik_host, (uint16_t*)ik_dev.p, ioff, ik.data(), (int)cfg.idx_dim);
            }
            std::vector<float> att((size_t)cfg.n_head * (size_t)cfg.head_dim);
            attend((int)li, L, pos, q.data(), att.data());
            gemv(L.wo, att.data(), proj.data());
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];

            rms(x.data(), L.ffn_norm.data(), h.data(), (int)cfg.n_embd, eps);
            if (!L.moe) mlp(L.gate, L.up, L.down, 0, h.data(), proj.data(), (int)cfg.n_ff_dense);
            else moe(L, h.data(), proj.data());
            for (int64_t i = 0; i < cfg.n_embd; ++i) x[(size_t)i] += proj[(size_t)i];
        }
        rms(x.data(), out_norm.data(), h.data(), (int)cfg.n_embd, eps);
        gemv(output, h.data(), logit_out);
        ++n_past;
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
            float cut = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < sp.top_k; ++i) {
                const float v = row[(size_t)order[(size_t)i]];
                if (v < cut || cut == -std::numeric_limits<float>::infinity()) cut = v;
            }
            // nth_element leaves the k-th at order[k] in the "not less" partition; the cutoff is the
            // worst of the kept side, which nth_element places at index top_k (the first of the rest) ... 
            // Use the minimum among the first top_k of the nth partition: they are the top_k.
            cut = row[(size_t)order[0]];
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
    o << "prefill=token dense_gpu_layers=" << impl_->dense_gpu << " output_gpu=" << (impl_->output_gpu ? 1 : 0)
      << " experts=cpu kv=f16 norm_add1=" << (impl_->norm_add1 ? 1 : 0) << " threads=" << impl_->threads
      << " ctx=" << impl_->ctx;
    return o.str();
}

void Engine::teacher_forced(const std::vector<int32_t>& tokens, int last_k, std::vector<float>& logits, int& rows) {
    impl_->reset();
    const int vocab = (int)impl_->cfg.n_vocab;
    const int n = (int)tokens.size();
    if (n < 1) fail("teacher-forced sequence is empty");
    if ((int64_t)n > impl_->ctx) fail("teacher-forced sequence is longer than the context");
    std::vector<float> all((size_t)n * vocab);
    for (int i = 0; i < n; ++i) {
        impl_->forward_token(tokens[(size_t)i], all.data() + (size_t)i * vocab);
        if (!finite_row(all.data() + (size_t)i * vocab, vocab))
            fail("non-finite logits at position " + std::to_string(i));
    }
    int keep = last_k <= 0 || last_k > n ? n : last_k;
    const int start = n - keep;
    logits.assign(all.begin() + (size_t)start * vocab, all.end());
    rows = keep;
}

GenerateResult Engine::generate(const std::vector<int32_t>& prompt, int max_new, const SampleParams& sp,
                                std::atomic<bool>& stop, const std::function<void(int32_t)>& on_token,
                                const std::function<void(int, int, double)>& on_pp) {
    GenerateResult gr;
    if (prompt.empty()) fail("prompt is empty");
    if ((int64_t)prompt.size() > impl_->ctx) fail("prompt is longer than the context");
    if (max_new < 0) max_new = 0;
    if (sp.have_seed) impl_->rng.seed(sp.seed);
    else {
        std::random_device rd;
        impl_->rng.seed(((uint64_t)rd() << 32) ^ rd());
    }

    const bool extend = prompt.size() >= impl_->held.size() &&
                        std::equal(impl_->held.begin(), impl_->held.end(), prompt.begin());
    if (!extend) {
        impl_->reset();
        gr.reused = 0;
    } else {
        gr.reused = (int)impl_->held.size();
    }
    gr.prompt = (int)prompt.size();

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto ms_since = [&](clock::time_point a) {
        return std::chrono::duration<double, std::milli>(clock::now() - a).count();
    };
    const int chunk = impl_->opt.chunk > 0 ? impl_->opt.chunk : 16;
    for (int i = gr.reused; i < (int)prompt.size(); ++i) {
        if (stop.load(std::memory_order_relaxed)) {
            gr.finish = "cancel";
            gr.prompt_ms = ms_since(t0);
            return gr;
        }
        impl_->forward_token(prompt[(size_t)i], impl_->last_logits.data());
        impl_->have_logits = true;
        impl_->held.push_back(prompt[(size_t)i]);
        const int done = i + 1;
        if (on_pp && (done == (int)prompt.size() || done % chunk == 0))
            on_pp(done, (int)prompt.size(), ms_since(t0));
        if (!finite_row(impl_->last_logits.data(), (int)impl_->cfg.n_vocab)) {
            gr.finish = "nonfinite";
            gr.prompt_ms = ms_since(t0);
            return gr;
        }
    }
    gr.prompt_ms = ms_since(t0);
    if (!impl_->have_logits) {
        gr.finish = "length";
        return gr;
    }

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
