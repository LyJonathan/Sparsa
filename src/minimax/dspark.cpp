// src/minimax/dspark.cpp - the DSpark drafter's forward pass (see dspark.hpp for what it is).
//
// Every matrix is a Q8_0 GGUF tensor on the drafter's device and runs through the engine's native MMVQ path with
// `ncols` = the round's row count (1..8), so a round reads the drafter's weights once.  The 1-D weights (norms, the
// Markov pair, the confidence head) are F16 and tiny; the norms are expanded to fp32 (Gemma's `1 + w` applied) on the
// device, and the Markov/confidence rows are read from the pack's mapping on the host.
//
// The device work runs on one BLOCKING stream, which the legacy default stream (the one every cu::copy_* uses) orders
// itself against, so the host copies need no events.  The ring is a plain circular buffer of fp16 K/V rows per layer;
// only committed positions are ever written to it, so there is nothing to roll back after a rejected draft.
#include "dspark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_ops.hpp"

#include "ggml.h"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope.hpp"

namespace strata::minimax {
namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

/// Tokens within this many nats of a block row's best base logit are the ones the Markov bias is evaluated over.  A
/// draft is only a proposal the target verifies, so truncating costs acceptance, never correctness.
constexpr float kCandidateNats = 12.f;
constexpr int kCandidateCap = 8192;

float sigmoidf_(double x) { return (float)(1.0 / (1.0 + std::exp(-x))); }

namespace {
/// The caller's device, restored when a drafter call returns.
struct OnDevice {
    int prev;
    explicit OnDevice(int dev) : prev(cu::current_device()) { cu::set_device(dev); }
    ~OnDevice() { cu::set_device(prev); }
};
}  // namespace

}  // namespace

struct DsparkDrafter::Impl {
    DsparkGeometry g;
    int dev = -1;
    void* cs = nullptr;   ///< the blocking stream every kernel here runs on
    std::unique_ptr<strata::GgufFile> pack;

    struct Mat {
        const uint8_t* host = nullptr;
        size_t bytes = 0;
        int type = 0;
        int64_t ne0 = 0, ne1 = 0;   // GGUF order: ne0 = the contiguous (input) dim
        void* dev = nullptr;
    };
    std::map<std::string, Mat> mats;
    std::map<std::string, float*> norms;   ///< fp32 device gammas, `1 + w` applied
    std::map<std::string, const uint16_t*> hostf16;

    std::vector<void*> owned;
    void* grab(size_t bytes) {
        void* p = cu::alloc(bytes);
        if (!p) throw std::runtime_error("dspark: no device memory for " + std::to_string(bytes >> 10) + " KiB");
        owned.push_back(p);
        return p;
    }

    int64_t klen = 0, qlen = 0, tapw = 0, rows = 0, win = 0, tap_n = 0;

    float *tap_in = nullptr, *fcy = nullptr, *kc = nullptr, *vc = nullptr;
    float *x = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *o = nullptr;
    float *gate = nullptr, *up = nullptr, *hid = nullptr, *dn = nullptr, *fn = nullptr, *logits = nullptr;
    uint16_t *q16 = nullptr, *k16 = nullptr, *v16 = nullptr;
    void* xq = nullptr;
    float *cos_tab = nullptr, *sin_tab = nullptr;
    int64_t rope_cap = 0;
    int32_t* d_pos = nullptr;
    std::vector<int32_t> h_pos;

    std::vector<uint16_t*> ring_k, ring_v;

    const void* head_dev = nullptr;
    int head_type = -1;
    int64_t head_rows = 0, head_cols = 0;
    int64_t vram_extra = 0;

    std::vector<float> h_logits, h_fn;
    std::vector<int32_t> cand;   // the Markov head's candidate list for one block row

    const Mat& mat(const std::string& n) const {
        const auto it = mats.find(n);
        if (it == mats.end()) throw std::runtime_error("dspark: the pack has no " + n);
        return it->second;
    }
    float* norm(const std::string& n) const {
        const auto it = norms.find(n);
        if (it == norms.end()) throw std::runtime_error("dspark: the pack has no " + n);
        return it->second;
    }
    const uint16_t* h16(const std::string& n) const {
        const auto it = hostf16.find(n);
        if (it == hostf16.end()) throw std::runtime_error("dspark: the pack has no " + n);
        return it->second;
    }

    void quant(const float* src, int cols, int n_in) {
        strata::kernels::quantize_q8_1_rows(src, cols, n_in, xq, cs);
    }
    void mm(const char* name, int cols, int n_in, int n_out, float* out) {
        const Mat& m = mat(name);
        strata::kernels::native_mmvq(m.type, m.dev, xq, out, n_in, n_out, cols, cs);
    }
    void lin(const char* name, const float* src, int cols, int n_in, int n_out, float* out) {
        quant(src, cols, n_in);
        mm(name, cols, n_in, n_out, out);
    }
    void lin2(const char* a, float* oa, int na, const char* b, float* ob, int nb, const float* src, int cols,
              int n_in) {
        quant(src, cols, n_in);
        mm(a, cols, n_in, na, oa);
        mm(b, cols, n_in, nb, ob);
    }
    void lin3(const char* a, float* oa, int na, const char* b, float* ob, int nb, const char* c, float* oc, int nc,
              const float* src, int cols, int n_in) {
        quant(src, cols, n_in);
        mm(a, cols, n_in, na, oa);
        mm(b, cols, n_in, nb, ob);
        mm(c, cols, n_in, nc, oc);
    }

    void ensure_rope(int64_t need) {
        if (need <= rope_cap) return;
        int64_t cap = rope_cap ? rope_cap : 4096;
        while (cap < need) cap *= 2;
        std::vector<float> c((size_t)cap * (size_t)(g.n_rot / 2)), s(c.size());
        strata::kernels::build_rope_table((int)g.n_rot, g.theta, (int)cap, c.data(), s.data());
        if (cos_tab) cu::dealloc(cos_tab);
        if (sin_tab) cu::dealloc(sin_tab);
        cos_tab = (float*)cu::alloc(c.size() * 4);
        sin_tab = (float*)cu::alloc(s.size() * 4);
        cu::copy_h2d(cos_tab, c.data(), c.size() * 4);
        cu::copy_h2d(sin_tab, s.data(), s.size() * 4);
        rope_cap = cap;
    }
    void set_pos(int64_t p0, int64_t n) {
        h_pos.resize((size_t)n);
        for (int64_t i = 0; i < n; ++i) h_pos[(size_t)i] = (int32_t)(p0 + i);
        cu::copy_h2d(d_pos, h_pos.data(), (size_t)n * 4);
    }
    /// Full-row RMSNorm (the device gamma carries Gemma's `1 + w`) of `n` rows of `n_embd` at positions p0.. .
    void row_norm(const float* src, const std::string& gamma, float* dst, int n, int64_t p0) {
        set_pos(p0, n);
        cu::head_norm_rope(src, norm(gamma), dst, nullptr, n, 1, (int)g.n_embd, 0, cos_tab, sin_tab, d_pos,
                           (float)g.eps, cs);
    }
    /// Per-head RMSNorm + partial RoPE over `rows_n` head rows (`heads_per_tok` heads per token).
    void head_norm(const float* src, const std::string& gamma, uint16_t* o16, int rows_n, int heads_per_tok,
                   int64_t p0, int n_tok) {
        set_pos(p0, n_tok);
        cu::head_norm_rope(src, norm(gamma), nullptr, o16, rows_n, heads_per_tok, (int)g.head_dim, (int)g.n_rot,
                           cos_tab, sin_tab, d_pos, (float)g.eps, cs);
    }
};

DsparkDrafter::DsparkDrafter() = default;

DsparkDrafter::~DsparkDrafter() {
    if (!impl_) return;
    if (impl_->dev >= 0) {
        cu::set_device(impl_->dev);
        if (impl_->cos_tab) cu::dealloc(impl_->cos_tab);
        if (impl_->sin_tab) cu::dealloc(impl_->sin_tab);
        for (void* p : impl_->owned) cu::dealloc(p);
        if (impl_->cs) cu::stream_destroy(impl_->cs);
    }
    delete impl_;
}

bool DsparkDrafter::load(const std::string& pack_path, int device, int64_t window, std::string& err) {
    const int prev_device = cu::current_device();
    try {
        auto* im = new Impl();
        impl_ = im;
        im->dev = device;
        cu::set_device(device);
        im->cs = cu::stream_create_blocking();
        im->pack = std::make_unique<strata::GgufFile>(pack_path);
        const strata::GgufFile& f = *im->pack;
        const auto num = [&](const char* key, double dflt) -> double {
            const strata::MetaValue* v = f.get(key);
            return v ? v->num() : dflt;
        };
        DsparkGeometry g;
        g.n_layers = (int64_t)num("dspark.block_count", 5);
        g.n_embd = (int64_t)num("dspark.embedding_length", 6144);
        g.n_head = (int64_t)num("dspark.attention.head_count", 64);
        g.n_head_kv = (int64_t)num("dspark.attention.head_count_kv", 4);
        g.head_dim = (int64_t)num("dspark.attention.key_length", 128);
        g.n_rot = (int64_t)num("dspark.rope.dimension_count", 64);
        g.n_ff = (int64_t)num("dspark.feed_forward_length", 12288);
        g.block = (int64_t)num("dspark.block_size", 7);
        g.rank = (int64_t)num("dspark.markov_rank", 256);
        g.mask_id = (int64_t)num("dspark.mask_token_id", 200058);
        g.vocab = (int64_t)num("dspark.vocab_size", 200064);
        g.theta = num("dspark.rope.freq_base", 1e7);
        g.eps = num("dspark.attention.layer_norm_rms_epsilon", 1e-6);
        g.alpha = num("dspark.swiglu_alpha", 1.702);
        g.limit = num("dspark.swiglu_limit", 7.0);
        g.norm_add1 = num("dspark.norm_add1", 0) != 0;
        const int64_t packed_window = (int64_t)num("dspark.attention.sliding_window", 4096);
        if (const strata::MetaValue* taps = f.get("dspark.tap_layers"))
            if (taps->type == strata::MetaType::ARRAY)
                for (const auto& it : taps->items) g.taps.push_back((int64_t)it.u);
        if (g.taps.empty()) {
            err = "dspark: the pack has no dspark.tap_layers";
            return false;
        }
        g.window = window > 0 ? window : packed_window;
        if (g.window < 64) g.window = 64;
        if (g.window > packed_window) g.window = packed_window;

        for (const strata::TensorInfo& t : f.tensors()) {
            Impl::Mat m;
            m.host = f.tensor_data(t);
            m.bytes = (size_t)strata::tensor_payload_bytes(t);
            m.type = (int)t.type;
            m.ne0 = t.shape.empty() ? 0 : (int64_t)t.shape[0];
            m.ne1 = t.shape.size() > 1 ? (int64_t)t.shape[1] : 1;
            if (m.host == nullptr || m.bytes == 0) {
                err = "dspark: " + t.name + ": unusable payload";
                return false;
            }
            im->mats[t.name] = m;
        }
        const Impl::Mat& fc = im->mat("fc.weight");
        g.n_embd = fc.ne1;
        im->tapw = fc.ne0;
        im->tap_n = (int64_t)g.taps.size();
        if (im->tapw != im->tap_n * g.n_embd) {
            err = "dspark: fc.weight takes " + std::to_string(fc.ne0) + " values but " + std::to_string(im->tap_n) +
                  " taps x " + std::to_string(g.n_embd) + " do not add up";
            return false;
        }
        im->klen = g.n_head_kv * g.head_dim;
        im->qlen = g.n_head * g.head_dim;
        im->rows = g.rows();
        im->win = g.window;
        if (im->rows > 8 || im->rows < 2) {
            err = "dspark: the block needs 2..8 rows (block_size + 1), got " + std::to_string(im->rows);
            return false;
        }
        if (im->tap_n > 8 || g.rank > 512) {
            err = "dspark: at most 8 taps and a Markov rank of at most 512";
            return false;
        }
        geo_ = g;
        im->g = g;

        for (auto& kv : im->mats)
            if (kv.second.ne1 > 1) cu::copy_h2d(kv.second.dev = im->grab(kv.second.bytes), kv.second.host,
                                                kv.second.bytes);
        std::vector<std::string> nn = {"hidden_norm.weight", "final_norm.weight"};
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            nn.push_back(p + "input_layernorm.weight");
            nn.push_back(p + "post_attention_layernorm.weight");
            nn.push_back(p + "self_attn.q_norm.weight");
            nn.push_back(p + "self_attn.k_norm.weight");
        }
        for (const std::string& s : nn) {
            const Impl::Mat& m = im->mat(s);
            if (m.type != GGML_TYPE_F16) {
                err = "dspark: " + s + " must be F16";
                return false;
            }
            const uint16_t* src = (const uint16_t*)m.host;
            std::vector<float> val((size_t)m.ne0);
            for (int64_t i = 0; i < m.ne0; ++i)
                val[(size_t)i] = ggml_fp16_to_fp32(src[i]) + (g.norm_add1 ? 1.f : 0.f);
            float* d = (float*)im->grab((size_t)m.ne0 * 4);
            cu::copy_h2d(d, val.data(), val.size() * 4);
            im->norms[s] = d;
        }
        for (const char* s : {"markov_w1.weight", "markov_w2.weight", "confidence_head.weight",
                              "confidence_head.bias"}) {
            const Impl::Mat& m = im->mat(s);
            if (m.type != GGML_TYPE_F16) {
                err = std::string("dspark: ") + s + " must be F16";
                return false;
            }
            im->hostf16[s] = (const uint16_t*)m.host;
        }

        const int64_t R = im->rows, E = g.n_embd, Q = im->qlen, K = im->klen, I = g.n_ff;
        im->xq = im->grab((size_t)R * (size_t)((im->tapw + 31) / 32) * 36 + 4096);
        im->tap_in = (float*)im->grab((size_t)R * (size_t)im->tapw * 4);
        im->fcy = (float*)im->grab((size_t)R * (size_t)E * 4);
        im->kc = (float*)im->grab((size_t)R * (size_t)K * 4);
        im->vc = (float*)im->grab((size_t)R * (size_t)K * 4);
        im->x = (float*)im->grab((size_t)R * (size_t)E * 4);
        im->xn = (float*)im->grab((size_t)R * (size_t)E * 4);
        im->q = (float*)im->grab((size_t)R * (size_t)Q * 4);
        im->k = (float*)im->grab((size_t)R * (size_t)K * 4);
        im->v = (float*)im->grab((size_t)R * (size_t)K * 4);
        im->ao = (float*)im->grab((size_t)R * (size_t)Q * 4);
        im->o = (float*)im->grab((size_t)R * (size_t)E * 4);
        im->gate = (float*)im->grab((size_t)R * (size_t)I * 4);
        im->up = (float*)im->grab((size_t)R * (size_t)I * 4);
        im->hid = (float*)im->grab((size_t)R * (size_t)I * 4);
        im->dn = (float*)im->grab((size_t)R * (size_t)E * 4);
        im->fn = (float*)im->grab((size_t)R * (size_t)E * 4);
        im->logits = (float*)im->grab((size_t)R * (size_t)g.vocab * 4);
        im->q16 = (uint16_t*)im->grab((size_t)R * (size_t)Q * 2);
        im->k16 = (uint16_t*)im->grab((size_t)R * (size_t)K * 2);
        im->v16 = (uint16_t*)im->grab((size_t)R * (size_t)K * 2);
        im->d_pos = (int32_t*)im->grab((size_t)std::max<int64_t>(R, im->tap_n) * sizeof(int32_t));
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const size_t rb = (size_t)im->win * (size_t)K * 2;
            im->ring_k.push_back((uint16_t*)im->grab(rb));
            im->ring_v.push_back((uint16_t*)im->grab(rb));
        }
        im->h_logits.assign((size_t)R * (size_t)g.vocab, 0.f);
        im->h_fn.assign((size_t)R * (size_t)E, 0.f);
        im->ensure_rope(4096);
        cu::sync();
        int64_t bytes = 0;
        for (const auto& kv : im->mats)
            if (kv.second.dev) bytes += (int64_t)kv.second.bytes;
        vram_ = bytes;
        cu::set_device(prev_device);
        return true;
    } catch (const std::exception& e) {
        cu::set_device(prev_device);
        err = e.what();
        return false;
    }
}

bool DsparkDrafter::bind_head(const void* dev, const void* host, size_t host_bytes, int ggml_type, int64_t rows,
                              int64_t cols, std::string& err) {
    if (!impl_) {
        err = "dspark: not loaded";
        return false;
    }
    Impl* im = impl_;
    OnDevice on(im->dev);
    if (cols != geo_.n_embd) {
        err = "dspark: the head is " + std::to_string(cols) + " wide, the drafter " + std::to_string(geo_.n_embd);
        return false;
    }
    if (rows < geo_.vocab) {
        err = "dspark: the head has " + std::to_string(rows) + " rows, the pack's vocabulary " +
              std::to_string(geo_.vocab);
        return false;
    }
    if (dev) {
        im->head_dev = dev;
    } else if (host) {
        void* p = cu::alloc(host_bytes);
        if (!p) {
            err = "dspark: the head does not fit on the drafter's device";
            return false;
        }
        im->owned.push_back(p);
        cu::copy_h2d(p, host, host_bytes);
        im->head_dev = p;
        im->vram_extra = (int64_t)host_bytes;
    } else {
        err = "dspark: a head needs either device or host bytes";
        return false;
    }
    im->head_type = ggml_type;
    im->head_rows = rows;
    im->head_cols = cols;
    vram_ += im->vram_extra;
    return true;
}

void DsparkDrafter::reset() {
    if (!impl_) return;
    OnDevice on(impl_->dev);
    seeded_ = 0;
}

bool DsparkDrafter::seed(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (!impl_) {
        err = "dspark: not loaded";
        return false;
    }
    Impl* im = impl_;
    try {
        OnDevice on(im->dev);
        const double t0 = now_ms();
        const DsparkGeometry& g = geo_;
        const int64_t R = im->rows;
        im->ensure_rope(pos0 + n + R + 1);
        for (int64_t r0 = 0; r0 < n; r0 += R) {
            const int m = (int)std::min<int64_t>(R, n - r0);
            cu::copy_h2d(im->tap_in, taps + (size_t)r0 * (size_t)im->tapw, (size_t)m * (size_t)im->tapw * 4);
            im->lin("fc.weight", im->tap_in, m, (int)im->tapw, (int)g.n_embd, im->fcy);
            im->row_norm(im->fcy, "hidden_norm.weight", im->xn, m, pos0 + r0);
            im->quant(im->xn, m, (int)g.n_embd);
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                im->mm((p + "self_attn.k_proj.weight").c_str(), m, (int)g.n_embd, (int)im->klen, im->kc);
                im->mm((p + "self_attn.v_proj.weight").c_str(), m, (int)g.n_embd, (int)im->klen, im->vc);
                im->head_norm(im->kc, p + "self_attn.k_norm.weight", im->k16, m * (int)g.n_head_kv,
                              (int)g.n_head_kv, pos0 + r0, m);
                cu::f32_to_f16(im->vc, im->v16, (int64_t)m * im->klen, im->cs);
                for (int i = 0; i < m; ++i) {
                    const int64_t pos = pos0 + r0 + i;
                    uint16_t* kd = im->ring_k[(size_t)l] + (size_t)(pos % im->win) * (size_t)im->klen;
                    uint16_t* vd = im->ring_v[(size_t)l] + (size_t)(pos % im->win) * (size_t)im->klen;
                    cu::copy_d2d(kd, im->k16 + (size_t)i * (size_t)im->klen, (size_t)im->klen * 2);
                    cu::copy_d2d(vd, im->v16 + (size_t)i * (size_t)im->klen, (size_t)im->klen * 2);
                }
            }
        }
        cu::sync();
        ms_seed += now_ms() - t0;
        seed_rows += n;
        seeded_ = pos0 + n;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool DsparkDrafter::draft(const float* ids, int32_t anchor, int64_t pos0, int32_t* drafts, float* conf,
                          std::string& err) {
    if (!impl_) {
        err = "dspark: not loaded";
        return false;
    }
    Impl* im = impl_;
    try {
        OnDevice on(im->dev);
        const double t0 = now_ms();
        const DsparkGeometry& g = geo_;
        const int R = (int)im->rows;
        const int E = (int)g.n_embd, Q = (int)im->qlen, K = (int)im->klen, I = (int)g.n_ff;
        if (im->head_dev == nullptr) {
            err = "dspark: no head bound";
            return false;
        }
        im->ensure_rope(pos0 + R + 1);
        const int nring = (int)std::min<int64_t>(pos0, im->win);
        cu::copy_h2d(im->x, ids, (size_t)R * (size_t)E * 4);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            im->row_norm(im->x, p + "input_layernorm.weight", im->xn, R, pos0);
            im->lin3((p + "self_attn.q_proj.weight").c_str(), im->q, Q, (p + "self_attn.k_proj.weight").c_str(),
                     im->k, K, (p + "self_attn.v_proj.weight").c_str(), im->v, K, im->xn, R, E);
            im->head_norm(im->q, p + "self_attn.q_norm.weight", im->q16, R * (int)g.n_head, (int)g.n_head, pos0, R);
            im->head_norm(im->k, p + "self_attn.k_norm.weight", im->k16, R * (int)g.n_head_kv, (int)g.n_head_kv,
                          pos0, R);
            cu::f32_to_f16(im->v, im->v16, (int64_t)R * K, im->cs);
            cu::dspark_attn(im->q16, im->ring_k[(size_t)l], im->ring_v[(size_t)l], nring, im->k16, im->v16, R,
                            (int)g.n_head, (int)g.n_head_kv, (int)g.head_dim, im->ao, im->cs);
            im->lin((p + "self_attn.o_proj.weight").c_str(), im->ao, R, Q, E, im->o);
            cu::dspark_add(im->x, im->o, (int64_t)R * E, im->cs);
            im->row_norm(im->x, p + "post_attention_layernorm.weight", im->xn, R, pos0);
            im->lin2((p + "mlp.gate_proj.weight").c_str(), im->gate, I, (p + "mlp.up_proj.weight").c_str(), im->up,
                     I, im->xn, R, E);
            cu::swiglu_rows_f32(im->gate, im->up, im->hid, R, I, (float)g.alpha, (float)g.limit, im->cs);
            im->lin((p + "mlp.down_proj.weight").c_str(), im->hid, R, I, E, im->dn);
            cu::dspark_add(im->x, im->dn, (int64_t)R * E, im->cs);
        }
        im->row_norm(im->x, "final_norm.weight", im->fn, R, pos0);
        im->quant(im->fn, R, E);
        strata::kernels::native_mmvq(im->head_type, im->head_dev, im->xq, im->logits, E, (int)g.vocab, R, im->cs);
        cu::sync();
        const double t_gpu = now_ms();
        ms_block += t_gpu - t0;

        cu::copy_d2h(im->h_logits.data(), im->logits, (size_t)R * (size_t)g.vocab * 4);
        cu::copy_d2h(im->h_fn.data(), im->fn, (size_t)R * E * 4);

        // ---- the Markov chain (host): pick[j] = argmax_v (logit_j[v] + <w1[v], w2[prev]>), prev = the pick before
        const int64_t V = g.vocab, rank = g.rank;
        const uint16_t* w1 = im->h16("markov_w1.weight");
        const uint16_t* w2 = im->h16("markov_w2.weight");
        const uint16_t* cw = im->h16("confidence_head.weight");
        const uint16_t* cb = im->h16("confidence_head.bias");
        float code[512];
        int32_t prev = anchor;
        const int kd = (int)im->rows;      // one draft per block row, row 0 = the anchor
        for (int j = 0; j < kd; ++j) {
            const float* row = im->h_logits.data() + (size_t)j * (size_t)V;
            for (int64_t r = 0; r < rank; ++r)
                code[r] = ggml_fp16_to_fp32(w1[(size_t)prev * (size_t)rank + (size_t)r]);
            // The Markov bias only reorders tokens that are already close, so it is evaluated over the row's top
            // candidates: one scan for the cutoff, then the bias on the survivors (a full-vocab walk would read
            // 100 MiB of markov_w2 per position).  W1 embeds the previous token into the rank space, W2 projects it
            // back to the vocabulary: bias[v] = <markov_w2[v], markov_w1[prev]>.
            float top = -INFINITY;
            for (int64_t v = 0; v < V; ++v)
                if (row[v] > top) top = row[v];
            const float floor_v = top - kCandidateNats;
            im->cand.clear();
            for (int64_t v = 0; v < V; ++v)
                if (row[v] >= floor_v && (int)im->cand.size() < kCandidateCap) im->cand.push_back((int32_t)v);
            float best = -INFINITY;
            int32_t pick = 0;
            for (int32_t v : im->cand) {
                const uint16_t* wr = w2 + (size_t)v * (size_t)rank;
                float s = row[v];
                for (int64_t r = 0; r < rank; ++r) s += ggml_fp16_to_fp32(wr[r]) * code[r];
                if (s > best) {
                    best = s;
                    pick = v;
                }
            }
            drafts[j] = pick;
            prev = pick;
            const float* fr = im->h_fn.data() + (size_t)j * (size_t)E;
            double acc = ggml_fp16_to_fp32(cb[0]);
            for (int64_t c = 0; c < E; ++c) acc += (double)ggml_fp16_to_fp32(cw[c]) * (double)fr[c];
            for (int64_t c = 0; c < rank; ++c)
                acc += (double)ggml_fp16_to_fp32(cw[E + c]) * (double)code[c];
            conf[j] = sigmoidf_(acc);
        }
        ms_head += now_ms() - t_gpu;
        ++rounds;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::minimax
