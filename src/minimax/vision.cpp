// src/minimax/vision.cpp - MiniMax-M3.1-Flash ViT + projector, read from a llama.cpp `clip` mmproj.
//
// The layout and the reference arithmetic are llama.cpp's (tools/mtmd, MIT): conversion/minimax.py writes
// v.patch_embd.weight[.1], v.blk.N.*, mm.1/mm.2 and mm.merger.fc1/fc2.  The model math is transformers'
// MiniMaxM3VLVisionModel (3D RoPE with t=0, exact-erf GELU, pre-LN) followed by its MultiModalProjector.
//
// Everything here is fp32 on the CPU: the mmproj is 1.7 GB on disk, 3.5 GB expanded, and the tower is run
// once per image (a few seconds) rather than per token.  The engine never needs a CUDA context for it.
#include "strata/minimax/vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "third_party/stb_image.h"

namespace strata::minimax {
namespace {

constexpr double kPi = 3.14159265358979323846;

int workers(int asked) {
    if (asked > 0) return asked;
    unsigned n = std::thread::hardware_concurrency();
    return (int)std::max(1u, std::min(n, 16u));
}

/// Rows [0,n) split over `threads`; fn(begin, end).  A small grain keeps the tail from waiting on one row.
template <class Fn>
void parallel_rows(int64_t n, int threads, Fn&& fn) {
    if (n <= 0) return;
    if (threads <= 1 || n < 8) {
        fn((int64_t)0, n);
        return;
    }
    const int nt = (int)std::min<int64_t>(threads, n);
    std::vector<std::thread> pool;
    pool.reserve((size_t)nt - 1);
    const int64_t per = (n + nt - 1) / nt;
    for (int t = 1; t < nt; ++t) {
        const int64_t b = (int64_t)t * per;
        const int64_t e = std::min<int64_t>(n, b + per);
        if (b >= e) break;
        pool.emplace_back([&fn, b, e] { fn(b, e); });
    }
    fn((int64_t)0, std::min<int64_t>(n, per));
    for (auto& th : pool) th.join();
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f)); }

void layer_norm(const float* x, const float* w, const float* b, float* y, int64_t n, double eps) {
    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) mean += x[i];
    mean /= (double)n;
    double var = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double)x[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    const float inv = (float)(1.0 / std::sqrt(var + eps));
    const float m = (float)mean;
    for (int64_t i = 0; i < n; ++i) y[i] = ((x[i] - m) * inv) * w[i] + b[i];
}

/// y[row, o] = sum_i x[row, i] * w[o, i] + b[o]; w is GGUF row-major [out][in].
void linear(const float* x, int64_t rows, const float* w, const float* b, float* y, int64_t in, int64_t out,
            int threads) {
    parallel_rows(rows, threads, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const float* xr = x + r * in;
            float* yr = y + r * out;
            for (int64_t o = 0; o < out; ++o) {
                const float* wr = w + o * in;
                float acc = 0.0f;
                int64_t i = 0;
                for (; i + 4 <= in; i += 4)
                    acc += xr[i] * wr[i] + xr[i + 1] * wr[i + 1] + xr[i + 2] * wr[i + 2] + xr[i + 3] * wr[i + 3];
                for (; i < in; ++i) acc += xr[i] * wr[i];
                yr[o] = acc + (b ? b[o] : 0.0f);
            }
        }
    });
}

/// The bicubic kernel PIL/torchvision use for BICUBIC (a = -0.5).
inline double cubic(double x) {
    const double a = -0.5;
    const double ax = std::fabs(x);
    if (ax <= 1.0) return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;
    if (ax < 2.0) return (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a;
    return 0.0;
}

struct Tap {
    int64_t i0 = 0;
    std::vector<int64_t> idx;
    std::vector<double> w;
};

/// torchvision's antialiased resize weights: the kernel widens by `scale` when downsampling and is normalized.
std::vector<Tap> resize_taps(int64_t in, int64_t out) {
    std::vector<Tap> taps((size_t)out);
    const double scale = (double)in / (double)out;
    const double filterscale = std::max(1.0, scale);
    const double support = 2.0 * filterscale;
    for (int64_t o = 0; o < out; ++o) {
        const double center = (double)o * scale + 0.5 * (scale - 1.0);
        const int64_t left = std::max<int64_t>(0, (int64_t)std::floor(center - support + 0.5));
        const int64_t right = std::min<int64_t>(in - 1, (int64_t)std::floor(center + support + 0.5));
        Tap& t = taps[(size_t)o];
        double sum = 0.0;
        for (int64_t i = left; i <= right; ++i) {
            const double w = cubic(((double)i - center) / filterscale);
            if (w == 0.0) continue;
            t.idx.push_back(i);
            t.w.push_back(w);
            sum += w;
        }
        if (sum != 0.0)
            for (double& w : t.w) w /= sum;
    }
    return taps;
}

/// Resize uint8 RGB [h,w,3] to [oh,ow,3], bicubic, antialiased, clamped and rounded back to uint8 like
/// torchvision's functional resize on a uint8 tensor.
void resize_rgb(const uint8_t* src, int64_t h, int64_t w, int64_t oh, int64_t ow, std::vector<uint8_t>& dst) {
    if (h == oh && w == ow) {
        dst.assign(src, src + (size_t)h * w * 3);
        return;
    }
    const auto tx = resize_taps(w, ow);
    const auto ty = resize_taps(h, oh);
    std::vector<float> tmp((size_t)h * ow * 3);
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t ox = 0; ox < ow; ++ox) {
            const Tap& t = tx[(size_t)ox];
            double r = 0, g = 0, b = 0;
            for (size_t k = 0; k < t.idx.size(); ++k) {
                const uint8_t* p = src + ((size_t)y * w + (size_t)t.idx[k]) * 3;
                r += t.w[k] * p[0];
                g += t.w[k] * p[1];
                b += t.w[k] * p[2];
            }
            float* q = tmp.data() + ((size_t)y * ow + ox) * 3;
            q[0] = (float)r;
            q[1] = (float)g;
            q[2] = (float)b;
        }
    }
    dst.resize((size_t)oh * ow * 3);
    for (int64_t oy = 0; oy < oh; ++oy) {
        const Tap& t = ty[(size_t)oy];
        for (int64_t x = 0; x < ow; ++x) {
            double r = 0, g = 0, b = 0;
            for (size_t k = 0; k < t.idx.size(); ++k) {
                const float* p = tmp.data() + ((size_t)t.idx[k] * ow + x) * 3;
                r += t.w[k] * p[0];
                g += t.w[k] * p[1];
                b += t.w[k] * p[2];
            }
            uint8_t* q = dst.data() + ((size_t)oy * ow + x) * 3;
            const double v[3] = {r, g, b};
            for (int c = 0; c < 3; ++c)
                q[c] = (uint8_t)std::clamp<long long>((long long)std::llround(std::clamp(v[c], 0.0, 255.0)), 0, 255);
        }
    }
}

struct Mat {
    int64_t in = 0, out = 0;
    std::vector<float> w, b;
};

struct Layer {
    std::vector<float> ln1, ln1b, ln2, ln2b;
    Mat q, k, v, o, up, down;
};

double meta_num(const GgufFile& f, const std::string& key, double dflt) {
    const MetaValue* v = f.get(key);
    return v ? v->num() : dflt;
}

}  // namespace

struct Vision::Impl {
    VisionConfig cfg;
    std::string path;
    int threads = 1;
    Mat patch0, patch1;   // [1280][3*14*14] effective Conv2d sum
    std::vector<float> pre_w, pre_b;
    std::vector<Layer> layers;
    Mat mm1, mm2, merge1, merge2;
    // cos/sin for the 13 RoPE pairs of each axis, per patch
    mutable std::vector<float> cos_h, sin_h, cos_w, sin_w;
    mutable int64_t positions = 0;

    const float* tensor(const GgufFile& f, const std::string& name, std::vector<float>& store,
                        std::vector<uint64_t>* shape = nullptr) {
        const TensorInfo* t = f.find(name);
        if (!t) throw std::runtime_error("mmproj is missing tensor " + name);
        if (t->type != 0 && t->type != 1)
            throw std::runtime_error("mmproj tensor " + name + " is " + t->type_name() + "; only F16/F32 are read");
        if (shape) *shape = t->shape;
        const uint8_t* src = f.tensor_data(*t);
        const uint64_t n = t->elements();
        store.resize((size_t)n);
        if (t->type == 0) {
            std::memcpy(store.data(), src, (size_t)n * sizeof(float));
        } else {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
            for (uint64_t i = 0; i < n; ++i) store[(size_t)i] = ggml_fp16_to_fp32(s[i]);
        }
        return store.data();
    }

    void load_mat(const GgufFile& f, const std::string& base, Mat& m) {
        std::vector<uint64_t> sh;
        tensor(f, base + ".weight", m.w, &sh);
        if (sh.size() != 2) throw std::runtime_error("mmproj " + base + ".weight is not a matrix");
        m.in = (int64_t)sh[0];
        m.out = (int64_t)sh[1];
        if ((uint64_t)(m.in * m.out) != m.w.size())
            throw std::runtime_error("mmproj " + base + ".weight has a bad shape");
        if (const TensorInfo* bt = f.find(base + ".bias")) {
            tensor(f, base + ".bias", m.b);
            if ((int64_t)m.b.size() != m.out) throw std::runtime_error("mmproj " + base + ".bias has a bad shape");
        }
    }

    void load(const std::string& p, int asked_threads) {
        path = p;
        threads = workers(asked_threads);
        GgufFile f(p);
        const MetaValue* arch = f.get("general.architecture");
        if (!arch || arch->s != "clip") throw std::runtime_error("mmproj: general.architecture is not clip");
        const MetaValue* pt = f.get("clip.projector_type");
        if (!pt || pt->s != "minimax_m3")
            throw std::runtime_error("mmproj: clip.projector_type is not minimax_m3 (this is not a MiniMax-M3 vision file)");
        cfg.hidden = (int64_t)meta_num(f, "clip.vision.embedding_length", 1280);
        cfg.ff = (int64_t)meta_num(f, "clip.vision.feed_forward_length", 5120);
        cfg.layers = (int64_t)meta_num(f, "clip.vision.block_count", 32);
        cfg.heads = (int64_t)meta_num(f, "clip.vision.attention.head_count", 16);
        cfg.patch = (int64_t)meta_num(f, "clip.vision.patch_size", 14);
        cfg.merge = (int64_t)meta_num(f, "clip.vision.spatial_merge_size", 2);
        cfg.image_size = (int64_t)meta_num(f, "clip.vision.image_size", 672);
        cfg.eps = meta_num(f, "clip.vision.attention.layer_norm_epsilon", 1e-5);
        cfg.text_hidden = (int64_t)meta_num(f, "clip.vision.projection_dim", 6144);
        cfg.projector_hidden = cfg.text_hidden;
        if (const MetaValue* m = f.get("clip.vision.image_mean"))
            if (m->type == MetaType::ARRAY && m->count == 3 && m->items.size() == 3)
                for (int i = 0; i < 3; ++i) cfg.mean[(size_t)i] = (float)m->items[(size_t)i].num();
        if (const MetaValue* s = f.get("clip.vision.image_std"))
            if (s->type == MetaType::ARRAY && s->count == 3 && s->items.size() == 3)
                for (int i = 0; i < 3; ++i) cfg.std[(size_t)i] = (float)s->items[(size_t)i].num();
        if (cfg.hidden <= 0 || cfg.layers <= 0 || cfg.heads <= 0 || cfg.patch <= 0 ||
            cfg.hidden % cfg.heads != 0 || cfg.merge <= 0)
            throw std::runtime_error("mmproj: the vision metadata is not a MiniMax-M3 tower");
        if (cfg.temporal != 2) throw std::runtime_error("mmproj: temporal_patch_size is not 2");

        // patch embedding: the converter splits Conv3d into two 2D slices; an image is one frame duplicated,
        // so conv(slice0) + conv(slice1) is conv over the two identical temporal frames.
        std::vector<uint64_t> sh0, sh1;
        std::vector<float> w0, w1;
        tensor(f, "v.patch_embd.weight", w0, &sh0);
        tensor(f, "v.patch_embd.weight.1", w1, &sh1);
        if (sh0 != sh1 || sh0.size() != 4 || sh0[0] != (uint64_t)cfg.patch || sh0[1] != (uint64_t)cfg.patch ||
            sh0[2] != 3 || sh0[3] != (uint64_t)cfg.hidden)
            throw std::runtime_error("mmproj: v.patch_embd.weight has a bad shape");
        const int64_t cin = 3, kk = cfg.patch * cfg.patch;
        patch0.in = cin * kk;
        patch0.out = cfg.hidden;
        patch0.w.resize((size_t)patch0.in * (size_t)patch0.out);
        for (int64_t o = 0; o < cfg.hidden; ++o)
            for (int64_t c = 0; c < cin; ++c)
                for (int64_t k = 0; k < kk; ++k) {
                    const size_t src = (size_t)(((o * cin + c) * kk) + k);
                    patch0.w[(size_t)(o * patch0.in + c * kk + k)] = w0[src] + w1[src];
                }

        tensor(f, "v.pre_ln.weight", pre_w);
        tensor(f, "v.pre_ln.bias", pre_b);
        if ((int64_t)pre_w.size() != cfg.hidden || (int64_t)pre_b.size() != cfg.hidden)
            throw std::runtime_error("mmproj: v.pre_ln has a bad shape");

        layers.resize((size_t)cfg.layers);
        for (int64_t i = 0; i < cfg.layers; ++i) {
            const std::string b = "v.blk." + std::to_string(i) + ".";
            Layer& L = layers[(size_t)i];
            tensor(f, b + "ln1.weight", L.ln1);
            tensor(f, b + "ln1.bias", L.ln1b);
            tensor(f, b + "ln2.weight", L.ln2);
            tensor(f, b + "ln2.bias", L.ln2b);
            load_mat(f, b + "attn_q", L.q);
            load_mat(f, b + "attn_k", L.k);
            load_mat(f, b + "attn_v", L.v);
            load_mat(f, b + "attn_out", L.o);
            load_mat(f, b + "ffn_up", L.up);
            load_mat(f, b + "ffn_down", L.down);
        }
        load_mat(f, "mm.1", mm1);
        load_mat(f, "mm.2", mm2);
        load_mat(f, "mm.merger.fc1", merge1);
        load_mat(f, "mm.merger.fc2", merge2);
        if (mm1.in != cfg.hidden || mm1.out != cfg.projector_hidden || mm2.in != cfg.projector_hidden ||
            mm2.out != cfg.text_hidden || merge1.in != cfg.text_hidden * cfg.merge * cfg.merge ||
            merge1.out != cfg.projector_hidden || merge2.in != cfg.projector_hidden || merge2.out != cfg.text_hidden)
            throw std::runtime_error("mmproj: the projector shapes do not match a MiniMax-M3 vision file");
    }

    void set_positions(const std::vector<int32_t>& ph, const std::vector<int32_t>& pw) const {
        const int64_t n = (int64_t)ph.size();
        const int64_t d_head = cfg.hidden / cfg.heads;
        const int64_t pairs = (d_head / 2) / 3;   // 13 for d_head 80: axis_dim 26, 13 pairs
        cos_h.assign((size_t)n * pairs, 0.0f);
        sin_h.assign((size_t)n * pairs, 0.0f);
        cos_w.assign((size_t)n * pairs, 0.0f);
        sin_w.assign((size_t)n * pairs, 0.0f);
        for (int64_t p = 0; p < n; ++p) {
            for (int64_t i = 0; i < pairs; ++i) {
                const double inv = 1.0 / std::pow(cfg.rope_theta, (double)(2 * i) / (double)(2 * pairs));
                const double ah = (double)ph[(size_t)p] * inv;
                const double aw = (double)pw[(size_t)p] * inv;
                cos_h[(size_t)(p * pairs + i)] = (float)std::cos(ah);
                sin_h[(size_t)(p * pairs + i)] = (float)std::sin(ah);
                cos_w[(size_t)(p * pairs + i)] = (float)std::cos(aw);
                sin_w[(size_t)(p * pairs + i)] = (float)std::sin(aw);
            }
        }
        positions = n;
    }

    void rope(float* q, float* k, int64_t n) const {
        const int64_t d_head = cfg.hidden / cfg.heads;
        const int64_t pairs = (d_head / 2) / 3;   // 13 for d_head 80
        // GGUF q/k layout is [T Tdup H Hdup W Wdup pad].  With t = 0 the T pair is an identity; H is
        // (26+i, 39+i) and W is (52+i, 65+i), i.e. two NEOX blocks of `pairs` pairs.
        auto rot = [&](float* v, int64_t off, const float* cs, const float* sn) {
            for (int64_t i = 0; i < pairs; ++i) {
                const float c = cs[(size_t)(n * pairs + i)];
                const float s = sn[(size_t)(n * pairs + i)];
                float& a = v[off + i];
                float& b = v[off + pairs + i];
                const float a0 = a, b0 = b;
                a = a0 * c - b0 * s;
                b = b0 * c + a0 * s;
            }
        };
        for (int64_t h = 0; h < cfg.heads; ++h) {
            rot(q + h * d_head, 26, cos_h.data(), sin_h.data());
            rot(q + h * d_head, 52, cos_w.data(), sin_w.data());
            rot(k + h * d_head, 26, cos_h.data(), sin_h.data());
            rot(k + h * d_head, 52, cos_w.data(), sin_w.data());
        }
    }

    ImageEmbeddings run(const float* pixels, int64_t n_patch, int64_t width, int64_t height) const {
        const int64_t ne = cfg.hidden, nh = cfg.heads, dh = ne / nh;
        const int64_t gh = height / cfg.patch, gw = width / cfg.patch;
        if (gh * gw != n_patch || gh % cfg.merge || gw % cfg.merge)
            throw std::runtime_error("vision: the pixel rows do not match the resized image grid");
        const int64_t n = n_patch;

        std::vector<int32_t> ph((size_t)n), pw((size_t)n);
        int64_t p = 0;
        for (int64_t bh = 0; bh < gh / cfg.merge; ++bh)
            for (int64_t bw = 0; bw < gw / cfg.merge; ++bw)
                for (int64_t mh = 0; mh < cfg.merge; ++mh)
                    for (int64_t mw = 0; mw < cfg.merge; ++mw) {
                        ph[(size_t)p] = (int32_t)(bh * cfg.merge + mh);
                        pw[(size_t)p] = (int32_t)(bw * cfg.merge + mw);
                        ++p;
                    }
        set_positions(ph, pw);

        std::vector<float> x((size_t)n * ne), y((size_t)n * ne), z((size_t)n * ne);
        std::vector<float> q((size_t)n * ne), k((size_t)n * ne), v((size_t)n * ne), o((size_t)n * ne);
        std::vector<float> ff((size_t)n * cfg.ff);
        std::vector<float> att((size_t)n), out_row((size_t)ne);

        // patch embedding (conv over the summed temporal slices)
        parallel_rows(n, threads, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const float* xr = pixels + r * patch0.in;
                float* yr = x.data() + r * ne;
                for (int64_t oc = 0; oc < ne; ++oc) {
                    const float* wr = patch0.w.data() + oc * patch0.in;
                    float acc = 0.0f;
                    for (int64_t i = 0; i < patch0.in; ++i) acc += xr[i] * wr[i];
                    yr[oc] = acc;
                }
            }
        });
        parallel_rows(n, threads, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r)
                layer_norm(x.data() + r * ne, pre_w.data(), pre_b.data(), z.data() + r * ne, ne, cfg.eps);
        });
        x.swap(z);

        for (int64_t li = 0; li < cfg.layers; ++li) {
            const Layer& L = layers[(size_t)li];
            parallel_rows(n, threads, [&](int64_t r0, int64_t r1) {
                for (int64_t r = r0; r < r1; ++r)
                    layer_norm(x.data() + r * ne, L.ln1.data(), L.ln1b.data(), z.data() + r * ne, ne, cfg.eps);
            });
            linear(z.data(), n, L.q.w.data(), L.q.b.data(), q.data(), ne, ne, threads);
            linear(z.data(), n, L.k.w.data(), L.k.b.data(), k.data(), ne, ne, threads);
            linear(z.data(), n, L.v.w.data(), L.v.b.data(), v.data(), ne, ne, threads);
            for (int64_t r = 0; r < n; ++r) rope(q.data() + r * ne, k.data() + r * ne, r);

            // non-causal multi-head attention, one head per worker
            const float scale = 1.0f / std::sqrt((float)dh);
            std::vector<std::thread> pool;
            const int nt = (int)std::min<int64_t>(threads, nh);
            auto head_fn = [&](int64_t h) {
                const int64_t base = h * dh;
                std::vector<float> sc((size_t)n);
                for (int64_t i = 0; i < n; ++i) {
                    const float* qi = q.data() + i * ne + base;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int64_t j = 0; j < n; ++j) {
                        const float* kj = k.data() + j * ne + base;
                        float s = 0.0f;
                        for (int64_t d = 0; d < dh; ++d) s += qi[d] * kj[d];
                        s *= scale;
                        sc[(size_t)j] = s;
                        mx = std::max(mx, s);
                    }
                    float sum = 0.0f;
                    for (int64_t j = 0; j < n; ++j) {
                        sc[(size_t)j] = std::exp(sc[(size_t)j] - mx);
                        sum += sc[(size_t)j];
                    }
                    const float inv = 1.0f / sum;
                    for (int64_t d = 0; d < dh; ++d) {
                        float acc = 0.0f;
                        for (int64_t j = 0; j < n; ++j)
                            acc += sc[(size_t)j] * inv * v[(size_t)(j * ne + base + d)];
                        o[(size_t)(i * ne + base + d)] = acc;
                    }
                }
            };
            for (int t = 1; t < nt; ++t) pool.emplace_back(head_fn, t);
            head_fn(0);
            for (auto& th : pool) th.join();

            linear(o.data(), n, L.o.w.data(), L.o.b.data(), z.data(), ne, ne, threads);
            for (int64_t i = 0; i < n * ne; ++i) x[(size_t)i] += z[(size_t)i];

            parallel_rows(n, threads, [&](int64_t r0, int64_t r1) {
                for (int64_t r = r0; r < r1; ++r)
                    layer_norm(x.data() + r * ne, L.ln2.data(), L.ln2b.data(), z.data() + r * ne, ne, cfg.eps);
            });
            linear(z.data(), n, L.up.w.data(), L.up.b.data(), ff.data(), ne, cfg.ff, threads);
            for (int64_t i = 0; i < n * cfg.ff; ++i) ff[(size_t)i] = gelu(ff[(size_t)i]);
            linear(ff.data(), n, L.down.w.data(), L.down.b.data(), z.data(), cfg.ff, ne, threads);
            for (int64_t i = 0; i < n * ne; ++i) x[(size_t)i] += z[(size_t)i];
        }

        // projector: linear_1 -> GELU -> linear_2, then merge 2x2 blocks in channel order
        std::vector<float> p1((size_t)n * cfg.projector_hidden);
        linear(x.data(), n, mm1.w.data(), mm1.b.data(), p1.data(), ne, cfg.projector_hidden, threads);
        for (float& f : p1) f = gelu(f);
        std::vector<float> p2((size_t)n * cfg.text_hidden);
        linear(p1.data(), n, mm2.w.data(), mm2.b.data(), p2.data(), cfg.projector_hidden, cfg.text_hidden, threads);

        const int64_t unit = cfg.merge * cfg.merge;
        const int64_t merged = n / unit;
        std::vector<float> m((size_t)merged * cfg.text_hidden * unit);
        for (int64_t b = 0; b < merged; ++b)
            std::memcpy(m.data() + (size_t)b * cfg.text_hidden * unit,
                        p2.data() + (size_t)b * unit * cfg.text_hidden,
                        (size_t)cfg.text_hidden * unit * sizeof(float));

        std::vector<float> m1((size_t)merged * cfg.projector_hidden);
        linear(m.data(), merged, merge1.w.data(), merge1.b.data(), m1.data(), cfg.text_hidden * unit,
               cfg.projector_hidden, threads);
        for (float& f : m1) f = gelu(f);
        ImageEmbeddings out;
        out.tokens = merged;
        out.dim = cfg.text_hidden;
        out.width = width;
        out.height = height;
        out.data.resize((size_t)merged * cfg.text_hidden);
        linear(m1.data(), merged, merge2.w.data(), merge2.b.data(), out.data.data(), cfg.projector_hidden,
               cfg.text_hidden, threads);
        return out;
    }
};

void smart_resize(int64_t height, int64_t width, int64_t& out_h, int64_t& out_w, int64_t factor,
                  int64_t min_pixels, int64_t max_pixels) {
    if (height <= 0 || width <= 0) throw std::runtime_error("vision: empty image");
    if ((double)std::max(height, width) / (double)std::min(height, width) > 200.0)
        throw std::runtime_error("vision: the image's aspect ratio is over 200:1");
    auto round_to = [&](int64_t v) { return (int64_t)std::nearbyint((double)v / (double)factor) * factor; };
    auto floor_to = [&](double v) { return (int64_t)std::floor(v / (double)factor) * factor; };
    auto ceil_to = [&](double v) { return (int64_t)std::ceil(v / (double)factor) * factor; };
    int64_t h = std::max(factor, round_to(height));
    int64_t w = std::max(factor, round_to(width));
    if (h * w > max_pixels) {
        const double beta = std::sqrt((double)height * (double)width / (double)max_pixels);
        h = floor_to((double)height / beta);
        w = floor_to((double)width / beta);
    } else if (h * w < min_pixels) {
        const double beta = std::sqrt((double)min_pixels / ((double)height * (double)width));
        h = ceil_to((double)height * beta);
        w = ceil_to((double)width * beta);
    }
    out_h = std::max(factor, h);
    out_w = std::max(factor, w);
}

Vision::Vision(const std::string& mmproj_path, int threads) : impl_(new Impl) {
    try {
        impl_->load(mmproj_path, threads);
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

Vision::~Vision() { delete impl_; }

const VisionConfig& Vision::config() const { return impl_->cfg; }
const std::string& Vision::path() const { return impl_->path; }

int64_t Vision::tokens_for(int64_t width, int64_t height) const {
    int64_t h = 0, w = 0;
    smart_resize(height, width, h, w, impl_->cfg.patch * impl_->cfg.merge, 4 * 28 * 28, impl_->cfg.max_pixels);
    return (h / impl_->cfg.patch) * (w / impl_->cfg.patch) / (impl_->cfg.merge * impl_->cfg.merge);
}

ImageEmbeddings Vision::encode_pixels(const float* pixels, int64_t n_patch, int64_t width, int64_t height) const {
    // The checkpoint's pixel_values are [n_patch, 3, 2, 14, 14]: one image duplicated over two temporal
    // frames.  The tower reads the 2D frame (the conv slices are already summed), so keep frame 0.
    const VisionConfig& c = impl_->cfg;
    const int64_t kk = c.patch * c.patch;
    const int64_t stride = c.temporal * kk;
    std::vector<float> x((size_t)n_patch * 3 * kk);
    for (int64_t p = 0; p < n_patch; ++p)
        for (int64_t ci = 0; ci < 3; ++ci)
            std::memcpy(x.data() + ((size_t)p * 3 + ci) * kk,
                        pixels + ((size_t)p * 3 + ci) * stride, (size_t)kk * sizeof(float));
    return impl_->run(x.data(), n_patch, width, height);
}

ImageEmbeddings Vision::encode_rgb(const uint8_t* rgb, int64_t width, int64_t height, int64_t max_tokens) const {
    const VisionConfig& c = impl_->cfg;
    std::vector<float> pixels;
    int64_t n = 0, rw = 0, rh = 0;
    preprocess_rgb(rgb, width, height, pixels, n, rw, rh);
    if (max_tokens > 0 && n / (c.merge * c.merge) > max_tokens)
        throw std::runtime_error("vision: the image becomes " + std::to_string(n / (c.merge * c.merge)) +
                                 " image tokens, over the limit of " + std::to_string(max_tokens));
    // collapse the duplicated temporal frame back to the 2D frame the tower reads
    const int64_t kk = c.patch * c.patch;
    std::vector<float> frame((size_t)n * 3 * kk);
    for (int64_t p = 0; p < n; ++p)
        for (int64_t ci = 0; ci < 3; ++ci)
            std::memcpy(frame.data() + ((size_t)p * 3 + ci) * kk,
                        pixels.data() + ((size_t)p * 3 + ci) * (c.temporal * kk),
                        (size_t)kk * sizeof(float));
    return impl_->run(frame.data(), n, rw, rh);
}

void Vision::preprocess_rgb(const uint8_t* rgb, int64_t width, int64_t height,
                            std::vector<float>& pixels, int64_t& n_patch, int64_t& resized_w,
                            int64_t& resized_h) const {
    const VisionConfig& c = impl_->cfg;
    int64_t rh = 0, rw = 0;
    smart_resize(height, width, rh, rw, c.patch * c.merge, 4 * 28 * 28, c.max_pixels);
    std::vector<uint8_t> resized;
    resize_rgb(rgb, height, width, rh, rw, resized);
    const int64_t gh = rh / c.patch, gw = rw / c.patch;
    const int64_t n = gh * gw;
    const int64_t kk = c.patch * c.patch;
    pixels.assign((size_t)n * 3 * c.temporal * kk, 0.0f);
    int64_t p = 0;
    for (int64_t bh = 0; bh < gh / c.merge; ++bh)
        for (int64_t bw = 0; bw < gw / c.merge; ++bw)
            for (int64_t mh = 0; mh < c.merge; ++mh)
                for (int64_t mw = 0; mw < c.merge; ++mw) {
                    const int64_t y0 = (bh * c.merge + mh) * c.patch;
                    const int64_t x0 = (bw * c.merge + mw) * c.patch;
                    for (int64_t kh = 0; kh < c.patch; ++kh)
                        for (int64_t kw = 0; kw < c.patch; ++kw) {
                            const uint8_t* src = resized.data() + ((size_t)(y0 + kh) * rw + (x0 + kw)) * 3;
                            for (int ci = 0; ci < 3; ++ci) {
                                const float v = ((float)src[ci] / 255.0f - c.mean[(size_t)ci]) / c.std[(size_t)ci];
                                const size_t idx = (size_t)(((ci * c.temporal) * c.patch + kh) * c.patch + kw);
                                pixels[(size_t)p * 3 * c.temporal * kk + idx] = v;
                                pixels[(size_t)p * 3 * c.temporal * kk + idx + (size_t)kk] = v;
                            }
                        }
                    ++p;
                }
    n_patch = n;
    resized_w = rw;
    resized_h = rh;
}

ImageEmbeddings Vision::encode_file(const std::string& file, int64_t max_tokens) const {
    int w = 0, h = 0, comp = 0;
    unsigned char* data = stbi_load(file.c_str(), &w, &h, &comp, 3);
    if (!data) throw std::runtime_error("vision: cannot read " + file + " (" + stbi_failure_reason() + ")");
    ImageEmbeddings out;
    try {
        out = encode_rgb(data, w, h, max_tokens);
    } catch (...) {
        stbi_image_free(data);
        throw;
    }
    stbi_image_free(data);
    return out;
}

void Vision::preprocess_file(const std::string& file, std::vector<float>& pixels, int64_t& n_patch,
                             int64_t& resized_w, int64_t& resized_h) const {
    int w = 0, h = 0, comp = 0;
    unsigned char* data = stbi_load(file.c_str(), &w, &h, &comp, 3);
    if (!data) throw std::runtime_error("vision: cannot read " + file + " (" + stbi_failure_reason() + ")");
    try {
        preprocess_rgb(data, w, h, pixels, n_patch, resized_w, resized_h);
    } catch (...) {
        stbi_image_free(data);
        throw;
    }
    stbi_image_free(data);
}

}  // namespace strata::minimax
