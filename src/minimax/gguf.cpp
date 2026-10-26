// src/minimax/gguf.cpp - the expected `minimax-m3` tensor directory and the --minimax-inspect report.
#include "strata/minimax/gguf.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#include "strata/artifact/gguf_reader.hpp"

namespace strata::minimax {
namespace {

struct Expected {
    std::string name;
    TensorClass cls;
    std::vector<uint64_t> shape;
    bool optional = false;
};

void expect(std::vector<Expected>& out, std::string name, TensorClass cls, std::vector<uint64_t> shape) {
    out.push_back(Expected{std::move(name), cls, std::move(shape), false});
}

// Optional tensors: absent in a file that does not use them, accepted (and shape-checked) when present.
void expect_opt(std::vector<Expected>& out, std::string name, TensorClass cls, std::vector<uint64_t> shape) {
    out.push_back(Expected{std::move(name), cls, std::move(shape), true});
}

std::string shape_text(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) r += ", ";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

uint64_t product(const std::vector<uint64_t>& s) {
    uint64_t n = 1;
    for (uint64_t d : s) {
        if (d == 0 || n > (std::numeric_limits<uint64_t>::max)() / d) return 0;   // 0 = overflow
        n *= d;
    }
    return n;
}

/// Every tensor the P2 forward pass will read, from spec F.2 of _ref/MINIMAX_REFERENCE_SPEC.md.  Shapes are GGUF
/// order (ne0 first), i.e. the reverse of the numpy/HF shape.  Experts are expert-major and stacked, as the
/// converter writes them; indexer tensors are F32 in the converter but that is the file's choice, not a shape.
std::vector<Expected> expected_tensors(const Config& c) {
    std::vector<Expected> out;
    const uint64_t ne = (uint64_t) c.n_embd;
    const uint64_t hq = (uint64_t) (c.n_head * c.head_dim);
    const uint64_t hkv = (uint64_t) (c.n_head_kv * c.head_dim);
    const uint64_t ff = (uint64_t) c.n_ff_dense;
    const uint64_t ffe = (uint64_t) c.n_ff_exp;
    const uint64_t shexp = (uint64_t) (c.n_ff_exp * c.n_shared_experts);
    const uint64_t nxp = (uint64_t) c.n_expert;
    const uint64_t iq = (uint64_t) (c.idx_heads * c.idx_dim);

    expect(out, "token_embd.weight", TensorClass::Embd, {ne, (uint64_t) c.n_vocab});
    expect(out, "output_norm.weight", TensorClass::Dense, {ne});
    expect(out, "output.weight", TensorClass::Output, {ne, (uint64_t) c.n_vocab});

    for (int64_t i = 0; i < c.n_layers; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        expect(out, b + "attn_norm.weight", TensorClass::Dense, {ne});
        expect(out, b + "attn_q.weight", TensorClass::Dense, {ne, hq});
        expect(out, b + "attn_k.weight", TensorClass::Dense, {ne, hkv});
        expect(out, b + "attn_v.weight", TensorClass::Dense, {ne, hkv});
        expect(out, b + "attn_output.weight", TensorClass::Dense, {hq, ne});
        expect(out, b + "attn_q_norm.weight", TensorClass::Dense, {(uint64_t) c.head_dim});
        expect(out, b + "attn_k_norm.weight", TensorClass::Dense, {(uint64_t) c.head_dim});
        expect(out, b + "ffn_norm.weight", TensorClass::Dense, {ne});

        if (!c.is_moe(i)) {
            expect(out, b + "ffn_gate.weight", TensorClass::Dense, {ne, ff});
            expect(out, b + "ffn_up.weight", TensorClass::Dense, {ne, ff});
            expect(out, b + "ffn_down.weight", TensorClass::Dense, {ff, ne});
        } else {
            expect(out, b + "ffn_gate_inp.weight", TensorClass::Dense, {ne, nxp});
            expect(out, b + "exp_probs_b.bias", TensorClass::Dense, {nxp});
            expect(out, b + "ffn_gate_exps.weight", TensorClass::Experts, {ne, ffe, nxp});
            expect(out, b + "ffn_down_exps.weight", TensorClass::Experts, {ffe, ne, nxp});
            expect(out, b + "ffn_up_exps.weight", TensorClass::Experts, {ne, ffe, nxp});
            // MM-NVFP4: per-expert global scales (a divisor) beside the exact NVFP4 expert bytes.
            expect_opt(out, b + "ffn_gate_exps.scale", TensorClass::Experts, {nxp});
            expect_opt(out, b + "ffn_up_exps.scale", TensorClass::Experts, {nxp});
            expect_opt(out, b + "ffn_down_exps.scale", TensorClass::Experts, {nxp});
            if (c.n_shared_experts > 0) {
                expect(out, b + "ffn_gate_shexp.weight", TensorClass::Dense, {ne, shexp});
                expect(out, b + "ffn_down_shexp.weight", TensorClass::Dense, {shexp, ne});
                expect(out, b + "ffn_up_shexp.weight", TensorClass::Dense, {ne, shexp});
            }
        }

        if (c.has_indexer(i)) {
            expect(out, b + "indexer.q_proj.weight", TensorClass::Dense, {ne, iq});
            expect(out, b + "indexer.k_proj.weight", TensorClass::Dense, {ne, (uint64_t) c.idx_dim});
            expect(out, b + "indexer.q_norm.weight", TensorClass::Dense, {(uint64_t) c.idx_dim});
            expect(out, b + "indexer.k_norm.weight", TensorClass::Dense, {(uint64_t) c.idx_dim});
        }
    }
    return out;
}

}  // namespace

const char* tensor_class_name(TensorClass c) {
    switch (c) {
    case TensorClass::Dense: return "dense";
    case TensorClass::Experts: return "experts";
    case TensorClass::Embd: return "embd";
    case TensorClass::Output: return "output";
    }
    return "?";
}

Model load_model(const std::string& path, bool require_m3_1) {
    const GgufModel file = GgufModel::open(path);
    Model m;
    m.config = from_gguf(file.meta(), file);

    const std::string bad = check_structure(m.config);
    if (!bad.empty()) throw std::runtime_error(path + ": " + bad);
    m.m3_1_mismatches = check_m3_1(m.config);
    if (require_m3_1 && !m.m3_1_mismatches.empty()) {
        std::string err = path + ": this is not MiniMax-M3.1-Flash's geometry:";
        for (const std::string& s : m.m3_1_mismatches) err += "\n  " + s;
        throw std::runtime_error(err);
    }

    const std::vector<Expected> expected = expected_tensors(m.config);
    std::map<std::string, const Expected*> by_name;
    for (const Expected& e : expected) by_name.emplace(e.name, &e);

    m.tensors.reserve(expected.size());
    for (const Expected& e : expected) {
        size_t shard = 0;
        const TensorInfo* t = file.find(e.name, &shard);
        if (!t) {
            if (e.optional) continue;
            throw std::runtime_error(path + ": missing tensor " + e.name);
        }
        if (t->shape != e.shape)
            throw std::runtime_error(path + ": " + e.name + " has shape " + shape_text(t->shape) + ", expected " +
                                     shape_text(e.shape));
        if (!file.in_bounds(*t, shard))
            throw std::runtime_error(path + ": " + e.name + " has an unknown type or lies past the end of its shard");

        TensorRecord r;
        r.name = e.name;
        r.cls = e.cls;
        r.shape = t->shape;
        r.type = t->type;
        r.offset = t->offset;
        r.elements = product(t->shape);
        r.bytes = tensor_payload_bytes(*t);
        r.shard = shard;
        if (r.elements == 0)
            throw std::runtime_error(path + ": " + e.name + " has a shape whose element count overflows");
        if (r.bytes == 0) throw std::runtime_error(path + ": " + e.name + " has no payload size (type " +
                                                   std::to_string(t->type) + ")");
        m.tensors.push_back(std::move(r));
    }

    // A higher layer index than the file has, or a per-layer tensor this layout does not name, would be silently
    // ignored by a loader that only looks up known names: refuse it, as core::check_gguf_layout does for qwen4exp.
    for (size_t s = 0; s < file.size(); ++s)
        for (const TensorInfo& t : file.shard(s).tensors()) {
            if (t.name.rfind("blk.", 0) != 0) continue;
            if (!by_name.count(t.name))
                throw std::runtime_error(path + ": unexpected per-layer tensor " + t.name +
                                         " (not in this architecture's layout)");
        }

    for (const TensorRecord& r : m.tensors) {
        m.bytes_total += r.bytes;
        switch (r.cls) {
        case TensorClass::Dense: m.bytes_dense += r.bytes; break;
        case TensorClass::Experts: m.bytes_experts += r.bytes; break;
        case TensorClass::Embd: m.bytes_embd += r.bytes; break;
        case TensorClass::Output: m.bytes_output += r.bytes; break;
        }
    }
    return m;
}

std::string report(const Model& m) {
    std::ostringstream o;
    o << "MiniMax GGUF: " << m.config.n_layers << " layers, embd " << m.config.n_embd << ", " << m.tensors.size()
      << " tensors mapped\n";
    o << "  " << summary(m.config) << "\n";
    if (m.m3_1_mismatches.empty()) {
        o << "  geometry: MiniMax-M3.1-Flash signature OK\n";
    } else {
        o << "  geometry: NOT MiniMax-M3.1-Flash (inspection only; --serve refuses it):\n";
        for (const std::string& s : m.m3_1_mismatches) o << "    - " << s << "\n";
    }

    size_t wname = 4, wtype = 4;
    for (const TensorRecord& t : m.tensors) {
        wname = (std::max)(wname, t.name.size());
        wtype = (std::max)(wtype, std::string(ggml_type_name(t.type)).size());
    }
    o << "\n  " << std::left << std::setw(4) << "#" << "  " << std::setw((int) wname) << "tensor" << "  "
      << std::setw((int) wtype) << "type" << "  " << std::right << std::setw(24) << "shape" << "  " << std::setw(14)
      << "bytes" << "  " << std::setw(7) << "offset" << "  class\n";
    o << "  " << std::string(4 + 2 + wname + 2 + wtype + 2 + 24 + 2 + 14 + 2 + 7 + 2 + 8, '-') << "\n";
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const TensorRecord& t = m.tensors[i];
        o << "  " << std::left << std::setw(4) << i << "  " << std::setw((int) wname) << t.name << "  "
          << std::setw((int) wtype) << ggml_type_name(t.type) << "  " << std::right << std::setw(24)
          << shape_text(t.shape) << "  " << std::setw(14) << t.bytes << "  " << std::setw(7) << t.offset << "  "
          << tensor_class_name(t.cls) << "\n";
    }
    o << "\n  bytes by class:\n";
    size_t counts[4] = {0, 0, 0, 0};
    for (const TensorRecord& t : m.tensors) ++counts[(int) t.cls];
    o << "    " << std::left << std::setw(10) << "dense" << std::right << std::setw(6) << counts[0] << " tensors  "
      << std::setw(14) << m.bytes_dense << "\n";
    o << "    " << std::left << std::setw(10) << "experts" << std::right << std::setw(6) << counts[1] << " tensors  "
      << std::setw(14) << m.bytes_experts << "\n";
    o << "    " << std::left << std::setw(10) << "embd" << std::right << std::setw(6) << counts[2] << " tensors  "
      << std::setw(14) << m.bytes_embd << "\n";
    o << "    " << std::left << std::setw(10) << "output" << std::right << std::setw(6) << counts[3] << " tensors  "
      << std::setw(14) << m.bytes_output << "\n";
    o << "    " << std::left << std::setw(10) << "total" << std::right << std::setw(6) << m.tensors.size()
      << " tensors  " << std::setw(14) << m.bytes_total << "\n";
    return o.str();
}

}  // namespace strata::minimax
