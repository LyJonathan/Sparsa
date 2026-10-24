// src/artifact/gguf_reader_test.cpp - the reader refuses what it cannot arbitrate (synthetic GGUF, no model,
// no GPU).
//
//   1. two tensors with distinct names: the file opens and find() sees both;
//   2. the same name twice: refused at open, and the error names the tensor and the file - GGUF has no
//      index to say which of the two a name means, and find() is first-match, so the alternative is a
//      lookup that silently picks by position.
//   3. no tensors, and the file ends at its header, before the 32-byte-aligned data start: the file opens,
//      with data_start() clamped to the file size.  A split model's first shard can hold metadata only and
//      end unpadded (unsloth's UD-Q5_K_XL shard 1 ends 6 bytes before its aligned data start; llama.cpp's
//      gguf-py reads it); nothing is ever read from its data section.
//   4. one tensor, and the file ends at its header: still refused at open ("data section starts past EOF").
//
// The fixture is a minimal GGUF v3 written here (header, no metadata, two F32[8] tensors, 32-byte
// alignment), so the test needs neither gguf-py nor a shard.  The file is closed before it is removed:
// on Windows an open mapping keeps it.
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// A GGUF v3 file of F32[8] tensors named `names`, laid out 32 bytes apart from a 32-byte-aligned data start.
// `header_only`: the file ends right after the tensor directory - no alignment padding, no data.
std::filesystem::path write_gguf(const std::vector<std::string>& names, bool header_only = false,
                                 uint64_t align_override = 0) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, names.size());  // n_tensors
    put<uint64_t>(b, align_override ? 1 : 0);   // n_kv
    if (align_override) {
        put_str(b, "general.alignment");
        put<uint32_t>(b, 4);         // GGUF_METADATA_VALUE_TYPE_UINT32
        put<uint32_t>(b, (uint32_t) align_override);
    }
    for (size_t i = 0; i < names.size(); ++i) {
        put_str(b, names[i]);
        put<uint32_t>(b, 1);         // n_dims
        put<uint64_t>(b, 8);
        put<uint32_t>(b, 0);         // F32
        put<uint64_t>(b, 32 * i);    // offset from data_start
    }
    if (!header_only) b.resize((b.size() + 31) / 32 * 32 + 32 * names.size(), 0);
    const auto path = std::filesystem::temp_directory_path() / "strata_gguf_reader_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}

// One tensor of `type` with an untrusted shape, and a data section of `payload` zero bytes.  Used to check
// that a crafted file is rejected by the arithmetic, never by an out-of-bounds read.
std::filesystem::path write_shape_gguf(const std::string& name, const std::vector<uint64_t>& shape,
                                       uint32_t type, uint64_t payload) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);             // n_tensors
    put<uint64_t>(b, 0);             // n_kv
    put_str(b, name);
    put<uint32_t>(b, (uint32_t) shape.size());
    for (uint64_t d : shape) put<uint64_t>(b, d);
    put<uint32_t>(b, type);
    put<uint64_t>(b, 0);             // offset
    b.resize((b.size() + 31) / 32 * 32 + payload, 0);
    const auto path = std::filesystem::temp_directory_path() / "strata_gguf_reader_shape_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}
}  // namespace

int main() {
    std::printf("gguf_reader_test\n");
    {
        const auto path = write_gguf({"blk.0.attn_q.weight", "blk.0.attn_k.weight"});
        std::string err;
        size_t n = 0;
        bool both = false;
        try {
            strata::GgufFile g(path.string());
            n = g.tensors().size();
            both = g.find("blk.0.attn_q.weight") && g.find("blk.0.attn_k.weight");
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && n == 2 && both, "distinct names: the file opens and find() sees both tensors");
    }
    {
        const auto path = write_gguf({"blk.0.attn_q.weight", "blk.0.attn_q.weight"});
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(!err.empty(), "the same name twice: refused at open");
        check(err.find("blk.0.attn_q.weight") != std::string::npos, "  the error names the tensor");
        check(err.find(path.filename().string()) != std::string::npos, "  the error names the file");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf({}, true);   // 24 bytes: the aligned data start (32) is past EOF
        std::string err;
        uint64_t size = 0, start = 1;
        size_t n = 1;
        try {
            strata::GgufFile g(path.string());
            size = g.file_size();
            start = g.data_start();
            n = g.tensors().size();
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && n == 0, "no tensors, ends before the aligned data start: the file opens");
        check(err.empty() && start == size, "  data_start() is the file size (nothing to read past it)");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf({"blk.0.attn_q.weight"}, true);
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.find("data section starts past EOF") != std::string::npos,
              "one tensor, ends before the aligned data start: refused at open");
    }
    // Untrusted shapes: a GGUF is downloaded from the internet.  tensor_payload_bytes must return 0 (so
    // in_bounds says no) rather than wrap to a small number that passes both the size and the bounds check.
    {
        const uint64_t huge = 1ull << 40;
        const auto path = write_shape_gguf("blk.0.ffn_gate_exps.weight", {huge, huge}, 160 /* IQ2_KT */, 64);
        std::string err;
        bool ok = false;
        try {
            strata::GgufModel m({path.string()});
            const auto* t = m.find("blk.0.ffn_gate_exps.weight");
            ok = t && m.in_bounds(*t, 0);
            if (t) std::printf("  (payload %llu bytes at %llu)\n", (unsigned long long) strata::tensor_payload_bytes(*t),
                               (unsigned long long) t->offset);
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && !ok, "IQ2_KT with 2^40 x 2^40 values: no payload size, in_bounds refuses it");
    }
    {
        const uint64_t huge = 1ull << 40;
        const auto path = write_shape_gguf("blk.0.attn_q.weight", {huge, huge}, 0 /* F32 */, 64);
        std::string err;
        bool ok = false;
        try {
            strata::GgufModel m({path.string()});
            const auto* t = m.find("blk.0.attn_q.weight");
            ok = t && m.in_bounds(*t, 0);
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && !ok, "F32 with 2^40 x 2^40 values: the checked ggml path refuses it too");
    }
    {
        // shape[0] wider than int64: the ikq row geometry must not be handed the truncated width.
        const auto path = write_shape_gguf("blk.0.ffn_down_exps.weight", {0x8000000000000010ull, 2}, 163 /* IQ3_KT */, 64);
        std::string err;
        bool ok = false;
        try {
            strata::GgufModel m({path.string()});
            const auto* t = m.find("blk.0.ffn_down_exps.weight");
            ok = t && m.in_bounds(*t, 0);
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && !ok, "IQ3_KT with a width above INT64_MAX: refused");
    }
    {
        // an absurd general.alignment must not wrap the aligned data start
        const auto path = write_gguf({"blk.0.attn_q.weight"}, true, 0xFFFFFFFFu);
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.find("general.alignment") != std::string::npos, "general.alignment = 2^32-1: refused at open");
    }
    std::printf(g_fail ? "gguf_reader_test: %d FAILED\n" : "gguf_reader_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
