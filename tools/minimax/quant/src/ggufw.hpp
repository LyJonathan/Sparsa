#pragma once

// Streaming GGUF v3 writer. Tensor offsets are relative to the data section and
// equal the running sum of GGML_PAD(nbytes, alignment), which is what
// ggml's reader checks. The data section itself starts on an alignment boundary
// measured from the start of the file.

#include "mapped.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace mmq {

static_assert(static_cast<unsigned char>(uint16_t{1}) == 1, "GGUF writer emits host order and requires little endian");

enum GgufType : uint32_t {
    GGUF_U8 = 0,
    GGUF_I8 = 1,
    GGUF_U16 = 2,
    GGUF_I16 = 3,
    GGUF_U32 = 4,
    GGUF_I32 = 5,
    GGUF_F32 = 6,
    GGUF_BOOL = 7,
    GGUF_STR = 8,
    GGUF_ARR = 9,
    GGUF_U64 = 10,
    GGUF_I64 = 11,
    GGUF_F64 = 12,
};

inline uint64_t gguf_pad(uint64_t x, uint64_t a) {
    return (x + a - 1) & ~(a - 1);
}

struct TensorDesc {
    std::string name;
    int n_dims = 1;
    int64_t ne[4] = {1, 1, 1, 1};
    uint32_t type = 0;
    uint64_t nbytes = 0;
    uint64_t offset = 0;
};

class GgufWriter {
public:
    uint32_t alignment = 32;

    void open(const std::filesystem::path & path) {
        f = open_file(path, "wb");
        if (!f) {
            throw std::runtime_error("cannot create " + path.string());
        }
    }

    void add_u16(const std::string & k, uint16_t v) {
        Kv kv;
        kv.key = k;
        kv.type = GGUF_U16;
        const auto * p = reinterpret_cast<const uint8_t *>(&v);
        kv.val.insert(kv.val.end(), p, p + sizeof(v));
        kvs.push_back(std::move(kv));
    }
    void add_u32(const std::string & k, uint32_t v) { add_pod(k, GGUF_U32, v); }
    void add_i32(const std::string & k, int32_t v) { add_pod(k, GGUF_I32, v); }
    void add_u64(const std::string & k, uint64_t v) { add_pod(k, GGUF_U64, v); }
    void add_f32(const std::string & k, float v) { add_pod(k, GGUF_F32, v); }
    void add_bool(const std::string & k, bool v) {
        const uint8_t b = v ? 1 : 0;
        add_pod(k, GGUF_BOOL, b);
    }
    void add_str(const std::string & k, const std::string & v) {
        Kv kv;
        kv.key = k;
        kv.type = GGUF_STR;
        append_str(kv.val, v);
        kvs.push_back(std::move(kv));
    }
    void add_str_array(const std::string & k, const std::vector<std::string> & xs) {
        Kv kv;
        kv.key = k;
        kv.type = GGUF_ARR;
        append_u32(kv.val, GGUF_STR);
        append_u64(kv.val, xs.size());
        for (const auto & s : xs) {
            append_str(kv.val, s);
        }
        kvs.push_back(std::move(kv));
    }
    void add_i32_array(const std::string & k, const std::vector<int32_t> & xs) {
        Kv kv;
        kv.key = k;
        kv.type = GGUF_ARR;
        append_u32(kv.val, GGUF_I32);
        append_u64(kv.val, xs.size());
        for (int32_t x : xs) {
            append_u32(kv.val, static_cast<uint32_t>(x));
        }
        kvs.push_back(std::move(kv));
    }

    void add_tensor(TensorDesc t) { tensors.push_back(std::move(t)); }

    void write_header() {
        if (!f) {
            throw std::runtime_error("gguf: file not open");
        }
        uint64_t off = 0;
        for (auto & t : tensors) {
            t.offset = off;
            off += gguf_pad(t.nbytes, alignment);
        }
        std::vector<uint8_t> hdr;
        append_u32(hdr, 0x46554747u);  // GGUF
        append_u32(hdr, 3);
        append_u64(hdr, tensors.size());
        append_u64(hdr, kvs.size());
        for (const auto & kv : kvs) {
            append_str(hdr, kv.key);
            append_u32(hdr, kv.type);
            hdr.insert(hdr.end(), kv.val.begin(), kv.val.end());
        }
        for (const auto & t : tensors) {
            append_str(hdr, t.name);
            append_u32(hdr, static_cast<uint32_t>(t.n_dims));
            for (int d = 0; d < t.n_dims; ++d) {
                append_u64(hdr, static_cast<uint64_t>(t.ne[d]));
            }
            append_u32(hdr, t.type);
            append_u64(hdr, t.offset);
        }
        write_raw(hdr.data(), hdr.size());
        const uint64_t pos = tell();
        const uint64_t aligned = gguf_pad(pos, alignment);
        zeros(aligned - pos);
        data_start = tell();
        next_tensor = 0;
        filled = 0;
    }

    void write_payload(const void * p, size_t n) {
        if (next_tensor >= tensors.size()) {
            throw std::runtime_error("gguf: payload past last tensor");
        }
        if (filled + n > tensors[next_tensor].nbytes) {
            throw std::runtime_error("gguf: tensor overrun " + tensors[next_tensor].name);
        }
        write_raw(p, n);
        filled += n;
    }

    void end_tensor() {
        if (next_tensor >= tensors.size()) {
            throw std::runtime_error("gguf: end_tensor past last tensor");
        }
        if (filled != tensors[next_tensor].nbytes) {
            throw std::runtime_error("gguf: short tensor " + tensors[next_tensor].name +
                                     " wrote " + std::to_string(filled) + " expected " +
                                     std::to_string(tensors[next_tensor].nbytes));
        }
        const uint64_t pos = tell();
        const uint64_t rel = pos - data_start;
        const uint64_t aligned = gguf_pad(rel, alignment);
        zeros(aligned - rel);
        ++next_tensor;
        filled = 0;
    }

    void close() {
        if (f && next_tensor != tensors.size()) {
            std::fclose(f);
            f = nullptr;
            throw std::runtime_error("gguf: closed before all tensors were written");
        }
        if (f) {
            std::fclose(f);
            f = nullptr;
        }
    }

    ~GgufWriter() {
        if (f) {
            std::fclose(f);
            f = nullptr;
        }
    }

    const std::vector<TensorDesc> & tensor_list() const { return tensors; }

private:
    struct Kv {
        std::string key;
        uint32_t type = 0;
        std::vector<uint8_t> val;
    };

    template <typename T>
    void add_pod(const std::string & k, uint32_t type, T v) {
        Kv kv;
        kv.key = k;
        kv.type = type;
        static_assert(sizeof(T) == 1 || sizeof(T) == 4 || sizeof(T) == 8, "pod width");
        const auto * p = reinterpret_cast<const uint8_t *>(&v);
        kv.val.insert(kv.val.end(), p, p + sizeof(T));
        kvs.push_back(std::move(kv));
    }

    static void append_u32(std::vector<uint8_t> & b, uint32_t v) {
        b.push_back(static_cast<uint8_t>(v));
        b.push_back(static_cast<uint8_t>(v >> 8));
        b.push_back(static_cast<uint8_t>(v >> 16));
        b.push_back(static_cast<uint8_t>(v >> 24));
    }
    static void append_u64(std::vector<uint8_t> & b, uint64_t v) {
        for (int s = 0; s < 8; ++s) {
            b.push_back(static_cast<uint8_t>(v >> (8 * s)));
        }
    }
    static void append_str(std::vector<uint8_t> & b, const std::string & s) {
        append_u64(b, s.size());
        b.insert(b.end(), s.begin(), s.end());
    }

    void write_raw(const void * p, size_t n) {
        if (n == 0) {
            return;
        }
        const size_t w = std::fwrite(p, 1, n, f);
        if (w != n) {
            throw std::runtime_error("gguf: short write");
        }
    }
    void zeros(uint64_t n) {
        uint8_t buf[4096] = {};
        while (n) {
            const size_t c = static_cast<size_t>(n > sizeof(buf) ? sizeof(buf) : n);
            write_raw(buf, c);
            n -= c;
        }
    }
    uint64_t tell() const {
#ifdef _WIN32
        const auto p = _ftelli64(f);
#else
        const auto p = ftello(f);
#endif
        if (p < 0) {
            throw std::runtime_error("gguf: tell failed");
        }
        return static_cast<uint64_t>(p);
    }

    FILE * f = nullptr;
    std::vector<Kv> kvs;
    std::vector<TensorDesc> tensors;
    uint64_t data_start = 0;
    size_t next_tensor = 0;
    uint64_t filled = 0;
};

}  // namespace mmq
