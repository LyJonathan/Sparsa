// include/strata/minimax/gguf.hpp - the MiniMax-M3.1-Flash GGUF tensor map (llama.cpp's `minimax-m3` names).
//
// P2a: this reads the directory only - name, class, GGML type, shape, offset, byte count - and checks every tensor
// the forward pass will need.  No payload is uploaded or repacked here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/minimax/config.hpp"

namespace strata::minimax {

enum class TensorClass { Dense = 0, Experts, Embd, Output };

const char* tensor_class_name(TensorClass c);

struct TensorRecord {
    std::string name;
    TensorClass cls = TensorClass::Dense;
    std::vector<uint64_t> shape;   // GGUF order (dim 0 varies fastest)
    uint32_t type = 0;             // ggml type id
    uint64_t offset = 0;           // from the shard's data section
    uint64_t elements = 0;
    uint64_t bytes = 0;
    size_t shard = 0;
};

struct Model {
    Config config;
    std::vector<TensorRecord> tensors;
    uint64_t bytes_dense = 0;
    uint64_t bytes_experts = 0;
    uint64_t bytes_embd = 0;
    uint64_t bytes_output = 0;
    uint64_t bytes_total = 0;
    std::vector<std::string> m3_1_mismatches;
};

/// Open `path` (any shard of a split model), read its config and map every expected tensor.  Throws
/// std::runtime_error naming the first missing tensor, the first shape that differs, or a malformed byte count.
/// `require_m3_1` also refuses a file that is not MiniMax-M3.1-Flash's exact fingerprint (serve path); the inspect
/// path passes false and reports the differences instead.
Model load_model(const std::string& path, bool require_m3_1);

/// The per-tensor table and the byte totals by class, as printed by `--minimax-inspect`.
std::string report(const Model& m);

}  // namespace strata::minimax
