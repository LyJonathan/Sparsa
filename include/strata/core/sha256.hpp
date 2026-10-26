// include/strata/core/sha256.hpp - a small, dependency-free SHA-256, used only by the opt-in proofs that compare
// the bytes a tier returned with the bytes a pack holds.  Not on any hot path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

/// SHA-256 of `len` bytes at `data`, as 64 lowercase hex characters.
std::string sha256_hex(const void* data, size_t len);

}  // namespace strata::core
