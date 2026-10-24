#pragma once

// Row quantization through ggml_quantize_chunk. Chunks stay inside one expert:
// the caller passes the importance row for that expert (length n_per_row), and
// every row of the chunk uses it. ggml_quantize_init is invoked on the calling
// thread before workers start; the chunk function's own init is then a no-op.

#include <ggml.h>

#include "tqio.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mmq {

inline bool type_needs_imatrix(ggml_type type) {
    return type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
           type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS;
}

inline int gguf_n_dims(const int64_t ne[4]) {
    int n = 1;
    for (int i = 1; i < 4; ++i) {
        if (ne[i] > 1) {
            n = i + 1;
        }
    }
    return n;
}

inline uint64_t tensor_nbytes(ggml_type type, const int64_t ne[4]) {
    if (ne[0] <= 0 || ne[1] <= 0 || ne[2] <= 0 || ne[3] <= 0) {
        throw std::runtime_error("non-positive tensor dimension");
    }
    const int64_t nrows = ne[1] * ne[2] * ne[3];
    return static_cast<uint64_t>(row_size(type, ne[0])) * static_cast<uint64_t>(nrows);
}

inline size_t quantize_rows(ggml_type type, const float * src, void * dst, int64_t nrows,
                            int64_t n_per_row, const float * imatrix, int nthreads) {
    if (nrows <= 0 || n_per_row <= 0) {
        throw std::runtime_error("quantize: empty tensor");
    }
    if (type_needs_imatrix(type) && !imatrix) {
        throw std::runtime_error(std::string("quantize: ") + type_name(type) + " requires an imatrix");
    }
    const size_t row_size = mmq::row_size(type, n_per_row);
    if (type == GGML_TYPE_F32) {
        const size_t n = static_cast<size_t>(nrows) * static_cast<size_t>(n_per_row) * sizeof(float);
        std::memcpy(dst, src, n);
        return n;
    }
    if (is_tq(type)) {
        // The trellis encoder: one Viterbi search per 128-weight block, rows independent.
        const int64_t blck = 128;
        if ((n_per_row % blck) != 0) {
            throw std::runtime_error(std::string("quantize: row length ") + std::to_string(n_per_row) +
                                     " is not a multiple of 128 for " + type_name(type));
        }
        if (nthreads < 1) {
            nthreads = 1;
        }
        if (static_cast<int64_t>(nthreads) > nrows) {
            nthreads = static_cast<int>(nrows);
        }
        strata::kernels::cpu::TqEncodeParams p;
        p.weighted = imatrix != nullptr;
        auto one = [&](int64_t r0, int64_t r1) {
            strata::kernels::cpu::tq_encode_row(static_cast<int>(type), src + r0 * n_per_row,
                                                imatrix ? imatrix : nullptr, (r1 - r0) * n_per_row, p,
                                                static_cast<uint8_t *>(dst) + r0 * row_size);
        };
        if (nthreads == 1) {
            one(0, nrows);
            return static_cast<size_t>(nrows) * row_size;
        }
        std::vector<std::thread> workers;
        std::vector<std::exception_ptr> errors(static_cast<size_t>(nthreads));
        const int64_t chunk = (nrows + nthreads - 1) / nthreads;
        for (int t = 0; t < nthreads; ++t) {
            const int64_t a = static_cast<int64_t>(t) * chunk;
            if (a >= nrows) {
                break;
            }
            const int64_t b = std::min(nrows, a + chunk);
            workers.emplace_back([&, t, a, b]() {
                try {
                    one(a, b);
                } catch (...) {
                    errors[static_cast<size_t>(t)] = std::current_exception();
                }
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        for (auto & e : errors) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
        return static_cast<size_t>(nrows) * row_size;
    }
    const int64_t blck = blck_size(type);
    if (blck <= 0 || (n_per_row % blck) != 0) {
        throw std::runtime_error(std::string("quantize: row length ") + std::to_string(n_per_row) +
                                 " is not a multiple of block size for " + type_name(type));
    }
    ggml_quantize_init(type);
    if (nthreads < 1) {
        nthreads = 1;
    }
    if (static_cast<int64_t>(nthreads) > nrows) {
        nthreads = static_cast<int>(nrows);
    }

    auto one = [&](int64_t row, int64_t nrow) {
        void * out = static_cast<char *>(dst) + row * row_size;
        const size_t got = ggml_quantize_chunk(type, src + row * n_per_row, out, 0, nrow, n_per_row, imatrix);
        if (got != static_cast<size_t>(nrow) * row_size) {
            throw std::runtime_error("quantize: unexpected chunk size");
        }
        if (!ggml_validate_row_data(type, out, got)) {
            throw std::runtime_error(std::string("quantize: row validation failed for ") + type_name(type));
        }
    };

    if (nthreads == 1) {
        one(0, nrows);
        return static_cast<size_t>(nrows) * row_size;
    }

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(static_cast<size_t>(nthreads));
    const int64_t chunk = (nrows + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t a = static_cast<int64_t>(t) * chunk;
        if (a >= nrows) {
            break;
        }
        const int64_t b = std::min(nrows, a + chunk);
        workers.emplace_back([&, t, a, b]() {
            try {
                one(a, b - a);
            } catch (...) {
                errors[static_cast<size_t>(t)] = std::current_exception();
            }
        });
    }
    for (auto & w : workers) {
        w.join();
    }
    for (auto & e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
    return static_cast<size_t>(nrows) * row_size;
}

}  // namespace mmq
