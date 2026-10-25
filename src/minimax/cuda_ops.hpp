// src/minimax/cuda_ops.hpp - device launches for the MiniMax forward pass. Host-only types; the .cu file owns CUDA.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::minimax::cu {

struct MemInfo {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

void init();
MemInfo mem();
void* alloc(size_t bytes);
void dealloc(void* p);
void copy_h2d(void* dst, const void* src, size_t bytes);
void copy_d2h(void* dst, const void* src, size_t bytes);
void copy_d2h_async(void* dst, const void* src, size_t bytes, void* stream);
void copy_d2d(void* dst, const void* src, size_t bytes);

// ---- P4: more than one device.  Every call above acts on the thread's CURRENT device; `set_device` is the only
// thing that changes it.  The engine owns the policy (which layer runs where); these are the primitives.
int device_count();
void set_device(int dev);
int current_device();
MemInfo mem_on(int dev);
void* alloc_on(int dev, size_t bytes);

/// Page-locked host memory for the KV cache and the cross-GPU hop.  Throws when the driver refuses it.
void* host_alloc(size_t bytes);
void host_free(void* p);

// ---- P4: MSA attention on the device (fp16 KV, fp32 scores).
/// bs[h*nb + b] = max over t in block b with t < ntok of dot(q_h, ik[t]).  `ik` is fp16 [ctx][dim].
void idx_block_scores(const void* ik, int64_t ntok, const float* q, int heads, int dim, int blk, int nb, float* bs,
                      void* stream);
/// P10: the indexer's top-k block selection on the device, from `bs` [heads][nb].  Writes the selected token positions
/// of every head into `slots` [heads][maxslot] in ascending position order and their count into `counts` [heads].
/// Same ranking as the host `select_slots` (score desc, block index asc); `local` blocks ending at the query block are
/// forced in, exactly as the host path forces them to +inf.  No host round trip.
void idx_select(const float* bs, int heads, int nb, int top_k, int blk, int64_t pos, int local, int maxslot,
                int32_t* slots, int32_t* counts, void* stream);
/// Gather the selected slots of every KV head from the device fp16 cache [ctx][hkv][dim] into a contiguous
/// [hkv][maxslot][dim] staging buffer.  `slot_pos`/`counts` are device arrays [hkv][maxslot] / [hkv].
void kv_gather(const void* k, const void* v, int hkv, int dim, const int32_t* slot_pos, const int32_t* counts,
               int maxslot, void* kdst, void* vdst, void* stream);
/// out[qh*dim..] = softmax(q_h . K[hkv] / sqrt(dim)) . V[hkv] over the gathered slots.
void msa_attn(const void* kstage, const void* vstage, const float* q, float* out, int nheads, int q_per_kv,
              int maxslot, const int32_t* counts, int dim, void* stream);

/// y[rows] = W[rows, cols] * x[cols], all device pointers, row-major W. Synchronizes before return.
void gemv(const float* W, const float* x, float* y, int rows, int cols);

/// Dequantize `rows` rows of a GGUF matrix starting at `row0` into device fp32. Synchronizes.
void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out);

void sync();

/// A non-blocking side stream for the adaptive VRAM swaps.
void* stream_create();
/// A blocking stream: ordered against the legacy default stream without events (the decode norm+rope chain).
void* stream_create_blocking();
/// The legacy default stream's non-null handle.  For a kernel API that rejects nullptr (native MMVQ) but must stay
/// ordered with the default-stream kernels around it (`gemv_dev_in`: MSA wrote the activation on the default stream).
void* stream_legacy();
/// CUDA graph capture for the per-token GPU block.  Begin/end on a dedicated stream; `end` returns an
/// instantiated graph executable (or null when capture is unsupported here).  Capture does NOT execute the work:
/// the caller launches once after ending, then replays per token.
void* graph_capture_begin(void* stream);
void* graph_capture_end(void* stream);
void graph_launch(void* graph, void* stream);
void graph_destroy(void* graph);
void stream_destroy(void* stream);
void copy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
void stream_sync(void* stream);
/// CUDA events for the staged expert ring (copy stream -> compute stream ordering).
void* event_create();
void event_destroy(void* event);
void event_record(void* event, void* stream);
void event_sync(void* event);
bool event_ready(void* event);
/// P8: a timing event pair.  `event_create_timed` makes an event `cudaEventElapsedTime` accepts;
/// `event_elapsed_ms` is the device-busy time between two records (0 when either is null).
void* event_create_timed();
double event_elapsed_ms(void* a, void* b);

/// Grouped routed-expert MLP for F32/F16/BF16 experts, all on the device.  `n_entries` selected experts:
/// gate[e]/up[e] are `n_ff` rows of `n_embd`, down[e] is `n_embd` rows of `n_ff`, GGUF row-major bytes.
/// `gu_row`/`d_row` are their ggml row sizes.  `h` is scratch `n_entries * n_ff` floats, `out` is
/// `n_entries * n_embd` floats.  swigluoai is fused into the gate/up pass.  Synchronizes.
void moe_expert_f16(int type, const void* const* gate, const void* const* up, const void* const* down,
                    int n_entries, int n_embd, int n_ff, size_t gu_row, size_t d_row, const float* x, float* h,
                    float* out, float alpha, float limit);

// ============================================================================================= P5a: batched prefill
//
// The prompt path.  Every kernel here is asynchronous on the stream it is given (nullptr = the default stream);
// the engine synchronizes once per stage.  Activations are fp16 between the batched GEMMs (tensor cores) and fp32
// for the combine; the arithmetic is the same as the per-token path, only the order of the reductions changes.

/// y[i] = (half) x[i], n values.  Device pointers.
void f32_to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
/// P5c: per-head RMSNorm + NEOX partial RoPE + (optional) fp32/fp16 store, one block per row.
/// `x` is [rows][dim]; `heads_per_tok` rows belong to one token and take `pos[token]`.  `o32` and/or `o16`
/// may be null and may equal `x` (the rotation is in-place safe: a pair writes only its two inputs).
/// `cos_tab`/`sin_tab` are [max_pos][n_rot/2] from `strata::kernels::build_rope_table`.
void head_norm_rope(const float* x, const float* gamma, float* o32, uint16_t* o16, int64_t rows, int heads_per_tok,
                    int dim, int n_rot, const float* cos_tab, const float* sin_tab, const int32_t* pos, float eps,
                    void* stream);
/// y[i] *= s, n values (the MM-NVFP4 per-expert global divisor on a batched expert result).
void scale_rows(float* y, int64_t n, float s, void* stream);
/// y[r][c] = (half) x[rows[r]][c], n_rows x cols, row-major.
void gather_rows_f16(const float* x, const int32_t* rows, int64_t n_rows, int64_t cols, uint16_t* y, void* stream);
/// y[r][c] = x[rows[r]][c] (fp32).
void gather_rows_f32(const float* x, const int32_t* rows, int64_t n_rows, int64_t cols, float* y, void* stream);
/// Dequantize `n` contiguous values of `ggml_type` to fp16 (F32/F16/BF16 handled here, the GGUF quants through the
/// kernels' dequantizer).  n must be a multiple of the type's block.
void dequant_to_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
/// The same to fp32 (the parity-first batched path: same weights as the per-token GPU GEMV, no fp16 rounding).
void dequant_to_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Y[T, N] (fp32) = X[T, K] . W[N, K]^T with both operands fp32 (cuBLAS SGEMM).
void gemm_f32(const float* X, const float* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
              void* stream);
/// Y[T, N] (fp32, row stride ldy) = X[T, K] . W[N, K]^T with both operands fp16 row-major.  `beta` 0 replaces,
/// 1 adds.  cuBLAS FP16 tensor cores with an fp32 accumulator.  Initializes the handle on first use (per device).
void gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              float beta, void* stream);
/// h[t][r] = swigluoai(gate[t][r], up[t][r]) in fp16 (rows x n_ff), the down matrix's activation.
void swiglu_rows_f16(const float* g, const float* u, uint16_t* h, int64_t rows, int64_t n_ff, float alpha,
                     float limit, void* stream);
/// The same in fp32 (the q8_1 activation of the int8 expert kernels).
void swiglu_rows_f32(const float* g, const float* u, float* h, int64_t rows, int64_t n_ff, float alpha, float limit,
                     void* stream);
/// out[t][c] += ent_w[r] * Y[r][c], where token t = ent_tok[r] (n_rows entries, row back to back).  Groups are
/// launched one after another on the stream, so the plain `+=` is ordered; `zero` clears out first.
void moe_combine_rows(const float* Y, const int32_t* ent_tok, const float* ent_w, int64_t n_rows, int64_t n_embd,
                      float* out, int64_t n_tok, bool zero, void* stream);
/// Indexer block maxima for a whole prompt chunk: bs[t][h][b] = max_{p in block b, p <= pos0+t} q[t][h] . ik[p].
/// `q` is fp16 [n_tok][heads][dim], `ik` fp16 [ctx][dim].
void idx_block_scores_batch(const void* ik, const uint16_t* q, int heads, int dim, int blk, int64_t nb,
                            int64_t n_tok, int64_t pos0, float* bs, void* stream);
/// Causal MSA attention for a whole chunk, reading the fp16 KV cache directly (no gather).
/// `q` fp16 [n_tok][heads][dim]; `blk` int32 [n_tok][idx_heads][blocks_per_q] and `nblk` [n_tok][idx_heads] are the
/// selected block ids (ascending).  `dense_all` != 0 ignores them and attends every position <= pos (no indexer).
/// `out` fp32 [n_tok][heads][dim].
void msa_attn_batch(const void* k, const void* v, const uint16_t* q, const int32_t* blk, const int32_t* nblk,
                    int heads, int q_per_kv, int q_per_idx, int dim, int blk_size, int dense_all, int64_t pos0,
                    int64_t n_tok, int64_t max_blk, float* out, void* stream);

}  // namespace strata::minimax::cu
