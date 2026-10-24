// include/strata/minimax/vision.hpp - the MiniMax-M3.1-Flash vision tower (P6).
//
// An `mmproj` GGUF (llama.cpp's `clip` architecture, `clip.projector_type = minimax_m3`) holds the ViT, the
// 1280->6144 GELU projector and the 2x2 patch-merge MLP.  encode() turns an RGB image into the merged image
// embeddings the language model consumes: one row per image token, each `n_embd` (6144) wide.
//
// Arithmetic follows transformers' MiniMaxM3VLVisionModel with the temporal axis at 0:
//   Conv3d(2,14,14) patch embed (the two temporal slices are summed: an image is one frame duplicated),
//   32 pre-LN ViT layers (LayerNorm eps 1e-5, exact-erf GELU), 3-axis partial RoPE (H and W live; T is
//   identity), then linear_1 -> GELU -> linear_2 -> 2x2 merge -> merge_linear_1 -> GELU -> merge_linear_2.
// The GGUF stores q/k in llama.cpp's [T Tdup H Hdup W Wdup pad] layout; the H rotation is the pair
// (26+i, 39+i) and W is (52+i, 65+i), which is the reference's 3D RoPE with t = 0.
//
// F16 and F32 tensors are accepted (the converter writes both).  Everything runs fp32 on the CPU with the
// caller's thread count; no CUDA context is created.  Credit: the mmproj layout and the vision graph follow
// llama.cpp's tools/mtmd (MIT), conversion/minimax.py and tools/mtmd/models/minimax-m3.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::minimax {

struct VisionConfig {
    int64_t hidden = 1280;
    int64_t ff = 5120;
    int64_t layers = 32;
    int64_t heads = 16;
    int64_t patch = 14;
    int64_t temporal = 2;
    int64_t merge = 2;
    int64_t text_hidden = 6144;
    int64_t projector_hidden = 6144;
    int64_t image_size = 672;
    int64_t max_pixels = 451584;   // 672 * 672, as preprocessor_config.json
    double eps = 1e-5;
    double rope_theta = 10000.0;
    std::vector<float> mean{0.48145466f, 0.4578275f, 0.40821073f};
    std::vector<float> std{0.26862954f, 0.26130258f, 0.27577711f};
};

struct ImageEmbeddings {
    int64_t tokens = 0;     ///< merged image tokens (e.g. 576 for 672x672)
    int64_t dim = 0;        ///< text hidden size (6144)
    int64_t width = 0;      ///< resized width
    int64_t height = 0;     ///< resized height
    std::vector<float> data;///< [tokens, dim], row-major, in merge-block order
};

/// smart_resize() from the checkpoint's image_processor.py: round both sides to a multiple of 28, then keep the
/// pixel count inside [3136, max_pixels] without changing the aspect ratio.
void smart_resize(int64_t height, int64_t width, int64_t& out_h, int64_t& out_w,
                  int64_t factor = 28, int64_t min_pixels = 3136, int64_t max_pixels = 451584);

class Vision {
public:
    /// Open `mmproj_path` (a llama.cpp clip mmproj) and expand its tensors to fp32.  Throws std::runtime_error
    /// naming the first missing/malformed tensor.  `threads` 0 = hardware_concurrency.
    Vision(const std::string& mmproj_path, int threads = 0);
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    const VisionConfig& config() const;
    const std::string& path() const;
    /// The number of image tokens an image of this pixel size becomes.
    int64_t tokens_for(int64_t width, int64_t height) const;
    /// Decode a JPEG/PNG/BMP/GIF (stb_image) and encode it.  `max_tokens` > 0 refuses a result above it.
    ImageEmbeddings encode_file(const std::string& path, int64_t max_tokens = 0) const;
    /// Encode 8-bit RGB pixels (3 bytes per pixel, row-major).
    ImageEmbeddings encode_rgb(const uint8_t* rgb, int64_t width, int64_t height,
                               int64_t max_tokens = 0) const;
    /// smart_resize + bicubic + normalize + merge-block patchify.  `pixels` is [n_patch, 3*2*14*14] in the
    /// checkpoint's pixel_values order (a duplicated temporal frame), i.e. exactly what the HF processor makes.
    void preprocess_rgb(const uint8_t* rgb, int64_t width, int64_t height, std::vector<float>& pixels,
                        int64_t& n_patch, int64_t& resized_w, int64_t& resized_h) const;
    void preprocess_file(const std::string& path, std::vector<float>& pixels, int64_t& n_patch,
                         int64_t& resized_w, int64_t& resized_h) const;
    /// Encode already-preprocessed pixel values (the checkpoint's [n_patch, 3*2*14*14] fp32 tensor).  Used by
    /// the parity harness to separate preprocessing from the tower.
    ImageEmbeddings encode_pixels(const float* pixels, int64_t n_patch, int64_t width, int64_t height) const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace strata::minimax
