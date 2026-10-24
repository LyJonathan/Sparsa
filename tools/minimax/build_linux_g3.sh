#!/usr/bin/env sh
# Build strata-minimax on Linux for the G3 runner (the same recipe rig-2 and the molab cloud box use).
#
#   CUDA_HOME=/usr/local/lib/python3.13/site-packages/nvidia/cu13 \
#   STRATA_GGML_DIR=$HOME/strata-work/llama.cpp STRATA_BUILD_DIR=/work/build \
#   STRATA_CUDA_ARCHS=120 sh tools/minimax/build_linux_g3.sh
#
# Differences from tools/minimax/build_linux.sh (which lives on release/*, not on this branch):
#   * it configures CMake directly, because build_linux.sh is not on feat/minimax-g3;
#   * it applies the two patches the ik formats need on GCC (tools/minimax/patches/):
#       iqk_avx2_fma.patch  - iqk_quants.cpp gets -mavx2 -mfma -mf16c, otherwise the codec
#                             stays scalar and an ik tier is ~10x slower;
#       iqk_gemm_q8k.patch  - gemm_cpu_base() projects ik types through their own q8_K codec,
#                             otherwise a batched prefill dies with "ggml-cpu has no
#                             vec_dot for type 164" (IQ2_KL) before one token runs.
#
# A CUDA toolkit installed from pip (nvidia-cu13) needs two cosmetic fixes before CMake will
# configure; the script does both and reports what it did:
#   * lib/libcudart.so and friends are only shipped with their version in the name;
#   * include/cccl is empty, and cuda_fp16.h includes <nv/target> from it.
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=${STRATA_BUILD_DIR:-$ROOT/build-g3}
ARCHS=${STRATA_CUDA_ARCHS:-"89;120"}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 8)}
CMAKE=${CMAKE:-cmake}
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
GGML=${STRATA_GGML_DIR:-}

if [ -z "$GGML" ]; then
    echo "build_linux_g3: set STRATA_GGML_DIR to a llama.cpp checkout at third_party/ggml/VERSION.txt" >&2
    exit 1
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "build_linux_g3: no nvcc under $CUDA_HOME" >&2
    exit 1
fi

for lib in libcudart libcublas libcublasLt; do
    [ -e "$CUDA_HOME/lib/$lib.so" ] || ln -sfn "$(ls "$CUDA_HOME"/lib/$lib.so.* | head -1)" "$CUDA_HOME/lib/$lib.so"
done
if [ ! -e "$CUDA_HOME/include/cccl/nv/target" ]; then
    echo "build_linux_g3: CUDA_HOME/include/cccl has no nv/target;" >&2
    echo "  link the CCCL headers in, e.g." >&2
    echo "  git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/cccl" >&2
    echo "  (sparse-checkout libcudacxx/include thrust cub) and symlink nv, cuda, thrust, cub into include/cccl" >&2
    exit 1
fi

git -C "$ROOT" apply --check tools/minimax/patches/iqk_avx2_fma.patch 2>/dev/null &&
    git -C "$ROOT" apply tools/minimax/patches/iqk_avx2_fma.patch && echo "applied iqk_avx2_fma"
git -C "$ROOT" apply --check tools/minimax/patches/iqk_gemm_q8k.patch 2>/dev/null &&
    git -C "$ROOT" apply tools/minimax/patches/iqk_gemm_q8k.patch && echo "applied iqk_gemm_q8k"

"$CMAKE" -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_Q6K_EXPERTS=ON \
    -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_NATIVE_EXPERTS=ON \
    -DSTRATA_GGML_DIR="$GGML" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"
"$CMAKE" --build "$BUILD" --target strata-minimax -j "$JOBS"
echo "built: $BUILD/strata-minimax"
