#!/usr/bin/env sh
# Build strata-minimax on Linux (CUDA 12.8 or newer).
#
#   tools/minimax/build_linux.sh                     # configure + build, every listed architecture
#   STRATA_CUDA_ARCHS=120 tools/minimax/build_linux.sh   # only the RTX 50 series (much faster)
#   JOBS=16 STRATA_BUILD_DIR=build-mm tools/minimax/build_linux.sh
#
# CUDA 13 dropped sm_70 and older, so the default range starts at 75 (Turing).  A card of the RTX 50
# series needs CUDA 12.8+ for sm_120.  STRATA_GGML_DIR names a llama.cpp checkout at the pinned
# commit (third_party/ggml/VERSION.txt); without it CMake fetches that commit (network needed).
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=${STRATA_BUILD_DIR:-$ROOT/build}
ARCHS=${STRATA_CUDA_ARCHS:-"75;80;86;89;120"}
CMAKE=${CMAKE:-cmake}
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
JOBS=${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}

if ! command -v "$CMAKE" >/dev/null 2>&1; then
    echo "build_linux: cmake is not on PATH (python -m pip install cmake ninja, or your package manager)" >&2
    exit 1
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "build_linux: no nvcc under $CUDA_HOME; set CUDA_HOME to the toolkit (12.8+ for the RTX 50 series)" >&2
    exit 1
fi

set -- -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_Q6K_EXPERTS=ON \
    -DSTRATA_MMQ_KQUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"
if [ -n "${STRATA_GGML_DIR:-}" ]; then
    set -- "$@" -DSTRATA_GGML_DIR="$STRATA_GGML_DIR"
fi

"$CMAKE" "$@"
"$CMAKE" --build "$BUILD" --target strata-minimax -j "$JOBS"
echo "built: $BUILD/strata-minimax"
