cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(strata_mm_quant LANGUAGES C CXX)
else()
    enable_language(C)
endif()

# Standalone quantizer. third_party/ggml in this repo is only the common headers
# Strata's native expert kernels include; the IQ quantizer and GGML_TYPE_NVFP4
# live in llama.cpp's ggml, which is also the ggml the patched loader was built
# with. _baseline/ is gitignored, so the path is explicit.

set(STRATA_LLAMA_CPP_DIR "" CACHE PATH "llama.cpp tree (ggml with IQ + NVFP4). The M3.1 baseline is the one to use.")
if(NOT STRATA_LLAMA_CPP_DIR AND DEFINED ENV{STRATA_LLAMA_CPP_DIR})
    set(STRATA_LLAMA_CPP_DIR "$ENV{STRATA_LLAMA_CPP_DIR}" CACHE PATH "" FORCE)
endif()
if(NOT STRATA_LLAMA_CPP_DIR)
    foreach(cand
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../_baseline/llama.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../Strata-MiniMax/_baseline/llama.cpp")
        if(EXISTS "${cand}/ggml/include/ggml.h")
            get_filename_component(STRATA_LLAMA_CPP_DIR "${cand}" ABSOLUTE)
            break()
        endif()
    endforeach()
endif()
if(NOT EXISTS "${STRATA_LLAMA_CPP_DIR}/ggml/include/ggml.h")
    message(FATAL_ERROR "STRATA_LLAMA_CPP_DIR does not contain ggml/include/ggml.h (set -DSTRATA_LLAMA_CPP_DIR=...)")
endif()
message(STATUS "strata-mm-quant ggml: ${STRATA_LLAMA_CPP_DIR}/ggml")

if(NOT TARGET ggml-base)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    foreach(_opt GGML_CUDA GGML_METAL GGML_BLAS GGML_RPC GGML_VULKAN GGML_HIP GGML_SYCL
                 GGML_OPENMP GGML_CPU GGML_BUILD_TESTS GGML_BUILD_EXAMPLES GGML_BACKEND_DL
                 GGML_CCACHE GGML_OPENCL GGML_CANN GGML_MUSA)
        set(${_opt} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
    add_subdirectory("${STRATA_LLAMA_CPP_DIR}/ggml" "${CMAKE_BINARY_DIR}/mm-quant-ggml" EXCLUDE_FROM_ALL)
endif()

# The formats ported from ik_llama.cpp (IQ2_KT, IQ3_KT) have their own codec, in the engine's tree: it
# depends on nothing but the standard library, so the tool and the engine share one implementation.
get_filename_component(STRATA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(STRATA_IKQ_SRC "${STRATA_REPO_ROOT}/src/kernels/cpu/iqk_quants.cpp")

add_executable(strata-mm-quant
    src/main.cpp
    src/convert.cpp
    ${STRATA_IKQ_SRC})
target_include_directories(strata-mm-quant PRIVATE src "${STRATA_REPO_ROOT}/include" "${STRATA_LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(strata-mm-quant PRIVATE ggml-base)
if(MSVC)
    # the trellis quantizer is AVX2 code, like ik's own (which refuses to run without it)
    set_source_files_properties(${STRATA_IKQ_SRC} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    set_source_files_properties(${STRATA_IKQ_SRC} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
target_compile_features(strata-mm-quant PRIVATE cxx_std_20)
if(MSVC)
    target_compile_definitions(strata-mm-quant PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(strata-mm-quant PRIVATE -Wall -Wextra -Wno-unused-function)
    find_package(Threads REQUIRED)
    target_link_libraries(strata-mm-quant PRIVATE Threads::Threads)
endif()

add_executable(strata-mm-quant-ikq-test src/test_ikq.cpp ${STRATA_IKQ_SRC})
target_include_directories(strata-mm-quant-ikq-test PRIVATE src "${STRATA_REPO_ROOT}/include")
target_compile_features(strata-mm-quant-ikq-test PRIVATE cxx_std_20)
if(MSVC)
    target_compile_definitions(strata-mm-quant-ikq-test PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    set_source_files_properties(${STRATA_IKQ_SRC} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    find_package(Threads REQUIRED)
    target_link_libraries(strata-mm-quant-ikq-test PRIVATE Threads::Threads)
    set_source_files_properties(${STRATA_IKQ_SRC} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

add_executable(strata-mm-quant-dequant-test src/test_dequant.cpp)
target_include_directories(strata-mm-quant-dequant-test PRIVATE
    src
    "${STRATA_LLAMA_CPP_DIR}/ggml/include"
    "${STRATA_LLAMA_CPP_DIR}/ggml/src")
target_link_libraries(strata-mm-quant-dequant-test PRIVATE ggml-base)
target_compile_features(strata-mm-quant-dequant-test PRIVATE cxx_std_20)
if(MSVC)
    target_compile_definitions(strata-mm-quant-dequant-test PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(strata-mm-quant-dequant-test PRIVATE -Wall -Wextra -Wno-unused-function)
    target_link_libraries(strata-mm-quant-dequant-test PRIVATE Threads::Threads)
endif()

add_custom_command(TARGET strata-mm-quant POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/recipes"
            "$<TARGET_FILE_DIR:strata-mm-quant>/recipes"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/mmproj.py"
            "$<TARGET_FILE_DIR:strata-mm-quant>/mmproj.py")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_test(NAME dequant_test COMMAND strata-mm-quant-dequant-test)
    add_test(NAME ikq_test COMMAND strata-mm-quant-ikq-test)
endif()
