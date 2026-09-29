cmake_minimum_required(VERSION 3.20)

project(legacy_ai LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LegacyGpuArch)

add_definitions(
    -DGGML_CUDA_USE_GRAPHS
    -DUSE_CUDA_GRAPH
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
    -DGGML_SCHED_MAX_COPIES=4
)

# Include directories
set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/llama.cpp" CACHE PATH "Pinned llama.cpp source")
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${LLAMA_SOURCE_DIR}/ggml/include
    ${LLAMA_SOURCE_DIR}/ggml/src
    ${LLAMA_SOURCE_DIR}/ggml/src/ggml-cuda
)

# Pinned llama.cpp library location
set(LLAMA_LIB_DIR "${LLAMA_SOURCE_DIR}/build-default/bin" CACHE PATH "Pinned llama.cpp libraries")
link_directories(${LLAMA_LIB_DIR})

# Common smoke kernel library (M0)
add_library(legacy_smoke STATIC
    kernels/smoke/smoke.cu
)
set_target_properties(legacy_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# 1. Device enumeration tool
add_executable(list_cuda_devices
    tools/list_cuda_devices.cu
)
target_link_libraries(list_cuda_devices PRIVATE legacy_smoke)

# 2. Correctness test suite with CPU oracle (M0)
add_executable(smoke_test
    tests/smoke_test.cu
)
target_link_libraries(smoke_test PRIVATE legacy_smoke)

# 3. Timing harness validation benchmark (M0)
add_executable(timer_smoke
    bench/micro/timer_smoke.cu
)
target_link_libraries(timer_smoke PRIVATE legacy_smoke)

# =============================================================================
# Milestone M2: Pascal DP4A INT8 Microkernel Targets
# =============================================================================
add_library(legacy_dp4a STATIC
    kernels/dp4a/dp4a_gemv.cu
)
set_target_properties(legacy_dp4a PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
# Enable ptxas verbose info to inspect register and resource usage during build
target_compile_options(legacy_dp4a PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 4. DP4A Correctness Test Suite
add_executable(dp4a_test
    tests/dp4a_test.cu
)
target_link_libraries(dp4a_test PRIVATE legacy_dp4a legacy_smoke)
target_compile_options(dp4a_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 5. DP4A Microbenchmark Harness
add_executable(dp4a_bench
    bench/micro/dp4a_bench.cu
)
target_link_libraries(dp4a_bench PRIVATE legacy_dp4a legacy_smoke)
target_compile_options(dp4a_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 6. DP4A Theoretical Occupancy Extraction Tool
add_executable(dump_m2_occupancy
    tools/dump_m2_occupancy.cu
)
target_link_libraries(dump_m2_occupancy PRIVATE legacy_dp4a legacy_smoke)
target_compile_options(dump_m2_occupancy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# =============================================================================
# Milestone M3 & M4: Pascal Q4_K x Q8_1 MMVQ & Parameterized Family Targets
# =============================================================================
add_library(legacy_q4k STATIC
    kernels/q4k/q4k_q8_mmvq.cu
    kernels/q4k/q4k_family.cu
    kernels/q4k/upstream_adapter.cu
)
set_target_properties(legacy_q4k PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(legacy_q4k PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 7. Q4_K x Q8_1 Correctness Test Suite (links exact pinned libggml-cuda and libggml-base)
add_executable(q4k_q8_test
    tests/q4k_q8_test.cu
)
target_link_libraries(q4k_q8_test PRIVATE legacy_q4k legacy_smoke ggml-cuda ggml-base ggml)
set_target_properties(q4k_q8_test PROPERTIES
    BUILD_RPATH "${LLAMA_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_LIB_DIR}"
)
target_compile_options(q4k_q8_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 8. Q4_K x Q8_1 Microbenchmark Harness (links exact pinned libggml-cuda and libggml-base)
add_executable(q4k_q8_bench
    bench/micro/q4k_q8_bench.cu
)
target_link_libraries(q4k_q8_bench PRIVATE legacy_q4k legacy_smoke ggml-cuda ggml-base ggml)
set_target_properties(q4k_q8_bench PROPERTIES
    BUILD_RPATH "${LLAMA_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_LIB_DIR}"
)
target_compile_options(q4k_q8_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 9. Q4_K x Q8_1 M3 Theoretical Occupancy Extraction Tool
add_executable(dump_m3_occupancy
    tools/dump_m3_occupancy.cu
)
target_link_libraries(dump_m3_occupancy PRIVATE legacy_q4k legacy_smoke ggml-cuda ggml-base ggml)
set_target_properties(dump_m3_occupancy PROPERTIES
    BUILD_RPATH "${LLAMA_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_LIB_DIR}"
)
target_compile_options(dump_m3_occupancy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 10. Q4_K x Q8_1 M4 Parameterized Family Occupancy Extraction Tool
add_executable(dump_m4_occupancy
    tools/dump_m4_occupancy.cu
)
target_link_libraries(dump_m4_occupancy PRIVATE legacy_q4k legacy_smoke ggml-cuda ggml-base ggml)
set_target_properties(dump_m4_occupancy PROPERTIES
    BUILD_RPATH "${LLAMA_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_LIB_DIR}"
)

# End-to-end runtime validation for the hardware port (independent of timing).
add_executable(runtime_logits tests/runtime_logits.cpp)
target_include_directories(runtime_logits PRIVATE ${LLAMA_SOURCE_DIR}/include)
target_link_libraries(runtime_logits PRIVATE llama ggml ggml-base)
set_target_properties(runtime_logits PROPERTIES BUILD_RPATH "${LLAMA_LIB_DIR}")

add_executable(lookup_decode tests/lookup_decode.cpp)
target_include_directories(lookup_decode PRIVATE ${LLAMA_SOURCE_DIR}/include)
target_link_libraries(lookup_decode PRIVATE llama ggml ggml-base)
set_target_properties(lookup_decode PROPERTIES BUILD_RPATH "${LLAMA_LIB_DIR}")

add_executable(runtime_kernel_test tests/runtime_kernel_test.cu)
target_link_libraries(runtime_kernel_test PRIVATE ggml-cuda ggml-base ggml)
set_target_properties(runtime_kernel_test PROPERTIES BUILD_RPATH "${LLAMA_LIB_DIR}")
add_executable(attention_kernel_test tests/attention_kernel_test.cu)
target_link_libraries(attention_kernel_test PRIVATE ggml-cuda ggml-base ggml)
set_target_properties(attention_kernel_test PROPERTIES BUILD_RPATH "${LLAMA_LIB_DIR}")
add_executable(value_kernel_test tests/value_kernel_test.cu)
target_link_libraries(value_kernel_test PRIVATE ggml-cuda ggml-base ggml)
set_target_properties(value_kernel_test PROPERTIES BUILD_RPATH "${LLAMA_LIB_DIR}")
target_compile_options(dump_m4_occupancy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
