cmake_minimum_required(VERSION 3.20)

project(abra LANGUAGES CXX CUDA)

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

set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/llama.cpp" CACHE PATH "Pinned llama.cpp source")
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${LLAMA_SOURCE_DIR}/ggml/include
    ${LLAMA_SOURCE_DIR}/ggml/src
    ${LLAMA_SOURCE_DIR}/ggml/src/ggml-cuda
)

set(LLAMA_LIB_DIR "${LLAMA_SOURCE_DIR}/build-default/bin" CACHE PATH "Pinned llama.cpp libraries")
link_directories(${LLAMA_LIB_DIR})

add_library(legacy_smoke STATIC
    kernels/smoke/smoke.cu
)
set_target_properties(legacy_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

add_executable(list_cuda_devices
    tools/list_cuda_devices.cu
)
target_link_libraries(list_cuda_devices PRIVATE legacy_smoke)

add_executable(smoke_test
    tests/smoke_test.cu
)
target_link_libraries(smoke_test PRIVATE legacy_smoke)

add_executable(timer_smoke
    bench/micro/timer_smoke.cu
)
target_link_libraries(timer_smoke PRIVATE legacy_smoke)

add_library(legacy_dp4a STATIC
    kernels/dp4a/dp4a_gemv.cu
)
set_target_properties(legacy_dp4a PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(legacy_dp4a PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

add_executable(dp4a_test
    tests/dp4a_test.cu
)
target_link_libraries(dp4a_test PRIVATE legacy_dp4a legacy_smoke)
target_compile_options(dp4a_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

add_executable(dp4a_bench
    bench/micro/dp4a_bench.cu
)
target_link_libraries(dp4a_bench PRIVATE legacy_dp4a legacy_smoke)
target_compile_options(dp4a_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

add_executable(dump_m2_occupancy
    tools/dump_m2_occupancy.cu
)
target_link_libraries(dump_m2_occupancy PRIVATE legacy_dp4a legacy_smoke)
target_compile_options(dump_m2_occupancy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

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

add_executable(dump_m4_occupancy
    tools/dump_m4_occupancy.cu
)
target_link_libraries(dump_m4_occupancy PRIVATE legacy_q4k legacy_smoke ggml-cuda ggml-base ggml)
set_target_properties(dump_m4_occupancy PROPERTIES
    BUILD_RPATH "${LLAMA_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_LIB_DIR}"
)

add_executable(runtime_logits tests/runtime_logits.cpp)
target_include_directories(runtime_logits PRIVATE ${LLAMA_SOURCE_DIR}/include)
target_link_libraries(runtime_logits PRIVATE llama ggml ggml-base ${CMAKE_DL_LIBS})
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
option(ABRA_BUILD_Q8_REUSE_TEST "Build tests requiring the activation reuse runtime patch" OFF)
if(ABRA_BUILD_Q8_REUSE_TEST)
    set(REUSE_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/.deps/llama.cpp-port")
    add_executable(q8_reuse_test tests/q8_reuse_test.cu)
    target_include_directories(q8_reuse_test BEFORE PRIVATE "${REUSE_RUNTIME}/ggml/src/ggml-cuda")
    target_link_libraries(q8_reuse_test PRIVATE "${REUSE_RUNTIME}/build-port/bin/libggml-cuda.so" ggml-base ggml)
    set_target_properties(q8_reuse_test PROPERTIES BUILD_RPATH "${REUSE_RUNTIME}/build-port/bin;${LLAMA_LIB_DIR}")
endif()
target_compile_options(dump_m4_occupancy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
