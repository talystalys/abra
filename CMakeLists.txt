cmake_minimum_required(VERSION 3.20)

project(legacy_ai LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LegacyGpuArch)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# Common smoke kernel library
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

# 2. Correctness test suite with CPU oracle
add_executable(smoke_test
    tests/smoke_test.cu
)
target_link_libraries(smoke_test PRIVATE legacy_smoke)

# 3. Timing harness validation benchmark
add_executable(timer_smoke
    bench/micro/timer_smoke.cu
)
target_link_libraries(timer_smoke PRIVATE legacy_smoke)
