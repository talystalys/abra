# LegacyGpuArch.cmake
# Configures dual-architecture compilation for Pascal (sm_61) and Turing (sm_75)

set(CMAKE_CUDA_ARCHITECTURES "61;75" CACHE STRING "Target CUDA architectures")
message(STATUS "Legacy GPU Target Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Suppress deprecation warning for SM 6.1 in CUDA 12.9
string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")

# Ensure compatibility overlay is prioritized in include paths if present
set(CUDA_COMPAT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/.cuda_compat/include")
if(EXISTS "${CUDA_COMPAT_INCLUDE_DIR}")
    message(STATUS "Using CUDA compatibility overlay from: ${CUDA_COMPAT_INCLUDE_DIR}")
    include_directories(BEFORE SYSTEM "${CUDA_COMPAT_INCLUDE_DIR}")
    string(APPEND CMAKE_CUDA_FLAGS " -I${CUDA_COMPAT_INCLUDE_DIR}")
endif()
