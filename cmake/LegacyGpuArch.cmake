set(CMAKE_CUDA_ARCHITECTURES "61;75" CACHE STRING "Target CUDA architectures")
message(STATUS "Legacy GPU Target Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")

set(CUDA_COMPAT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/.cuda_compat/include")
if(EXISTS "${CUDA_COMPAT_INCLUDE_DIR}")
    message(STATUS "Using CUDA compatibility overlay from: ${CUDA_COMPAT_INCLUDE_DIR}")
    include_directories(BEFORE SYSTEM "${CUDA_COMPAT_INCLUDE_DIR}")
    string(APPEND CMAKE_CUDA_FLAGS " -I${CUDA_COMPAT_INCLUDE_DIR}")
endif()
