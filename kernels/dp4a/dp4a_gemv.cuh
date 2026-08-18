#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace legacy_ai {
namespace dp4a {

// Kernel variant identifiers
enum class KernelVariant {
    SCALAR = 0,     // Variant A: Scalar INT8 multiply + INT32 accumulate (Non-DP4A baseline)
    DP4A_BASIC = 1, // Variant B: Basic explicit DP4A (1 int32 = 4 int8 per thread per step)
    DP4A_OPT = 2,   // Variant C: Optimized DP4A (vectorized 128-bit int4 loads + multi-warp block)
    DP4A_SHARED = 3 // Variant D: Shared-memory x-staged DP4A (block-cooperative x staging)
};

const char* get_variant_name(KernelVariant variant);

// Host dispatchers for GEMV: y[N] = W[N, K] * x[K]
// W: [N, K] row-major signed int8
// x: [K] contiguous signed int8
// y: [N] contiguous signed int32

cudaError_t launch_gemv_scalar(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream = 0
);

cudaError_t launch_gemv_dp4a_basic(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream = 0
);

cudaError_t launch_gemv_dp4a_opt(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream = 0
);

cudaError_t launch_gemv_dp4a_shared(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream = 0
);

cudaError_t launch_gemv(
    KernelVariant variant,
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream = 0
);

} // namespace dp4a
} // namespace legacy_ai
