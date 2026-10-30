#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace abra {
namespace dp4a {

enum class KernelVariant {
    SCALAR = 0,
    DP4A_BASIC = 1,
    DP4A_OPT = 2,
    DP4A_SHARED = 3
};

const char* get_variant_name(KernelVariant variant);

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

struct KernelOccupancyInfo {
    std::string variant_name;
    int block_dim_x;
    int block_dim_y;
    int threads_per_block;
    int registers_per_thread;
    size_t static_shared_bytes;
    int active_blocks_per_sm;
    int resident_threads_per_sm;
    int resident_warps_per_sm;
    int max_warps_per_sm;
    double theoretical_occupancy_pct;
};

KernelOccupancyInfo get_kernel_occupancy_info(KernelVariant variant, int device_id = 0);

}
}
