#include "kernels/dp4a/dp4a_gemv.cuh"
#include <cstdio>
#include <algorithm>

namespace abra {
namespace dp4a {

const char* get_variant_name(KernelVariant variant) {
    switch (variant) {
        case KernelVariant::SCALAR:
            return "Variant A: Scalar INT8 Baseline";
        case KernelVariant::DP4A_BASIC:
            return "Variant B: Basic Explicit DP4A";
        case KernelVariant::DP4A_OPT:
            return "Variant C: Optimized DP4A (Vectorized 128-bit Loads)";
        case KernelVariant::DP4A_SHARED:
            return "Variant D: Shared-Memory x-Staged DP4A";
        default:
            return "Unknown Variant";
    }
}

__global__ void gemv_scalar_kernel(
    const int8_t* __restrict__ W,
    const int8_t* __restrict__ x,
    int32_t* __restrict__ y,
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    int lane_id = threadIdx.x;
    int row = warp_id;

    if (row >= N) return;

    const int8_t* row_W = W + (size_t)row * K;
    int32_t acc = 0;

    for (int k = lane_id; k < K; k += 32) {
        int32_t w_val = static_cast<int32_t>(row_W[k]);
        int32_t x_val = static_cast<int32_t>(x[k]);
        int32_t prod;
        asm("mul.lo.s32 %0, %1, %2;" : "=r"(prod) : "r"(w_val), "r"(x_val));
        acc += prod;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

__global__ void gemv_dp4a_basic_kernel(
    const int8_t* __restrict__ W,
    const int8_t* __restrict__ x,
    int32_t* __restrict__ y,
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    int lane_id = threadIdx.x;
    int row = warp_id;

    if (row >= N) return;

    const int8_t* row_W = W + (size_t)row * K;
    int32_t acc = 0;

    uintptr_t w_addr = reinterpret_cast<uintptr_t>(row_W);
    uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);

    if ((w_addr & 3) == 0 && (x_addr & 3) == 0) {
        int K_vec = K / 4;
        const int32_t* W_vec = reinterpret_cast<const int32_t*>(row_W);
        const int32_t* x_vec = reinterpret_cast<const int32_t*>(x);

        for (int k4 = lane_id; k4 < K_vec; k4 += 32) {
            int32_t w4 = W_vec[k4];
            int32_t x4 = x_vec[k4];
            acc = __dp4a(w4, x4, acc);
        }

        int tail_start = K_vec * 4;
        if (lane_id == 0) {
            for (int k = tail_start; k < K; ++k) {
                acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(x[k]);
            }
        }
    } else {

        for (int k = lane_id; k < K; k += 32) {
            acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(x[k]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

__global__ void gemv_dp4a_opt_kernel(
    const int8_t* __restrict__ W,
    const int8_t* __restrict__ x,
    int32_t* __restrict__ y,
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    int lane_id = threadIdx.x;
    int row = warp_id;

    if (row >= N) return;

    const int8_t* row_W = W + (size_t)row * K;
    int32_t acc = 0;

    uintptr_t w_addr = reinterpret_cast<uintptr_t>(row_W);
    uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);

    if ((w_addr & 15) == 0 && (x_addr & 15) == 0) {

        int K_vec16 = K / 16;
        const int4* W_vec16 = reinterpret_cast<const int4*>(row_W);
        const int4* x_vec16 = reinterpret_cast<const int4*>(x);

        #pragma unroll 4
        for (int k16 = lane_id; k16 < K_vec16; k16 += 32) {
            int4 w4 = W_vec16[k16];
            int4 x4 = x_vec16[k16];
            acc = __dp4a(w4.x, x4.x, acc);
            acc = __dp4a(w4.y, x4.y, acc);
            acc = __dp4a(w4.z, x4.z, acc);
            acc = __dp4a(w4.w, x4.w, acc);
        }

        int k_rem_start = K_vec16 * 16;
        int K_vec4 = K / 4;
        const int32_t* W_vec4 = reinterpret_cast<const int32_t*>(row_W);
        const int32_t* x_vec4 = reinterpret_cast<const int32_t*>(x);
        for (int k4 = (k_rem_start / 4) + lane_id; k4 < K_vec4; k4 += 32) {
            int32_t w4 = W_vec4[k4];
            int32_t x4 = x_vec4[k4];
            acc = __dp4a(w4, x4, acc);
        }

        int tail_start = K_vec4 * 4;
        if (lane_id == 0) {
            for (int k = tail_start; k < K; ++k) {
                acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(x[k]);
            }
        }
    } else if ((w_addr & 3) == 0 && (x_addr & 3) == 0) {

        int K_vec4 = K / 4;
        const int32_t* W_vec4 = reinterpret_cast<const int32_t*>(row_W);
        const int32_t* x_vec4 = reinterpret_cast<const int32_t*>(x);
        for (int k4 = lane_id; k4 < K_vec4; k4 += 32) {
            int32_t w4 = W_vec4[k4];
            int32_t x4 = x_vec4[k4];
            acc = __dp4a(w4, x4, acc);
        }
        int tail_start = K_vec4 * 4;
        if (lane_id == 0) {
            for (int k = tail_start; k < K; ++k) {
                acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(x[k]);
            }
        }
    } else {

        for (int k = lane_id; k < K; k += 32) {
            acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(x[k]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

constexpr int SHARED_WARPS_PER_BLOCK = 8;
constexpr int SHARED_TILE_K = 2048;

__global__ void gemv_dp4a_shared_kernel(
    const int8_t* __restrict__ W,
    const int8_t* __restrict__ x,
    int32_t* __restrict__ y,
    int N, int K
) {
    __shared__ alignas(16) int8_t s_x[SHARED_TILE_K];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int total_threads = SHARED_WARPS_PER_BLOCK * 32;
    int row = blockIdx.x * SHARED_WARPS_PER_BLOCK + threadIdx.y;
    int lane_id = threadIdx.x;

    int32_t acc = 0;

    for (int k_tile = 0; k_tile < K; k_tile += SHARED_TILE_K) {
        int cur_tile_size = min(SHARED_TILE_K, K - k_tile);

        uintptr_t x_tile_addr = reinterpret_cast<uintptr_t>(x + k_tile);
        if ((x_tile_addr & 3) == 0) {
            int vec4_loads = cur_tile_size / 4;
            const int32_t* x_tile_vec4 = reinterpret_cast<const int32_t*>(x + k_tile);
            int32_t* s_x_vec4 = reinterpret_cast<int32_t*>(s_x);

            for (int i4 = tid; i4 < vec4_loads; i4 += total_threads) {
                s_x_vec4[i4] = x_tile_vec4[i4];
            }
            int rem_bytes_start = vec4_loads * 4;
            for (int rem_b = rem_bytes_start + tid; rem_b < cur_tile_size; rem_b += total_threads) {
                s_x[rem_b] = x[k_tile + rem_b];
            }
        } else {
            for (int b = tid; b < cur_tile_size; b += total_threads) {
                s_x[b] = x[k_tile + b];
            }
        }
        __syncthreads();

        if (row < N) {
            const int8_t* row_W = W + (size_t)row * K + k_tile;
            uintptr_t w_addr = reinterpret_cast<uintptr_t>(row_W);

            if ((w_addr & 15) == 0) {
                int cur_vec16 = cur_tile_size / 16;
                const int4* W_vec16 = reinterpret_cast<const int4*>(row_W);
                const int4* s_x_vec16 = reinterpret_cast<const int4*>(s_x);

                #pragma unroll 4
                for (int k16 = lane_id; k16 < cur_vec16; k16 += 32) {
                    int4 w4 = W_vec16[k16];
                    int4 x4 = s_x_vec16[k16];
                    acc = __dp4a(w4.x, x4.x, acc);
                    acc = __dp4a(w4.y, x4.y, acc);
                    acc = __dp4a(w4.z, x4.z, acc);
                    acc = __dp4a(w4.w, x4.w, acc);
                }

                int k_rem_start = cur_vec16 * 16;
                int cur_vec4 = cur_tile_size / 4;
                const int32_t* W_vec4 = reinterpret_cast<const int32_t*>(row_W);
                const int32_t* s_x_vec4_r = reinterpret_cast<const int32_t*>(s_x);
                for (int k4 = (k_rem_start / 4) + lane_id; k4 < cur_vec4; k4 += 32) {
                    int32_t w4 = W_vec4[k4];
                    int32_t x4 = s_x_vec4_r[k4];
                    acc = __dp4a(w4, x4, acc);
                }

                int tile_scalar_start = cur_vec4 * 4;
                if (lane_id == 0) {
                    for (int k = tile_scalar_start; k < cur_tile_size; ++k) {
                        acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(s_x[k]);
                    }
                }
            } else if ((w_addr & 3) == 0) {
                int cur_vec4 = cur_tile_size / 4;
                const int32_t* W_vec4 = reinterpret_cast<const int32_t*>(row_W);
                const int32_t* s_x_vec4_r = reinterpret_cast<const int32_t*>(s_x);
                for (int k4 = lane_id; k4 < cur_vec4; k4 += 32) {
                    int32_t w4 = W_vec4[k4];
                    int32_t x4 = s_x_vec4_r[k4];
                    acc = __dp4a(w4, x4, acc);
                }
                int tile_scalar_start = cur_vec4 * 4;
                if (lane_id == 0) {
                    for (int k = tile_scalar_start; k < cur_tile_size; ++k) {
                        acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(s_x[k]);
                    }
                }
            } else {
                for (int k = lane_id; k < cur_tile_size; k += 32) {
                    acc += static_cast<int32_t>(row_W[k]) * static_cast<int32_t>(s_x[k]);
                }
            }
        }
        __syncthreads();
    }

    if (row < N) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane_id == 0) {
            y[row] = acc;
        }
    }
}

cudaError_t launch_gemv_scalar(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    gemv_scalar_kernel<<<grid, block, 0, stream>>>(d_W, d_x, d_y, N, K);
    return cudaGetLastError();
}

cudaError_t launch_gemv_dp4a_basic(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    gemv_dp4a_basic_kernel<<<grid, block, 0, stream>>>(d_W, d_x, d_y, N, K);
    return cudaGetLastError();
}

cudaError_t launch_gemv_dp4a_opt(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    gemv_dp4a_opt_kernel<<<grid, block, 0, stream>>>(d_W, d_x, d_y, N, K);
    return cudaGetLastError();
}

cudaError_t launch_gemv_dp4a_shared(
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream
) {
    dim3 block(32, SHARED_WARPS_PER_BLOCK);
    dim3 grid((N + SHARED_WARPS_PER_BLOCK - 1) / SHARED_WARPS_PER_BLOCK);
    gemv_dp4a_shared_kernel<<<grid, block, 0, stream>>>(d_W, d_x, d_y, N, K);
    return cudaGetLastError();
}

cudaError_t launch_gemv(
    KernelVariant variant,
    const int8_t* __restrict__ d_W,
    const int8_t* __restrict__ d_x,
    int32_t* __restrict__ d_y,
    int N, int K,
    cudaStream_t stream
) {
    switch (variant) {
        case KernelVariant::SCALAR:
            return launch_gemv_scalar(d_W, d_x, d_y, N, K, stream);
        case KernelVariant::DP4A_BASIC:
            return launch_gemv_dp4a_basic(d_W, d_x, d_y, N, K, stream);
        case KernelVariant::DP4A_OPT:
            return launch_gemv_dp4a_opt(d_W, d_x, d_y, N, K, stream);
        case KernelVariant::DP4A_SHARED:
            return launch_gemv_dp4a_shared(d_W, d_x, d_y, N, K, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

KernelOccupancyInfo get_kernel_occupancy_info(KernelVariant variant, int device_id) {
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    const void* kernel_func = nullptr;
    dim3 block_dim;

    switch (variant) {
        case KernelVariant::SCALAR:
            kernel_func = reinterpret_cast<const void*>(gemv_scalar_kernel);
            block_dim = dim3(32, 4);
            break;
        case KernelVariant::DP4A_BASIC:
            kernel_func = reinterpret_cast<const void*>(gemv_dp4a_basic_kernel);
            block_dim = dim3(32, 4);
            break;
        case KernelVariant::DP4A_OPT:
            kernel_func = reinterpret_cast<const void*>(gemv_dp4a_opt_kernel);
            block_dim = dim3(32, 8);
            break;
        case KernelVariant::DP4A_SHARED:
            kernel_func = reinterpret_cast<const void*>(gemv_dp4a_shared_kernel);
            block_dim = dim3(32, SHARED_WARPS_PER_BLOCK);
            break;
    }

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel_func);

    int threads_per_block = block_dim.x * block_dim.y;
    int active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        kernel_func,
        threads_per_block,
        0
    );

    int resident_threads = active_blocks * threads_per_block;
    int resident_warps = resident_threads / 32;
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    double occ_pct = (static_cast<double>(resident_threads) / static_cast<double>(prop.maxThreadsPerMultiProcessor)) * 100.0;

    KernelOccupancyInfo info;
    info.variant_name = get_variant_name(variant);
    info.block_dim_x = block_dim.x;
    info.block_dim_y = block_dim.y;
    info.threads_per_block = threads_per_block;
    info.registers_per_thread = attr.numRegs;
    info.static_shared_bytes = attr.sharedSizeBytes;
    info.active_blocks_per_sm = active_blocks;
    info.resident_threads_per_sm = resident_threads;
    info.resident_warps_per_sm = resident_warps;
    info.max_warps_per_sm = max_warps;
    info.theoretical_occupancy_pct = occ_pct;

    return info;
}

}
}
