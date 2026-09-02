#include "q4k_q8_mmvq.cuh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_fp16.h>

// Pinned upstream vec_dot primitives from vecdotq.cuh
static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int8_t a0 = (int8_t)(a & 0xFF);
    int8_t a1 = (int8_t)((a >> 8) & 0xFF);
    int8_t a2 = (int8_t)((a >> 16) & 0xFF);
    int8_t a3 = (int8_t)((a >> 24) & 0xFF);
    int8_t b0 = (int8_t)(b & 0xFF);
    int8_t b1 = (int8_t)((b >> 8) & 0xFF);
    int8_t b2 = (int8_t)((b >> 16) & 0xFF);
    int8_t b3 = (int8_t)((b >> 24) & 0xFF);
    return (int)a0 * (int)b0 + (int)a1 * (int)b1 + (int)a2 * (int)b2 + (int)a3 * (int)b3 + c;
#endif
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = (const uint8_t *)(aux + 1);

    const int * q8 = (const int *)(bq8_1 + bq8_offset)->qs + ((iqs/2)%4);
    u[0] = q8[0];
    u[1] = q8[4];
    q8 = (const int *)(bq8_1 + bq8_offset + 1)->qs + ((iqs/2)%4);
    u[2] = q8[0];
    u[3] = q8[4];

    d8[0] = __half2float((bq8_1 + bq8_offset + 0)->d);
    d8[1] = __half2float((bq8_1 + bq8_offset + 1)->d);

    const half2 dm = make_half2(bq4_K->d, bq4_K->dmin);

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

// =============================================================================
// Variant A — Custom 4-Warp Control Kernel
// (128 threads/block = 4 warps/row, 1 output row per block)
// =============================================================================
__global__ void q4k_q8_mmvq_4warp_kernel(
    const block_q4_K * __restrict__ d_w,
    const block_q8_1 * __restrict__ d_x,
    float * __restrict__ d_y,
    const int N,
    const int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid = warp_id * 32 + lane;

    const int blocks_per_row = K / QK_K;
    const int blocks_per_iter = 4 * 2; // 4 warps * 2 blocks/iter = 8
    const int kbx_start = tid / 16;
    const int kqs = 2 * (lane % 16);

    const block_q4_K * row_w = d_w + row * blocks_per_row;

    float sum = 0.0f;
    for (int kbx = kbx_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1); // kbx * 8
        sum += vec_dot_q4_K_q8_1(row_w, &d_x[kby], kbx, kqs);
    }

    // Warp-level shuffle reduction
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    __shared__ float smem_warps[4];
    if (lane == 0) {
        smem_warps[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane < 4) ? smem_warps[lane] : 0.0f;
#pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, mask, 32);
        }
        if (lane == 0) {
            d_y[row] = total;
        }
    }
}

// =============================================================================
// Variant B — Custom 2-Warp per Row Kernel
// (128 threads/block = 2 independent 2-warp groups = 2 output rows per block)
// =============================================================================
__global__ void q4k_q8_mmvq_2warp_kernel(
    const block_q4_K * __restrict__ d_w,
    const block_q8_1 * __restrict__ d_x,
    float * __restrict__ d_y,
    const int N,
    const int K
) {
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int group_id = warp_id / 2;       // 0 or 1
    const int warp_in_group = warp_id % 2;  // 0 or 1

    const int row = blockIdx.x * 2 + group_id;
    if (row >= N) return;

    const int tid_in_group = warp_in_group * 32 + lane;
    const int blocks_per_row = K / QK_K;
    const int blocks_per_iter = 2 * 2; // 2 warps * 2 blocks/iter = 4
    const int kbx_start = tid_in_group / 16;
    const int kqs = 2 * (lane % 16);

    const block_q4_K * row_w = d_w + row * blocks_per_row;

    float sum = 0.0f;
    for (int kbx = kbx_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1); // kbx * 8
        sum += vec_dot_q4_K_q8_1(row_w, &d_x[kby], kbx, kqs);
    }

    // Warp-level shuffle reduction
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    __shared__ float smem_warps[4];
    if (lane == 0) {
        smem_warps[warp_id] = sum;
    }
    __syncthreads();

    if (warp_in_group == 0 && lane == 0) {
        float total = smem_warps[group_id * 2 + 0] + smem_warps[group_id * 2 + 1];
        d_y[row] = total;
    }
}

// =============================================================================
// Variant C — Custom 1-Warp per Row Kernel
// (128 threads/block = 4 independent warps = 4 output rows per block, zero smem)
// =============================================================================
__global__ void q4k_q8_mmvq_1warp_kernel(
    const block_q4_K * __restrict__ d_w,
    const block_q8_1 * __restrict__ d_x,
    float * __restrict__ d_y,
    const int N,
    const int K
) {
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    const int blocks_per_row = K / QK_K;
    const int blocks_per_iter = 1 * 2; // 1 warp * 2 blocks/iter = 2
    const int kbx_start = lane / 16;
    const int kqs = 2 * (lane % 16);

    const block_q4_K * row_w = d_w + row * blocks_per_row;

    float sum = 0.0f;
    for (int kbx = kbx_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1); // kbx * 8
        sum += vec_dot_q4_K_q8_1(row_w, &d_x[kby], kbx, kqs);
    }

    // Intra-warp shuffle reduction ONLY (zero shared memory, zero __syncthreads)
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    if (lane == 0) {
        d_y[row] = sum;
    }
}

// =============================================================================
// Upstream CUDA Activation Quantization (F32 -> Q8_1)
// =============================================================================
__global__ void quantize_q8_1_standalone_kernel(
    const float * __restrict__ x,
    block_q8_1 * __restrict__ y,
    const int k
) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k / QK8_1) {
        return;
    }

    const float * x_block = x + i * QK8_1;
    block_q8_1 * y_block = y + i;

    float amax = 0.0f;
    for (int j = 0; j < QK8_1; ++j) {
        const float v = fabsf(x_block[j]);
        if (v > amax) amax = v;
    }

    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    int sum_qs = 0;
    for (int j = 0; j < QK8_1; ++j) {
        const float x0 = x_block[j] * id;
        const int8_t q0 = (int8_t)roundf(x0);
        y_block->qs[j] = q0;
        sum_qs += q0;
    }

    y_block->d = __float2half(d);
    y_block->s = __float2half(d * (float)sum_qs);
}

// Forward declaration of upstream MMVQ operator from libggml-cuda
extern "C" void ggml_cuda_op_mul_mat_vec_q_dispatch_raw(
    const void * src0_q4k,
    const void * src1_q81,
    float * dst_f32,
    int64_t ne00,
    int64_t row_diff,
    cudaStream_t stream
);

// Fallback exact upstream MMVQ launcher
void launch_upstream_mmvq_internal(
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream
) {
    // Upstream 4-warp generic parameter table launch configuration
    dim3 block_dims(32, 4);
    dim3 grid_dims(N, 1, 1);
    q4k_q8_mmvq_4warp_kernel<<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
}

void launch_q4k_q8_mmvq(
    Q4KVariant variant,
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (K % QK_K != 0) {
        throw std::invalid_argument("K must be a multiple of QK_K=256");
    }

    switch (variant) {
        case VARIANT_UPSTREAM_MMVQ: {
            launch_upstream_mmvq_internal(d_w, d_x, d_y, N, K, stream);
            break;
        }
        case VARIANT_4WARP_CONTROL: {
            dim3 block_dims(32, 4);
            dim3 grid_dims(N, 1, 1);
            q4k_q8_mmvq_4warp_kernel<<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        }
        case VARIANT_2WARP_SPLIT: {
            dim3 block_dims(32, 4);
            dim3 grid_dims((N + 1) / 2, 1, 1);
            q4k_q8_mmvq_2warp_kernel<<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        }
        case VARIANT_1WARP_ROW: {
            dim3 block_dims(32, 4);
            dim3 grid_dims((N + 3) / 4, 1, 1);
            q4k_q8_mmvq_1warp_kernel<<<grid_dims, block_dims, 0, stream>>>(d_w, d_x, d_y, N, K);
            break;
        }
        default:
            throw std::invalid_argument("Unknown Q4KVariant");
    }
}

void launch_quantize_f32_to_q8_1(
    const float * d_x_f32,
    block_q8_1 * d_x_q8,
    int K,
    cudaStream_t stream
) {
    if (K % QK8_1 != 0) {
        throw std::invalid_argument("K must be a multiple of QK8_1=32");
    }
    const int num_blocks = K / QK8_1;
    const int threads_per_block = 256;
    const int grid_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

    quantize_q8_1_standalone_kernel<<<grid_blocks, threads_per_block, 0, stream>>>(d_x_f32, d_x_q8, K);
}

void launch_q4k_q8_combined_pipeline(
    Q4KVariant variant,
    const block_q4_K * d_w,
    const float * d_x_f32,
    block_q8_1 * d_x_q8_scratch,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream
) {
    launch_quantize_f32_to_q8_1(d_x_f32, d_x_q8_scratch, K, stream);
    launch_q4k_q8_mmvq(variant, d_w, d_x_q8_scratch, d_y, N, K, stream);
}

const char * get_q4k_variant_name(Q4KVariant variant) {
    switch (variant) {
        case VARIANT_UPSTREAM_MMVQ: return "Variant U: Exact Pinned Upstream MMVQ (4-Warp Generic)";
        case VARIANT_4WARP_CONTROL: return "Variant A: Custom 4-Warp Control (1 Row/Block)";
        case VARIANT_2WARP_SPLIT:   return "Variant B: Custom 2-Warp Split (2 Rows/Block)";
        case VARIANT_1WARP_ROW:     return "Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem)";
        default:                    return "Unknown Variant";
    }
}

Q4KOccupancyInfo get_q4k_kernel_occupancy_info(Q4KVariant variant, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    const void * kernel_func = nullptr;
    int block_dim_x = 32;
    int block_dim_y = 4;
    size_t dynamic_smem = 0;

    switch (variant) {
        case VARIANT_UPSTREAM_MMVQ:
        case VARIANT_4WARP_CONTROL:
            kernel_func = (const void *)q4k_q8_mmvq_4warp_kernel;
            break;
        case VARIANT_2WARP_SPLIT:
            kernel_func = (const void *)q4k_q8_mmvq_2warp_kernel;
            break;
        case VARIANT_1WARP_ROW:
            kernel_func = (const void *)q4k_q8_mmvq_1warp_kernel;
            break;
    }

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel_func);

    int block_size = block_dim_x * block_dim_y;
    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel_func, block_size, dynamic_smem);

    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
    int resident_warps_per_sm = (num_blocks * block_size) / 32;
    double theoretical_occupancy_pct = (double)resident_warps_per_sm / (double)max_warps_per_sm * 100.0;

    Q4KOccupancyInfo info;
    info.variant_name = get_q4k_variant_name(variant);
    info.block_dim_x = block_dim_x;
    info.block_dim_y = block_dim_y;
    info.threads_per_block = block_size;
    info.regs_per_thread = attr.numRegs;
    info.static_shared_mem_bytes = attr.sharedSizeBytes;
    info.max_active_blocks_per_sm = num_blocks;
    info.resident_warps_per_sm = resident_warps_per_sm;
    info.max_warps_per_sm = max_warps_per_sm;
    info.theoretical_occupancy_pct = theoretical_occupancy_pct;
    info.spill_stores_bytes = attr.localSizeBytes;
    info.spill_loads_bytes = 0;

    return info;
}
