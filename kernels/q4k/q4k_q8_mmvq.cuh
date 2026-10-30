#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string>

#ifndef QK_K
#define QK_K 256
#endif

#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

#ifndef QI4_K
#define QI4_K 32
#endif

#ifndef QR4_K
#define QR4_K 2
#endif

#ifndef VDR_Q4_K_Q8_1_MMVQ
#define VDR_Q4_K_Q8_1_MMVQ 2
#endif

#ifndef QK8_1
#define QK8_1 32
#endif

#ifndef QI8_1
#define QI8_1 8
#endif

#ifndef QR8_1
#define QR8_1 4
#endif

#if !defined(GGML_COMMON_DECL_CUDA) && !defined(GGML_COMMON_DECL_HIP) && !defined(GGML_COMMON_DECL_MUSA) && !defined(GGML_COMMON_DECL_CPP) && !defined(GGML_COMMON_DECL_C)

typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    half d;
    half s;
    int8_t qs[QK8_1];
} block_q8_1;
#endif

static_assert(sizeof(block_q4_K) == 144, "sizeof(block_q4_K) must equal 144 bytes");
static_assert(sizeof(block_q8_1) == 36, "sizeof(block_q8_1) must equal 36 bytes");

enum Q4KVariant {
    VARIANT_UPSTREAM_MMVQ = 0,
    VARIANT_4WARP_CONTROL = 1,
    VARIANT_2WARP_SPLIT   = 2,
    VARIANT_1WARP_ROW     = 3
};

struct Q4KOccupancyInfo {
    std::string variant_name;
    int block_dim_x;
    int block_dim_y;
    int threads_per_block;
    int regs_per_thread;
    size_t static_shared_mem_bytes;
    int max_active_blocks_per_sm;
    int resident_warps_per_sm;
    int max_warps_per_sm;
    double theoretical_occupancy_pct;
    int spill_stores_bytes;
    int spill_loads_bytes;
};

void launch_q4k_q8_mmvq(
    Q4KVariant variant,
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream = 0
);

void launch_quantize_f32_to_q8_1(
    const float * d_x_f32,
    block_q8_1 * d_x_q8,
    int K,
    cudaStream_t stream = 0
);

void launch_custom_reference_q8_1_quantizer(
    const float * d_x_f32,
    block_q8_1 * d_x_q8,
    int K,
    cudaStream_t stream = 0
);

void launch_q4k_q8_combined_pipeline(
    Q4KVariant variant,
    const block_q4_K * d_w,
    const float * d_x_f32,
    block_q8_1 * d_x_q8_scratch,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream = 0
);

Q4KOccupancyInfo get_q4k_kernel_occupancy_info(Q4KVariant variant, int device_id);

const char * get_q4k_variant_name(Q4KVariant variant);
