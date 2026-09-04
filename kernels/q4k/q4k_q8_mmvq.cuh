#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string>

#define QK_K 256
#define K_SCALE_SIZE 12
#define QI4_K 32
#define QR4_K 2
#define VDR_Q4_K_Q8_1_MMVQ 2

#define QK8_1 32
#define QI8_1 8
#define QR8_1 4

// Standard canonical block_q4_K struct (144 bytes per 256 weights = 4.5 bpw)
typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// Standard canonical block_q8_1 struct (36 bytes per 32 activations)
typedef struct {
    half d;
    half s;
    int8_t qs[QK8_1];
} block_q8_1;

static_assert(sizeof(block_q4_K) == 144, "sizeof(block_q4_K) must equal 144 bytes");
static_assert(sizeof(block_q8_1) == 36, "sizeof(block_q8_1) must equal 36 bytes");

enum Q4KVariant {
    VARIANT_UPSTREAM_MMVQ = 0,    // Variant U: Exact Pinned llama.cpp MMVQ (libggml-cuda.so)
    VARIANT_4WARP_CONTROL = 1,    // Variant A: Custom 4-warp per row control
    VARIANT_2WARP_SPLIT   = 2,    // Variant B: Custom 2-warp per row (2 rows/block)
    VARIANT_1WARP_ROW     = 3     // Variant C: Custom 1-warp per row (4 rows/block)
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

// Host dispatch interface for Q4_K x Q8_1 MMVQ microkernels
void launch_q4k_q8_mmvq(
    Q4KVariant variant,
    const block_q4_K * d_w,
    const block_q8_1 * d_x,
    float * d_y,
    int N,
    int K,
    cudaStream_t stream = 0
);

// Pinned upstream activation quantizer: F32 -> Q8_1 (quantize_row_q8_1_cuda in libggml-cuda.so)
void launch_quantize_f32_to_q8_1(
    const float * d_x_f32,
    block_q8_1 * d_x_q8,
    int K,
    cudaStream_t stream = 0
);

// Custom standalone reference quantizer (for internal testing/smoke checks only)
void launch_custom_reference_q8_1_quantizer(
    const float * d_x_f32,
    block_q8_1 * d_x_q8,
    int K,
    cudaStream_t stream = 0
);

// Combined pipeline: F32 activation -> Q8_1 quantization (exact upstream) + Q4_K multiply
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

// Query theoretical occupancy information for a custom kernel variant via CUDA runtime
Q4KOccupancyInfo get_q4k_kernel_occupancy_info(Q4KVariant variant, int device_id);

const char * get_q4k_variant_name(Q4KVariant variant);
