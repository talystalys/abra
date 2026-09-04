#pragma once

#include "q4k_q8_mmvq.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Invokes the actual compiled pinned upstream llama.cpp MMVQ decode implementation
 * (ggml_cuda_op_mul_mat_vec_q) from libggml-cuda.so.
 */
void launch_pinned_upstream_mmvq(
    const void * d_w_q4,     // Device pointer to block_q4_K array [N, K/256]
    const void * d_x_q8,     // Device pointer to block_q8_1 array [K/32]
    float * d_y,             // Device pointer to float output array [N]
    int64_t N,
    int64_t K,
    cudaStream_t stream
);

/**
 * Invokes the actual compiled pinned upstream llama.cpp CUDA activation quantizer
 * (quantize_row_q8_1_cuda) from libggml-cuda.so.
 */
void launch_pinned_upstream_q8_1_quantize(
    const float * d_x_f32,   // Device pointer to input float array [K]
    void * d_x_q8,           // Device pointer to output block_q8_1 array [K/32]
    int64_t K,
    cudaStream_t stream
);

/**
 * Invokes the actual compiled pinned upstream CPU Q4_K quantizer
 * (quantize_row_q4_K_ref) from libggml-base.so.
 */
void quantize_row_q4_K_upstream_cpu(
    const float * x,
    block_q4_K * y,
    int64_t k
);

/**
 * Invokes the actual compiled pinned upstream CPU Q4_K dequantizer
 * (dequantize_row_q4_K) from libggml-base.so.
 */
void dequantize_row_q4_K_upstream_cpu(
    const block_q4_K * x,
    float * y,
    int64_t k
);

#ifdef __cplusplus
}
#endif
