#pragma once

#include "q4k_q8_mmvq.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void launch_pinned_upstream_mmvq(
    const void * d_w_q4,
    const void * d_x_q8,
    float * d_y,
    int64_t N,
    int64_t K,
    cudaStream_t stream
);

void launch_pinned_upstream_q8_1_quantize(
    const float * d_x_f32,
    void * d_x_q8,
    int64_t K,
    cudaStream_t stream
);

void quantize_row_q4_K_upstream_cpu(
    const float * x,
    block_q4_K * y,
    int64_t k
);

void dequantize_row_q4_K_upstream_cpu(
    const block_q4_K * x,
    float * y,
    int64_t k
);

#ifdef __cplusplus
}
#endif
