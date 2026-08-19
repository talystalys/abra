#include "common.cuh"
#include "upstream_adapter.cuh"
#include <iostream>
#include <cuda_runtime.h>

#include "ggml.h"

// C++ mangled signatures exported by libggml-cuda.so
void quantize_row_q8_1_cuda(
    const float * x, const int * imatrix, void * vy, enum ggml_type type, int64_t ne00,
    int64_t s01, int64_t s02, int64_t s03, int64_t ne00_padded,
    int64_t ne01, int64_t ne02, int64_t ne03, cudaStream_t stream);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i, const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

extern "C" {

void quantize_row_q4_K_ref(const float * x, void * y, int64_t k);
void dequantize_row_q4_K(const void * x, float * y, int64_t k);

void launch_pinned_upstream_mmvq(
    const void * d_w_q4,
    const void * d_x_q8,
    float * d_y,
    int64_t N,
    int64_t K,
    cudaStream_t stream
) {
    int dev_id = 0;
    cudaGetDevice(&dev_id);

    ggml_backend_cuda_context ctx(dev_id);

    ggml_tensor src0{};
    src0.type = GGML_TYPE_Q4_K;
    src0.ne[0] = K;
    src0.ne[1] = N;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = sizeof(block_q4_K);
    src0.nb[1] = (K / QK_K) * sizeof(block_q4_K);

    ggml_tensor src1{};
    src1.type = GGML_TYPE_F32;
    src1.ne[0] = K;
    src1.ne[1] = 1;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = sizeof(float);
    src1.nb[1] = K * sizeof(float);

    ggml_tensor dst{};
    dst.type = GGML_TYPE_F32;
    dst.ne[0] = N;
    dst.ne[1] = 1;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = sizeof(float);
    dst.nb[1] = N * sizeof(float);

    ggml_cuda_op_mul_mat_vec_q(
        ctx,
        &src0,
        &src1,
        &dst,
        (const char *)d_w_q4,
        nullptr,
        (const char *)d_x_q8,
        d_y,
        0,
        N,
        1,
        K,
        stream
    );
}

void launch_pinned_upstream_q8_1_quantize(
    const float * d_x_f32,
    void * d_x_q8,
    int64_t K,
    cudaStream_t stream
) {
    const int64_t K_padded = ((K + 511) / 512) * 512;
    quantize_row_q8_1_cuda(
        d_x_f32,
        nullptr,
        d_x_q8,
        GGML_TYPE_Q4_K,
        K,
        K,
        K,
        K,
        K_padded,
        1,
        1,
        1,
        stream
    );
}

void quantize_row_q4_K_upstream_cpu(
    const float * x,
    block_q4_K * y,
    int64_t k
) {
    quantize_row_q4_K_ref(x, (void *)y, k);
}

void dequantize_row_q4_K_upstream_cpu(
    const block_q4_K * x,
    float * y,
    int64_t k
) {
    dequantize_row_q4_K((const void *)x, y, k);
}

} // extern "C"
