#pragma once

#include "q4k_q8_mmvq.cuh"
#include <cuda_fp16.h>
#include <cstdint>

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

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
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
