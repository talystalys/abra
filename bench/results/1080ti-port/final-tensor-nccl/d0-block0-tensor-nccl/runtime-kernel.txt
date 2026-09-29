#pragma once
#include "legacy-ai-packed-q4.cuh"
// Runtime kernel with EXACT upstream virtual four-warp arithmetic order.
// Physical warps can differ: four independent partial sums are still formed
// over the same K subsequences, then added per lane in upstream order before
// the warp shuffle. Group multiple rows in a block; retain upstream vecdots.
static __device__ __forceinline__ float legacy_ai_q4_dot_cached_impl(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8, const int16_t * __restrict__ sums) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = sums[i * 4]; // exact precomputed four-byte-pair activation sum

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float legacy_ai_q4_dot_cached(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const int16_t * sums) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

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
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return legacy_ai_q4_dot_cached_impl(v, u, sc, m, bq4_K->dm, d8, sums + bq8_offset * 4 + ((iqs / 2) % 4));
}

static __device__ __forceinline__ float legacy_ai_q4_dot_packed(
    const void * weights, const block_q8_1 * activations, int index, int iqs) {
    const auto &w = static_cast<const legacy_ai_packed_q4 *>(weights)[index];
    const int offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int *q4 = reinterpret_cast<const int *>(w.qs + 16 * offset + 4 * ((iqs / 2) % 4));
    int v[2] = {q4[0], q4[4]};
    int u[2 * QR4_K];
    float d8[QR4_K];
    const uint32_t metadata = w.scale_min_pairs[offset / 2];
    const uint8_t *sc = reinterpret_cast<const uint8_t *>(&metadata);
    const uint8_t *m = sc + 2;
    for (int i = 0; i < QR4_K; ++i) {
        const auto &x = activations[offset + i];
        d8[i] = __low2float(x.ds);
        const int *q8 = reinterpret_cast<const int *>(x.qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    // Preserve the upstream float operations exactly; only metadata decoding moves.
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, w.dm, d8);
}

template <ggml_type TYPE, int WPR, int WPB, bool GATE = false, bool PRECOMPUTE = false, bool PACKED = false>
static __global__ void legacy_ai_mmvq_kernel(
    const void * __restrict__ weights, const block_q8_1 * __restrict__ x,
    float * __restrict__ output, int N, int K,
    const void * gate_weights = nullptr, ggml_glu_op glu = GGML_GLU_OP_SWIGLU) {
    static_assert(WPR == 1 || WPR == 2 || WPR == 4);
    static_assert(WPB % WPR == 0);
    constexpr int PARTS = 4 / WPR;
    constexpr int ROWS = WPB / WPR;
    constexpr int VDR = TYPE == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ : VDR_Q6_K_Q8_1_MMVQ;
    constexpr int QI = ggml_cuda_type_traits<TYPE>::qi;
    constexpr int STRIDE = VDR * 4 * 32 / QI;
    const int lane = threadIdx.x;
    const int physical_warp = threadIdx.y % WPR;
    const int group = threadIdx.y / WPR;
    const int row = blockIdx.x * ROWS + group;
    const int blocks = K / QK_K;
    const int iqs = VDR * (lane % (QI / VDR));
    extern __shared__ int16_t activation_sums[];
    if constexpr (PRECOMPUTE && TYPE == GGML_TYPE_Q4_K) {
        for (int index = threadIdx.y * 32 + lane; index < K / 8; index += WPB * 32) {
            const int * words = reinterpret_cast<const int *>(x[index / 4].qs);
            const int part = index % 4;
            activation_sums[index] = ggml_cuda_dp4a(0x01010101, words[part + 4],
                                                   ggml_cuda_dp4a(0x01010101, words[part], 0));
        }
        __syncthreads(); // All output rows share exact activation sums.
    }
    float sums[PARTS] = {};
    float gate_sums[PARTS] = {};
    if (row < N) {
        #pragma unroll
        for (int part = 0; part < PARTS; ++part) {
            const int virtual_warp = physical_warp + part * WPR;
            for (int k = (virtual_warp * 32 + lane) / (QI / VDR); k < blocks; k += STRIDE) {
                const int weight_index = row * blocks + k;
                const block_q8_1 * activations = x + k * (QK_K / QK8_1);
                if constexpr (TYPE == GGML_TYPE_Q4_K) {
                    if constexpr (PACKED) {
                        sums[part] += legacy_ai_q4_dot_packed(weights, activations, weight_index, iqs);
                        if constexpr (GATE) gate_sums[part] += legacy_ai_q4_dot_packed(gate_weights, activations, weight_index, iqs);
                    } else if constexpr (PRECOMPUTE) {
                        sums[part] += legacy_ai_q4_dot_cached(weights, activations, weight_index, iqs, activation_sums + k * 32);
                        if constexpr (GATE) gate_sums[part] += legacy_ai_q4_dot_cached(gate_weights, activations, weight_index, iqs, activation_sums + k * 32);
                    } else {
                        sums[part] += vec_dot_q4_K_q8_1(weights, activations, weight_index, iqs);
                        if constexpr (GATE) gate_sums[part] += vec_dot_q4_K_q8_1(gate_weights, activations, weight_index, iqs);
                    }
                } else {
                    sums[part] += vec_dot_q6_K_q8_1(weights, activations, weight_index, iqs);
                    if constexpr (GATE) gate_sums[part] += vec_dot_q6_K_q8_1(gate_weights, activations, weight_index, iqs);
                }
            }
        }
    }
    constexpr int SHARED_WARPS = ROWS * (WPR > 1 ? WPR - 1 : 1);
    __shared__ float shared[WPR > 1 ? SHARED_WARPS : 1][WPR > 1 ? PARTS : 1][WPR > 1 ? 32 : 1];
    __shared__ float shared_gate[GATE && WPR > 1 ? SHARED_WARPS : 1][GATE && WPR > 1 ? PARTS : 1][GATE && WPR > 1 ? 32 : 1];
    if constexpr (WPR > 1) {
        if (physical_warp > 0) {
            #pragma unroll
            for (int part = 0; part < PARTS; ++part) {
                shared[group * (WPR - 1) + physical_warp - 1][part][lane] = sums[part];
                if constexpr (GATE) shared_gate[group * (WPR - 1) + physical_warp - 1][part][lane] = gate_sums[part];
            }
        }
        __syncthreads(); // Every row, including out-of-range rows, participates.
    }
    if (physical_warp == 0 && row < N) {
        float total = sums[0];
        float gate_total = gate_sums[0];
        #pragma unroll
        for (int virtual_warp = 1; virtual_warp < 4; ++virtual_warp) {
            constexpr bool has_shared = WPR > 1;
            const int owner = virtual_warp % WPR;
            const int part = virtual_warp / WPR;
            if (owner == 0) {
                total += sums[part];
                if constexpr (GATE) gate_total += gate_sums[part];
            } else if constexpr (has_shared) {
                total += shared[group * (WPR - 1) + owner - 1][part][lane];
                if constexpr (GATE) gate_total += shared_gate[group * (WPR - 1) + owner - 1][part][lane];
            }
        }
        total = warp_reduce_sum<32>(total);
        if constexpr (GATE) {
            gate_total = warp_reduce_sum<32>(gate_total);
            // Upstream fused path adds zero bias before GLU when absent.
            total += 0.0f;
            gate_total += 0.0f;
            switch (glu) {
                case GGML_GLU_OP_SWIGLU: total *= ggml_cuda_op_silu_single(gate_total); break;
                case GGML_GLU_OP_GEGLU: total *= ggml_cuda_op_gelu_single(gate_total); break;
                case GGML_GLU_OP_SWIGLU_OAI: total = ggml_cuda_op_swiglu_oai_single(gate_total, total); break;
                default: total *= gate_total; break;
            }
        }
        if (lane == 0) output[row] = total;
    }
}

template <ggml_type TYPE, int WPR, int WPB>
static void legacy_ai_launch_mmvq(const void * w, const block_q8_1 * x, float * y,
    int N, int K, const ggml_cuda_mm_fusion_args_device & fusion, cudaStream_t stream, bool precompute = false, bool packed = false) {
    const dim3 grid((N + WPB / WPR - 1) / (WPB / WPR));
    const dim3 block(32, WPB);
    if (packed && TYPE == GGML_TYPE_Q4_K) {
        if (fusion.gate) legacy_ai_mmvq_kernel<TYPE, WPR, WPB, true, false, true><<<grid, block, 0, stream>>>(w, x, y, N, K, fusion.gate, fusion.glu_op);
        else legacy_ai_mmvq_kernel<TYPE, WPR, WPB, false, false, true><<<grid, block, 0, stream>>>(w, x, y, N, K);
    } else if (precompute && TYPE == GGML_TYPE_Q4_K) {
        if (fusion.gate) legacy_ai_mmvq_kernel<TYPE, WPR, WPB, true, true><<<grid, block, K / 4, stream>>>(w, x, y, N, K, fusion.gate, fusion.glu_op);
        else legacy_ai_mmvq_kernel<TYPE, WPR, WPB, false, true><<<grid, block, K / 4, stream>>>(w, x, y, N, K);
    } else if (fusion.gate) {
        legacy_ai_mmvq_kernel<TYPE, WPR, WPB, true><<<grid, block, 0, stream>>>(w, x, y, N, K, fusion.gate, fusion.glu_op);
    } else {
        legacy_ai_mmvq_kernel<TYPE, WPR, WPB, false><<<grid, block, 0, stream>>>(w, x, y, N, K);
    }
}

template <ggml_type TYPE>
static bool legacy_ai_dispatch_mmvq(int wpr, int wpb, const void * w, const block_q8_1 * x,
    float * y, int N, int K, const ggml_cuda_mm_fusion_args_device & fusion, cudaStream_t stream, bool precompute = false, bool packed = false) {
    #define LEGACY_CASE(R, B) case R * 100 + B: legacy_ai_launch_mmvq<TYPE, R, B>(w, x, y, N, K, fusion, stream, precompute, packed); return true
    switch (wpr * 100 + wpb) {
        LEGACY_CASE(1, 4); LEGACY_CASE(1, 8); LEGACY_CASE(1, 16);
        LEGACY_CASE(2, 4); LEGACY_CASE(2, 8); LEGACY_CASE(2, 16);
        LEGACY_CASE(4, 4); LEGACY_CASE(4, 8); LEGACY_CASE(4, 16);
        default: return false;
    }
    #undef LEGACY_CASE
}
