// Correctness of physical warp remapping against upstream and double oracle.
#include "common.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "kernels/q4k/legacy-ai-mmvq.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>
extern "C" {
void quantize_row_q4_K_ref(const float *, block_q4_K *, int64_t);
void quantize_row_q6_K_ref(const float *, block_q6_K *, int64_t);
}
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Independent Reference: High-precision double oracle directly from quantized blocks
void independent_q4k_q8_oracle_f64(
    const block_q4_K * w_q4,
    const block_q8_1 * x_q8,
    double * y_f64,
    int N,
    int K
) {
    const int blocks_per_row = K / QK_K;
    for (int n = 0; n < N; ++n) {
        double row_sum = 0.0;
        const block_q4_K * row_w = w_q4 + n * blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_K * bq4 = row_w + b;
            const block_q8_1 * bq8_base = x_q8 + b * 8;

            float d4 = __low2float(bq4->dm);
            float dmin4 = __high2float(bq4->dm);

            const uint8_t * q = bq4->qs;
            int is = 0;

            for (int j = 0; j < QK_K; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                get_scale_min_k4(is + 0, bq4->scales, &sc0, &m0);
                get_scale_min_k4(is + 1, bq4->scales, &sc1, &m1);

                double d1 = (double)d4 * (double)sc0;
                double m_val1 = (double)dmin4 * (double)m0;
                double d2 = (double)d4 * (double)sc1;
                double m_val2 = (double)dmin4 * (double)m1;

                const block_q8_1 * bq8_0 = bq8_base + is + 0;
                const block_q8_1 * bq8_1 = bq8_base + is + 1;
                double d8_0 = (double)__low2float(bq8_0->ds);
                double d8_1 = (double)__low2float(bq8_1->ds);

                for (int l = 0; l < 32; ++l) {
                    int q0 = q[l] & 0xF;
                    int q1 = (q[l] >> 4) & 0xF;
                    double w0 = d1 * (double)q0 - m_val1;
                    double w1 = d2 * (double)q1 - m_val2;
                    double x0 = d8_0 * (double)bq8_0->qs[l];
                    double x1 = d8_1 * (double)bq8_1->qs[l];
                    row_sum += w0 * x0 + w1 * x1;
                }

                q += 32;
                is += 2;
            }
        }
        y_f64[n] = row_sum;
    }
}

static void q6_oracle(const block_q6_K *w, const block_q8_1 *x, double *y, int N, int K) {
    for (int row = 0; row < N; ++row) {
        double total = 0;
        for (int b = 0; b < K / 256; ++b) {
            const auto &block = w[row * (K / 256) + b];
            const double d = __half2float(block.d);
            for (int half = 0; half < 2; ++half) {
                const auto *ql = block.ql + half * 64;
                const auto *qh = block.qh + half * 32;
                const auto *scales = block.scales + half * 8;
                for (int lane = 0; lane < 32; ++lane) {
                    for (int group = 0; group < 4; ++group) {
                        int low = group < 2 ? ql[lane + (group % 2) * 32] & 15 : ql[lane + (group % 2) * 32] >> 4;
                        int quant = (low | (((qh[lane] >> (group * 2)) & 3) << 4)) - 32;
                        int index = half * 128 + group * 32 + lane;
                        const auto &activation = x[b * 8 + index / 32];
                        total += d * scales[lane / 16 + group * 2] * quant *
                                 double(__low2float(activation.ds)) * activation.qs[index % 32];
                    }
                }
            }
        }
        y[row] = total;
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, CC %d.%d, PCI %04x:%02x:%02x.0\n", prop.name, prop.major, prop.minor,
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (std::strcmp(prop.name, "NVIDIA GeForce GTX 1080 Ti") || prop.major != 6 || prop.minor != 1) return 2;
    std::mt19937 random(12026);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    int count = 0;
    double max_error = 0;
    ggml_backend_cuda_context ctx(0);
    for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (int N : {1, 33, 64}) for (int K : {256, 768, 2560, 4096, 9728}) for (int pattern : {0, 1, 2}) {
            std::vector<float> weights(N * K), input(K), actual(N), reference(N);
            for (auto &value : weights) value = pattern == 0 ? 0 : distribution(random) * (pattern == 2 ? 100 : 1);
            for (auto &value : input) value = pattern == 0 ? 0 : distribution(random);
            const size_t bytes = N * (K / QK_K) * (type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q6_K));
            std::vector<char> packed(bytes);
            if (type == GGML_TYPE_Q4_K) quantize_row_q4_K_ref(weights.data(), reinterpret_cast<block_q4_K *>(packed.data()), N * K);
            else quantize_row_q6_K_ref(weights.data(), reinterpret_cast<block_q6_K *>(packed.data()), N * K);
            void *dw;
            legacy_ai_packed_q4 *dpacked = nullptr;
            float *dx, *dy;
            block_q8_1 *qx;
            int padded = (K + 511) / 512 * 512;
            CUDA_CHECK(cudaMalloc(&dw, bytes));
            CUDA_CHECK(cudaMalloc(&dx, K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&qx, padded / 32 * sizeof(block_q8_1)));
            CUDA_CHECK(cudaMemcpy(dw, packed.data(), bytes, cudaMemcpyHostToDevice));
            if (type == GGML_TYPE_Q4_K) {
                const int blocks = N * K / 256;
                CUDA_CHECK(cudaMalloc(&dpacked, blocks * sizeof(legacy_ai_packed_q4)));
                legacy_ai_pack_q4_kernel<<<(blocks + 255) / 256, 256>>>(static_cast<const block_q4_K *>(dw), dpacked, blocks);
                CUDA_CHECK(cudaGetLastError());
                std::vector<legacy_ai_packed_q4> actual_pack(blocks);
                CUDA_CHECK(cudaMemcpy(actual_pack.data(), dpacked, blocks * sizeof(legacy_ai_packed_q4), cudaMemcpyDeviceToHost));
                for (int b = 0; b < blocks; ++b) {
                    const auto &original = reinterpret_cast<const block_q4_K *>(packed.data())[b];
                    const auto &expanded = actual_pack[b];
                    if (std::memcmp(&original.dm, &expanded.dm, 4) || std::memcmp(original.qs, expanded.qs, 128)) return 4;
                    for (int pair = 0; pair < 4; ++pair) for (int index = 0; index < 2; ++index) {
                        uint8_t sc, minimum;
                        get_scale_min_k4(pair * 2 + index, original.scales, &sc, &minimum);
                        const auto *bytes = reinterpret_cast<const uint8_t *>(&expanded.scale_min_pairs[pair]);
                        if (bytes[index] != sc || bytes[index + 2] != minimum) return 5;
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(dx, input.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            quantize_row_q8_1_cuda(dx, nullptr, qx, type, K, K, K, K, padded, 1, 1, 1, 0);
            CUDA_CHECK(cudaGetLastError());
            std::vector<block_q8_1> host_q8(padded / 32);
            CUDA_CHECK(cudaMemcpy(host_q8.data(), qx, host_q8.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
            std::vector<double> oracle(N);
            if (type == GGML_TYPE_Q4_K) independent_q4k_q8_oracle_f64(reinterpret_cast<block_q4_K *>(packed.data()), host_q8.data(), oracle.data(), N, K);
            else q6_oracle(reinterpret_cast<block_q6_K *>(packed.data()), host_q8.data(), oracle.data(), N, K);
            ggml_tensor w{}, x{}, y{};
            w.type = type; x.type = y.type = GGML_TYPE_F32;
            w.ne[0] = x.ne[0] = K; w.ne[1] = y.ne[0] = N;
            w.ne[2] = w.ne[3] = x.ne[1] = x.ne[2] = x.ne[3] = y.ne[1] = y.ne[2] = y.ne[3] = 1;
            ggml_cuda_op_mul_mat_vec_q(ctx, &w, &x, &y, static_cast<const char *>(dw), nullptr,
                                      reinterpret_cast<const char *>(qx), dy, 0, N, 1, padded, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(reference.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
            for (int wpr : {1, 2, 4}) for (int wpb : {4, 8, 16}) for (bool sums : {false, true}) for (bool expanded : {false, true}) {
                if (expanded && (type != GGML_TYPE_Q4_K || sums)) continue;
                ggml_cuda_mm_fusion_args_device fusion{};
                bool dispatched = type == GGML_TYPE_Q4_K ?
                    legacy_ai_dispatch_mmvq<GGML_TYPE_Q4_K>(wpr, wpb, expanded ? static_cast<const void *>(dpacked) : dw, qx, dy, N, K, fusion, 0, sums, expanded) :
                    legacy_ai_dispatch_mmvq<GGML_TYPE_Q6_K>(wpr, wpb, dw, qx, dy, N, K, fusion, 0);
                if (!dispatched) return 3;
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(actual.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
                for (int row = 0; row < N; ++row) {
                    double error = std::abs(double(actual[row]) - oracle[row]);
                    double rel = error / std::max(1e-5, std::abs(oracle[row]));
                    max_error = std::max(max_error, error);
                    if (!std::isfinite(actual[row]) || (error > 0.05 && rel > 0.005) ||
                        std::memcmp(&actual[row], &reference[row], sizeof(float))) {
                        std::fprintf(stderr, "FAIL type=%d N=%d K=%d pattern=%d WPR=%d WPB=%d row=%d got=%.9g upstream=%.9g oracle=%.12g\n",
                                     type, N, K, pattern, wpr, wpb, row, actual[row], reference[row], oracle[row]);
                        return 1;
                    }
                }
                ++count;
            }
            if (dpacked) CUDA_CHECK(cudaFree(dpacked));
            CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(qx));
        }
    }
    std::printf("PASS: %d Q4_K/Q6_K tests, bit-identical to upstream and within unchanged double-oracle envelope; max abs error %.9g\n", count, max_error);
}
