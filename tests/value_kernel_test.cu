#include "common.cuh"
#include "mmvf.cuh"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include "kernels/q4k/legacy-ai-attention.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s CC %d.%d PCI %04x:%02x:%02x.0\n", prop.name, prop.major, prop.minor, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (std::strcmp(prop.name, "NVIDIA GeForce GTX 1080 Ti") || prop.major != 6 || prop.minor != 1) return 2;
    ggml_backend_cuda_context ctx(0);
    std::mt19937 random(12026);
    std::uniform_real_distribution<float> distribution(-1, 1);
    int tests = 0;
    double max_error = 0;
    for (int K : {256, 512, 2048, 2304}) for (int padding : {0, 8}) for (int pattern : {0, 1, 2}) {
        const int row_stride = K + padding, head_stride = row_stride * 130;
        std::vector<half> weights(head_stride * 8);
        std::vector<float> input(K * 32), upstream(128 * 32), actual(128 * 32);
        for (auto &w : weights) w = __float2half_rn(pattern ? distribution(random) * (pattern == 2 ? 100 : 1) : 0);
        for (auto &x : input) x = pattern ? distribution(random) : 0;
        half *dw;
        float *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dw, weights.size() * 2));
        CUDA_CHECK(cudaMalloc(&dx, input.size() * 4));
        CUDA_CHECK(cudaMalloc(&dy, actual.size() * 4));
        CUDA_CHECK(cudaMemcpy(dw, weights.data(), weights.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
        ggml_tensor w{}, x{}, y{};
        w.type = GGML_TYPE_F16; x.type = y.type = GGML_TYPE_F32;
        w.data = dw; x.data = dx; y.data = dy;
        w.ne[0] = x.ne[0] = K; w.ne[1] = y.ne[0] = 128;
        x.ne[1] = y.ne[1] = 1; w.ne[2] = 8; x.ne[2] = y.ne[2] = 32;
        w.ne[3] = x.ne[3] = y.ne[3] = 1;
        w.nb[0] = 2; w.nb[1] = row_stride * 2; w.nb[2] = head_stride * 2; w.nb[3] = weights.size() * 2;
        x.nb[0] = y.nb[0] = 4; x.nb[1] = x.nb[2] = K * 4; x.nb[3] = input.size() * 4;
        y.nb[1] = y.nb[2] = 128 * 4; y.nb[3] = actual.size() * 4;
        ggml_cuda_mul_mat_vec_f(ctx, &w, &x, nullptr, &y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaMemcpy(upstream.data(), dy, actual.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<double> oracle(actual.size());
        for (int head = 0; head < 32; ++head) for (int row = 0; row < 128; ++row) {
            double total = 0;
            for (int k = 0; k < K; ++k)
                total += double(__half2float(weights[head / 4 * head_stride + row * row_stride + k])) * input[head * K + k];
            oracle[head * 128 + row] = total;
        }
        const int warps = K == 256 ? 4 : 8;
        for (int rows : {2, 4}) {
            const dim3 grid(128 / rows, 32), block(32, rows * warps);
            #define TEST_VALUE(R, W) legacy_ai_attention_value<R, W><<<grid, block>>>(dw, dx, dy, K, row_stride, head_stride, K, 128)
            if (rows == 2 && warps == 4) { TEST_VALUE(2, 4); }
            else if (rows == 2) { TEST_VALUE(2, 8); }
            else if (warps == 4) { TEST_VALUE(4, 4); }
            else { TEST_VALUE(4, 8); }
            #undef TEST_VALUE
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(actual.data(), dy, actual.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < int(actual.size()); ++i) {
                double error = std::abs(double(actual[i]) - oracle[i]);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual[i]) || error > 0.001 + 0.0001 * std::abs(oracle[i]) || std::memcmp(&actual[i], &upstream[i], 4)) {
                    std::fprintf(stderr, "FAIL K=%d padding=%d pattern=%d rows=%d index=%d got=%.9g upstream=%.9g oracle=%.12g\n", K, padding, pattern, rows, i, actual[i], upstream[i], oracle[i]);
                    return 1;
                }
            }
            ++tests;
        }
        CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
    std::printf("PASS: %d value cases, bit-identical to pinned upstream; double-oracle max abs error %.9g\n", tests, max_error);
}
