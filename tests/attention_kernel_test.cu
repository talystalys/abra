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

int main(int argc, char **argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s CC %d.%d PCI %04x:%02x:%02x.0\n", prop.name, prop.major, prop.minor, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (std::strcmp(prop.name, "NVIDIA GeForce GTX 1080 Ti") || prop.major != 6 || prop.minor != 1) return 2;
    ggml_backend_cuda_context ctx(0);
    std::mt19937 random(12026);
    std::uniform_real_distribution<float> distribution(-1, 1);
    int tests = 0;
    double max_error = 0;
    for (int N : {1, 33, 256, 2048, 2176}) for (int padding : {0, 8}) for (int pattern : {0, 1, 2}) {
        const int row_stride = 128 + padding, head_stride = row_stride * (N + 2);
        std::vector<half> weights(head_stride * 8);
        std::vector<float> input(128 * 32), upstream(N * 32), actual(N * 32);
        for (auto &w : weights) w = __float2half_rn(pattern ? distribution(random) * (pattern == 2 ? 100 : 1) : 0);
        for (auto &x : input) x = pattern ? distribution(random) : 0;
        half *dw;
        float *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dw, weights.size() * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dx, input.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, actual.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dw, weights.data(), weights.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        ggml_tensor w{}, x{}, y{};
        w.type = GGML_TYPE_F16; x.type = y.type = GGML_TYPE_F32;
        w.data = dw; x.data = dx; y.data = dy;
        w.ne[0] = x.ne[0] = 128; w.ne[1] = y.ne[0] = N;
        x.ne[1] = y.ne[1] = 1; w.ne[2] = 8; x.ne[2] = y.ne[2] = 32;
        w.ne[3] = x.ne[3] = y.ne[3] = 1;
        w.nb[0] = 2; w.nb[1] = row_stride * 2; w.nb[2] = head_stride * 2; w.nb[3] = weights.size() * 2;
        x.nb[0] = y.nb[0] = 4; x.nb[1] = x.nb[2] = 128 * 4; x.nb[3] = input.size() * 4;
        y.nb[1] = y.nb[2] = N * 4; y.nb[3] = actual.size() * 4;
        ggml_cuda_mul_mat_vec_f(ctx, &w, &x, nullptr, &y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaMemcpy(upstream.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int rows : {1, 4, 8}) {
            const dim3 grid((N + rows - 1) / rows, 32), block(32, rows * 2);
            if (rows == 1) legacy_ai_attention_qk<1><<<grid, block>>>(dw, dx, dy, N, row_stride, head_stride, 128, N);
            else if (rows == 4) legacy_ai_attention_qk<4><<<grid, block>>>(dw, dx, dy, N, row_stride, head_stride, 128, N);
            else legacy_ai_attention_qk<8><<<grid, block>>>(dw, dx, dy, N, row_stride, head_stride, 128, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int head = 0; head < 32; ++head) for (int row = 0; row < N; ++row) {
                double oracle = 0;
                for (int k = 0; k < 128; ++k)
                    oracle += double(__half2float(weights[head / 4 * head_stride + row * row_stride + k])) * input[head * 128 + k];
                const int i = head * N + row;
                const double error = std::abs(double(actual[i]) - oracle);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual[i]) || error > 0.001 + 0.0001 * std::abs(oracle) || std::memcmp(&actual[i], &upstream[i], 4)) {
                    std::fprintf(stderr, "FAIL N=%d padding=%d pattern=%d rows=%d head=%d row=%d got=%.9g upstream=%.9g oracle=%.12g\n", N, padding, pattern, rows, head, row, actual[i], upstream[i], oracle);
                    return 1;
                }
            }
            ++tests;
        }
        if (benchmark && N == 2048 && !padding && pattern == 1) {
            for (int rows : {0, 1, 4, 8}) {
                auto launch = [&] {
                    if (!rows) ggml_cuda_mul_mat_vec_f(ctx, &w, &x, nullptr, &y);
                    else if (rows == 1) legacy_ai_attention_qk<1><<<dim3(N, 32), dim3(32, 2), 0, ctx.stream()>>>(dw, dx, dy, N, row_stride, head_stride, 128, N);
                    else if (rows == 4) legacy_ai_attention_qk<4><<<dim3((N + 3) / 4, 32), dim3(32, 8), 0, ctx.stream()>>>(dw, dx, dy, N, row_stride, head_stride, 128, N);
                    else legacy_ai_attention_qk<8><<<dim3((N + 7) / 8, 32), dim3(32, 16), 0, ctx.stream()>>>(dw, dx, dy, N, row_stride, head_stride, 128, N);
                };
                for (int warmup = 0; warmup < 10; ++warmup) launch();
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
                std::printf("MICRO_ROWS_%d_MS=[", rows);
                for (int sample = 0; sample < 50; ++sample) {
                    CUDA_CHECK(cudaEventRecord(start, ctx.stream()));
                    launch();
                    CUDA_CHECK(cudaEventRecord(stop, ctx.stream()));
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    float ms;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                    std::printf("%s%.9g", sample ? "," : "", ms);
                }
                std::printf("]\n");
                CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
            }
        }
        CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
    std::printf("PASS: %d attention cases, bit-identical to pinned upstream; double-oracle max abs error %.9g\n", tests, max_error);
}
