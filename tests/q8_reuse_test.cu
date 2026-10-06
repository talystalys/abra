// Regression tests for reuse ownership, alias invalidation and upstream math.
#include "common.cuh"
#include "mmvq.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
void quantize_row_q4_K_ref(const float *, block_q4_K *, int64_t);
void quantize_row_q6_K_ref(const float *, block_q6_K *, int64_t);
}

static ggml_tensor tensor(ggml_type type, int columns, int rows, void * data) {
    ggml_tensor result{};
    result.type = type;
    result.ne[0] = columns; result.ne[1] = rows; result.ne[2] = result.ne[3] = 1;
    result.nb[0] = ggml_type_size(type);
    result.nb[1] = result.nb[0] * columns / ggml_blck_size(type);
    result.nb[2] = result.nb[1] * rows; result.nb[3] = result.nb[2];
    result.data = data;
    result.flags = GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

int main() {
    if (!std::getenv("LEGACY_AI_REUSE_Q8") || std::strcmp(std::getenv("LEGACY_AI_REUSE_Q8"), "1")) return 2;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, CC %d.%d, PCI %04x:%02x:%02x.0\n", prop.name,
                prop.major, prop.minor, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (std::strcmp(prop.name, "NVIDIA GeForce GTX 1080 Ti") || prop.major != 6 || prop.minor != 1) return 2;
    constexpr int K = 2560, N = 33;
    ggml_backend_cuda_context ctx(0);
    std::mt19937 random(20261012);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> weights(K * N), first(K), second(K), hybrid(K), actual(N);
    for (auto & value : weights) value = distribution(random);
    std::vector<block_q4_K> q4(K * N / QK_K);
    std::vector<block_q6_K> q6(K * N / QK_K);
    quantize_row_q4_K_ref(weights.data(), q4.data(), weights.size());
    quantize_row_q6_K_ref(weights.data(), q6.data(), weights.size());
    auto * b4 = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0), q4.size() * sizeof(block_q4_K));
    auto * b6 = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0), q6.size() * sizeof(block_q6_K));
    if (!b4 || !b6) return 3;
    ggml_backend_buffer_set_usage(b4, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(b6, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tensor w4 = tensor(GGML_TYPE_Q4_K, K, N, ggml_backend_buffer_get_base(b4)); w4.buffer = b4;
    ggml_tensor w6 = tensor(GGML_TYPE_Q6_K, K, N, ggml_backend_buffer_get_base(b6)); w6.buffer = b6;
    CUDA_CHECK(cudaMemcpy(w4.data, q4.data(), q4.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w6.data, q6.data(), q6.size() * sizeof(block_q6_K), cudaMemcpyHostToDevice));
    float * dx, * dy;
    CUDA_CHECK(cudaMalloc(&dx, K * sizeof(float))); CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    ggml_tensor x = tensor(GGML_TYPE_F32, K, 1, dx);
    ggml_tensor y = tensor(GGML_TYPE_F32, N, 1, dy); y.op = GGML_OP_MUL_MAT;
    int checks = 0;
    auto compute = [&](const ggml_tensor & w, const ggml_tensor & input) {
        ggml_cuda_mul_mat_vec_q(ctx, &w, &input, nullptr, &y);
        // The backend uses a nonblocking stream. A default-stream host copy
        // alone does not wait for that stream's kernels to finish.
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaMemcpy(actual.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : actual) if (!std::isfinite(value)) std::abort();
        return actual;
    };
    auto check = [&](const ggml_tensor & w, const ggml_tensor & input, const std::vector<float> & expected) {
        const auto got = compute(w, input);
        if (std::memcmp(got.data(), expected.data(), N * sizeof(float))) {
            std::fprintf(stderr, "FAIL: activation reuse changed an upstream output at check %d\n", checks);
            std::abort();
        }
        ++checks;
    };
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (int i = 0; i < K; ++i) {
            first[i] = pattern == 0 ? 0.0f : distribution(random) * (pattern == 2 ? 100.0f : 1.0f);
            second[i] = distribution(random);
        }
        hybrid = first;
        std::copy(second.begin() + K / 2, second.end(), hybrid.begin() + K / 2);
        std::vector<std::vector<float>> expected;
        for (const auto * input : {&first, &second, &hybrid}) {
            CUDA_CHECK(cudaMemcpy(dx, input->data(), K * sizeof(float), cudaMemcpyHostToDevice));
            expected.push_back(compute(w4, x)); expected.push_back(compute(w6, x));
        }
        CUDA_CHECK(cudaMemcpy(dx, first.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        {
            legacy_ai_q8_reuse_scope scope(ctx, true);
            check(w4, x, expected[0]); check(w6, x, expected[1]); // shared Q4/Q6 bytes
            scope.invalidate_write(&y); // independent destination leaves the cache valid
            check(w4, x, expected[0]);
            ggml_tensor view = x; view.op = GGML_OP_VIEW;
            scope.invalidate_write(&view); // a view is not a write
            check(w6, x, expected[1]);
            CUDA_CHECK(cudaMemcpy(dx + K / 2, second.data() + K / 2, K / 2 * sizeof(float), cudaMemcpyHostToDevice));
            ggml_tensor partial = tensor(GGML_TYPE_F32, K / 2, 1, dx + K / 2); partial.op = GGML_OP_CPY;
            scope.invalidate_write(&partial); // partial alias must invalidate
            check(w4, x, expected[4]); check(w6, x, expected[5]);
            CUDA_CHECK(cudaMemcpy(dx, second.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            ggml_tensor alias = x; alias.op = GGML_OP_CPY;
            scope.invalidate_write(&alias); // different descriptor, same storage
            check(w4, x, expected[2]); check(w6, x, expected[3]);
            ggml_tensor different_descriptor = x;
            check(w4, different_descriptor, expected[2]); // conservative identity fallback
            check(w6, different_descriptor, expected[3]);
        }
        CUDA_CHECK(cudaMemcpy(dx, first.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        {
            legacy_ai_q8_reuse_scope next_graph(ctx, true);
            check(w4, x, expected[0]); check(w6, x, expected[1]); // no cross-graph reuse
        }
        {
            legacy_ai_q8_reuse_scope disabled(ctx, false);
            check(w4, x, expected[0]); check(w6, x, expected[1]); // capture/concurrency fallback
        }
    }
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    ggml_backend_buffer_free(b4); ggml_backend_buffer_free(b6);
    std::printf("PASS: %d activation reuse outputs bit-identical to upstream; aliases, graph reset and fallback covered\n", checks);
    return 0;
}
