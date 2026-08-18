#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <climits>
#include "kernels/dp4a/dp4a_gemv.cuh"

using namespace legacy_ai::dp4a;

// -----------------------------------------------------------------------------
// CPU Reference Oracle
// Mathematical accumulation using int64_t with explicit int32 range assertion
// -----------------------------------------------------------------------------
void cpu_gemv(
    const int8_t* W,
    const int8_t* x,
    int32_t* y,
    int N, int K
) {
    for (int n = 0; n < N; ++n) {
        int64_t acc64 = 0;
        const int8_t* row_W = W + (size_t)n * K;
        for (int k = 0; k < K; ++k) {
            acc64 += static_cast<int64_t>(row_W[k]) * static_cast<int64_t>(x[k]);
        }
        if (acc64 < INT32_MIN || acc64 > INT32_MAX) {
            fprintf(stderr, "FATAL: CPU Oracle integer overflow at row %d: %lld outside int32 range!\n", n, (long long)acc64);
            exit(1);
        }
        y[n] = static_cast<int32_t>(acc64);
    }
}

// -----------------------------------------------------------------------------
// Device Identification Helper
// -----------------------------------------------------------------------------
void verify_device_identity() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    char pci_bus_id[64];
    cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device);

    printf("============================================================\n");
    printf("DP4A Microkernel Test Suite\n");
    printf("Device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("PCI Bus ID: %s\n", pci_bus_id);
    printf("Total Global Memory: %.2f GiB (%zu bytes)\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.totalGlobalMem);
    printf("============================================================\n\n");
}

// -----------------------------------------------------------------------------
// Deterministic Input Generation
// -----------------------------------------------------------------------------
enum class InputFamily {
    ALL_ZERO,
    ALL_ONES,
    ALTERNATING,
    RANDOM_BOUNDED,
    EXTREMES_BOUNDED,
    LANE_PATTERNS
};

const char* get_family_name(InputFamily fam) {
    switch (fam) {
        case InputFamily::ALL_ZERO: return "ALL_ZERO";
        case InputFamily::ALL_ONES: return "ALL_ONES";
        case InputFamily::ALTERNATING: return "ALTERNATING";
        case InputFamily::RANDOM_BOUNDED: return "RANDOM_BOUNDED";
        case InputFamily::EXTREMES_BOUNDED: return "EXTREMES_BOUNDED";
        case InputFamily::LANE_PATTERNS: return "LANE_PATTERNS";
        default: return "UNKNOWN";
    }
}

void fill_inputs(
    InputFamily fam,
    std::vector<int8_t>& W,
    std::vector<int8_t>& x,
    int N, int K,
    uint32_t seed = 42
) {
    W.resize((size_t)N * K);
    x.resize(K);

    switch (fam) {
        case InputFamily::ALL_ZERO:
            std::fill(W.begin(), W.end(), 0);
            std::fill(x.begin(), x.end(), 0);
            break;

        case InputFamily::ALL_ONES:
            std::fill(W.begin(), W.end(), 1);
            std::fill(x.begin(), x.end(), 1);
            break;

        case InputFamily::ALTERNATING:
            for (size_t i = 0; i < W.size(); ++i) {
                W[i] = (i % 2 == 0) ? 1 : -1;
            }
            for (int k = 0; k < K; ++k) {
                x[k] = (k % 2 == 0) ? 1 : -1;
            }
            break;

        case InputFamily::RANDOM_BOUNDED: {
            uint32_t state = seed;
            auto lcg = [&state]() -> int8_t {
                state = state * 1664525u + 1013904223u;
                // Output values in [-32, 31] to guarantee no int32 overflow for large K (up to 100,000)
                int val = (int)(state >> 26) - 32;
                return static_cast<int8_t>(val);
            };
            for (size_t i = 0; i < W.size(); ++i) W[i] = lcg();
            for (int k = 0; k < K; ++k) x[k] = lcg();
            break;
        }

        case InputFamily::EXTREMES_BOUNDED: {
            // Mix of -128 and +127 with balanced zero padding
            uint32_t state = seed;
            for (size_t i = 0; i < W.size(); ++i) {
                state = state * 1664525u + 1013904223u;
                int r = state % 5;
                if (r == 0) W[i] = 127;
                else if (r == 1) W[i] = -128;
                else if (r == 2) W[i] = 64;
                else if (r == 3) W[i] = -64;
                else W[i] = 0;
            }
            for (int k = 0; k < K; ++k) {
                state = state * 1664525u + 1013904223u;
                int r = state % 5;
                if (r == 0) x[k] = 1;
                else if (r == 1) x[k] = -1;
                else if (r == 2) x[k] = 2;
                else if (r == 3) x[k] = -2;
                else x[k] = 0;
            }
            break;
        }

        case InputFamily::LANE_PATTERNS: {
            // Exercise 4 distinct lanes: e.g. lane 0=1, lane 1=2, lane 2=3, lane 3=-4
            const int8_t pat_w[4] = { 1, -2, 3, -4 };
            const int8_t pat_x[4] = { 4, 3, -2, 1 };
            for (size_t i = 0; i < W.size(); ++i) {
                W[i] = pat_w[i % 4];
            }
            for (int k = 0; k < K; ++k) {
                x[k] = pat_x[k % 4];
            }
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Test Case Runner
// -----------------------------------------------------------------------------
bool run_test_case(
    KernelVariant variant,
    InputFamily fam,
    int N, int K,
    bool verbose = false
) {
    std::vector<int8_t> h_W;
    std::vector<int8_t> h_x;
    fill_inputs(fam, h_W, h_x, N, K, 42 + N + K);

    std::vector<int32_t> h_y_ref(N, 0);
    cpu_gemv(h_W.data(), h_x.data(), h_y_ref.data(), N, K);

    int8_t* d_W = nullptr;
    int8_t* d_x = nullptr;
    int32_t* d_y = nullptr;

    size_t w_bytes = (size_t)N * K * sizeof(int8_t);
    size_t x_bytes = (size_t)K * sizeof(int8_t);
    size_t y_bytes = (size_t)N * sizeof(int32_t);

    cudaMalloc(&d_W, w_bytes);
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_y, y_bytes);

    cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, y_bytes);

    cudaError_t err = launch_gemv(variant, d_W, d_x, d_y, N, K);
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s (Variant: %s, N=%d, K=%d)\n",
                cudaGetErrorString(err), get_variant_name(variant), N, K);
        cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
        return false;
    }

    cudaDeviceSynchronize();

    std::vector<int32_t> h_y_gpu(N, 0);
    cudaMemcpy(h_y_gpu.data(), d_y, y_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);

    // Exact bit-for-bit comparison (0 tolerance)
    int mismatches = 0;
    for (int n = 0; n < N; ++n) {
        if (h_y_gpu[n] != h_y_ref[n]) {
            if (mismatches < 5) {
                fprintf(stderr, "MISMATCH at row %d: GPU=%d vs CPU_REF=%d (Variant=%s, Family=%s, N=%d, K=%d)\n",
                        n, h_y_gpu[n], h_y_ref[n], get_variant_name(variant), get_family_name(fam), N, K);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        fprintf(stderr, "FAILED: %d/%d mismatches for Variant=%s, Family=%s, N=%d, K=%d\n",
                mismatches, N, get_variant_name(variant), get_family_name(fam), N, K);
        return false;
    }

    if (verbose) {
        printf("PASS: Variant=%s | Family=%-16s | N=%5d | K=%5d\n",
               get_variant_name(variant), get_family_name(fam), N, K);
    }
    return true;
}

int main() {
    verify_device_identity();

    const std::vector<KernelVariant> variants = {
        KernelVariant::SCALAR,
        KernelVariant::DP4A_BASIC,
        KernelVariant::DP4A_OPT,
        KernelVariant::DP4A_SHARED
    };

    const std::vector<InputFamily> families = {
        InputFamily::ALL_ZERO,
        InputFamily::ALL_ONES,
        InputFamily::ALTERNATING,
        InputFamily::RANDOM_BOUNDED,
        InputFamily::EXTREMES_BOUNDED,
        InputFamily::LANE_PATTERNS
    };

    // Mandatory K edge cases (handles scalar tail 0..3)
    const std::vector<int> k_edges = {
        1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 1024, 2560, 4096, 9728
    };

    // Mandatory N boundary cases
    const std::vector<int> n_boundaries = {
        1, 2, 31, 32, 33, 63, 64, 65, 128, 256, 1024, 2560, 4096, 9728
    };

    // Mandatory Qwen3-4B real-model shapes
    struct ModelShape { const char* name; int N; int K; };
    const std::vector<ModelShape> qwen_shapes = {
        { "attn_q",      4096, 2560 },
        { "attn_kv",     1024, 2560 },
        { "attn_out",    2560, 4096 },
        { "ffn_gate_up", 9728, 2560 },
        { "ffn_down",    2560, 9728 }
    };

    int total_tests = 0;
    int passed_tests = 0;

    printf("=== Phase 1: Input Families x Variant Smoke Tests (N=128, K=256) ===\n");
    for (auto var : variants) {
        for (auto fam : families) {
            total_tests++;
            if (run_test_case(var, fam, 128, 256, true)) {
                passed_tests++;
            }
        }
    }
    printf("\n");

    printf("=== Phase 2: K-Edge Coverage Suite (N=64, various K) ===\n");
    for (auto var : variants) {
        for (int k : k_edges) {
            total_tests++;
            if (run_test_case(var, InputFamily::RANDOM_BOUNDED, 64, k, false)) {
                passed_tests++;
            }
        }
    }
    printf("K-edge coverage tests passed: %zu K values across %zu variants.\n\n", k_edges.size(), variants.size());

    printf("=== Phase 3: N-Boundary Coverage Suite (K=256, various N) ===\n");
    for (auto var : variants) {
        for (int n : n_boundaries) {
            total_tests++;
            if (run_test_case(var, InputFamily::LANE_PATTERNS, n, 256, false)) {
                passed_tests++;
            }
        }
    }
    printf("N-boundary tests passed: %zu N values across %zu variants.\n\n", n_boundaries.size(), variants.size());

    printf("=== Phase 4: Qwen3-4B Real Model Shapes Coverage Suite ===\n");
    for (auto var : variants) {
        for (const auto& shape : qwen_shapes) {
            total_tests++;
            if (run_test_case(var, InputFamily::RANDOM_BOUNDED, shape.N, shape.K, true)) {
                passed_tests++;
            }
        }
    }
    printf("\n");

    printf("============================================================\n");
    printf("DP4A Test Suite Summary: %d / %d Tests PASSED (100%% bit-exact)\n", passed_tests, total_tests);
    printf("============================================================\n");

    if (passed_tests != total_tests) {
        fprintf(stderr, "FATAL: Test suite encountered %d failures!\n", total_tests - passed_tests);
        return 1;
    }
    return 0;
}
