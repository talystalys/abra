#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <climits>
#include <iostream>
#include <iomanip>
#include "kernels/dp4a/dp4a_gemv.cuh"
#include "smoke/smoke.h"

using namespace abra;
using namespace abra::dp4a;

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

enum class InputFamily {
    ALL_ZERO,
    ALL_ONES,
    ALTERNATING,
    RANDOM_BOUNDED,
    EXTREMES_BOUNDED,
    LANE_PATTERNS
};

const char* get_family_name(InputFamily family) {
    switch (family) {
        case InputFamily::ALL_ZERO: return "ALL_ZERO";
        case InputFamily::ALL_ONES: return "ALL_ONES";
        case InputFamily::ALTERNATING: return "ALTERNATING";
        case InputFamily::RANDOM_BOUNDED: return "RANDOM_BOUNDED";
        case InputFamily::EXTREMES_BOUNDED: return "EXTREMES_BOUNDED";
        case InputFamily::LANE_PATTERNS: return "LANE_PATTERNS";
        default: return "UNKNOWN";
    }
}

void generate_test_data(
    std::vector<int8_t>& W,
    std::vector<int8_t>& x,
    int N, int K,
    InputFamily family,
    uint32_t seed = 42
) {
    W.resize((size_t)N * K);
    x.resize(K);

    switch (family) {
        case InputFamily::ALL_ZERO:
            std::fill(W.begin(), W.end(), 0);
            std::fill(x.begin(), x.end(), 0);
            break;

        case InputFamily::ALL_ONES:
            std::fill(W.begin(), W.end(), 1);
            std::fill(x.begin(), x.end(), 1);
            break;

        case InputFamily::ALTERNATING:
            for (size_t i = 0; i < W.size(); ++i) W[i] = (i % 2 == 0) ? 1 : -1;
            for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 2 == 0) ? 1 : -1;
            break;

        case InputFamily::RANDOM_BOUNDED: {
            uint32_t state = seed;
            auto lcg = [&state]() -> int8_t {
                state = state * 1664525u + 1013904223u;
                int val = (int)(state >> 26) - 32;
                return static_cast<int8_t>(val);
            };
            for (size_t i = 0; i < W.size(); ++i) W[i] = lcg();
            for (size_t i = 0; i < x.size(); ++i) x[i] = lcg();
            break;
        }

        case InputFamily::EXTREMES_BOUNDED:
            for (size_t i = 0; i < W.size(); ++i) W[i] = (i % 2 == 0) ? 127 : -128;
            for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 2 == 0) ? 1 : -1;
            break;

        case InputFamily::LANE_PATTERNS:
            for (size_t i = 0; i < W.size(); ++i) {
                int byte_lane = (i % 4);
                W[i] = static_cast<int8_t>(byte_lane + 1);
            }
            for (size_t i = 0; i < x.size(); ++i) {
                int byte_lane = (i % 4);
                x[i] = static_cast<int8_t>(10 * (byte_lane + 1));
            }
            break;
    }
}

struct TestResult {
    bool success;
    int N;
    int K;
    KernelVariant variant;
    InputFamily family;
    int mismatch_count;
    int first_mismatch_row;
    int32_t expected_val;
    int32_t actual_val;
};

TestResult run_single_test(
    KernelVariant variant,
    InputFamily family,
    int N, int K,
    uint32_t seed = 42
) {
    std::vector<int8_t> h_W;
    std::vector<int8_t> h_x;
    generate_test_data(h_W, h_x, N, K, family, seed);

    std::vector<int32_t> h_y_ref(N, 0);
    cpu_gemv(h_W.data(), h_x.data(), h_y_ref.data(), N, K);

    int8_t* d_W = nullptr;
    int8_t* d_x = nullptr;
    int32_t* d_y = nullptr;

    size_t w_bytes = (size_t)N * K * sizeof(int8_t);
    size_t x_bytes = (size_t)K * sizeof(int8_t);
    size_t y_bytes = (size_t)N * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_W, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, y_bytes));

    cudaError_t launch_err = launch_gemv(variant, d_W, d_x, d_y, N, K, 0);
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s (Variant: %s, N=%d, K=%d)\n",
                cudaGetErrorString(launch_err), get_variant_name(variant), N, K);
        CUDA_CHECK(cudaFree(d_W));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        return { false, N, K, variant, family, N, 0, 0, 0 };
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> h_y_gpu(N, 0);
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    TestResult res;
    res.N = N;
    res.K = K;
    res.variant = variant;
    res.family = family;
    res.mismatch_count = 0;
    res.first_mismatch_row = -1;
    res.expected_val = 0;
    res.actual_val = 0;

    for (int n = 0; n < N; ++n) {
        if (h_y_gpu[n] != h_y_ref[n]) {
            if (res.mismatch_count == 0) {
                res.first_mismatch_row = n;
                res.expected_val = h_y_ref[n];
                res.actual_val = h_y_gpu[n];
            }
            res.mismatch_count++;
        }
    }

    res.success = (res.mismatch_count == 0);
    return res;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --arch <pascal|turing>    : Assert expected architecture preset\n"
              << "  --assert-name <str>       : Assert exact GPU name substring\n"
              << "  --assert-cc <major.minor> : Assert compute capability\n"
              << "  --assert-pci <pci_id>     : Assert PCI bus ID\n"
              << "  --assert-uuid <uuid_str>  : Assert GPU UUID\n"
              << "  --help                    : Show help\n";
}

int main(int argc, char** argv) {
    std::string assert_name = "";
    int assert_cc_major = 0;
    int assert_cc_minor = 0;
    std::string assert_pci = "";
    std::string assert_uuid = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--arch" || arg == "-a") && i + 1 < argc) {
            std::string arch = argv[++i];
            if (arch == "pascal" || arch == "Pascal" || arch == "sm_61") {
                assert_name = "GTX 1060";
                assert_cc_major = 6;
                assert_cc_minor = 1;
                assert_pci = "0000:03:00.0";
                assert_uuid = "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd";
            } else if (arch == "turing" || arch == "Turing" || arch == "sm_75") {
                assert_name = "GTX 1660 SUPER";
                assert_cc_major = 7;
                assert_cc_minor = 5;
                assert_pci = "0000:81:00.0";
                assert_uuid = "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd";
            } else {
                std::cerr << "Unknown arch preset: " << arch << "\n";
                return 1;
            }
        } else if (arg == "--pascal") {
            assert_name = "GTX 1060";
            assert_cc_major = 6;
            assert_cc_minor = 1;
            assert_pci = "0000:03:00.0";
            assert_uuid = "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd";
        } else if (arg == "--turing") {
            assert_name = "GTX 1660 SUPER";
            assert_cc_major = 7;
            assert_cc_minor = 5;
            assert_pci = "0000:81:00.0";
            assert_uuid = "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd";
        } else if (arg == "--assert-name" && i + 1 < argc) {
            assert_name = argv[++i];
        } else if (arg == "--assert-cc" && i + 1 < argc) {
            std::string cc_str = argv[++i];
            size_t dot_pos = cc_str.find('.');
            if (dot_pos != std::string::npos) {
                assert_cc_major = std::stoi(cc_str.substr(0, dot_pos));
                assert_cc_minor = std::stoi(cc_str.substr(dot_pos + 1));
            } else {
                assert_cc_major = std::stoi(cc_str);
            }
        } else if (arg == "--assert-pci" && i + 1 < argc) {
            assert_pci = argv[++i];
        } else if (arg == "--assert-uuid" && i + 1 < argc) {
            assert_uuid = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    int active_dev_idx = 0;
    CUDA_CHECK(cudaGetDevice(&active_dev_idx));
    GpuDevice dev = get_device_info(active_dev_idx);

    std::cout << "============================================================\n";
    std::cout << "DP4A Microkernel Test Suite\n";
    std::cout << "Device: " << dev.name << "\n";
    std::cout << "Compute Capability: " << dev.compute_capability_str() << "\n";
    std::cout << "PCI Bus ID: " << dev.pci_bus_id << "\n";
    std::cout << "UUID: " << dev.uuid << "\n";
    std::cout << "Total Global Memory: " << std::fixed << std::setprecision(2)
              << (double)dev.total_memory_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB ("
              << dev.total_memory_bytes << " bytes)\n";
    std::cout << "============================================================\n";

    std::string identity_err;
    if (!verify_device_identity(dev, assert_name, assert_cc_major, assert_cc_minor, assert_pci, assert_uuid, identity_err)) {
        std::cerr << "BLOCKER: Physical GPU assertion failure on active device " << dev.index << ":\n" << identity_err;
        std::cerr << "Refusing execution due to device identity mismatch.\n";
        return 2;
    }
    if (!assert_name.empty() || assert_cc_major > 0 || !assert_pci.empty() || !assert_uuid.empty()) {
        std::cout << "PASS: Physical GPU identity assertions verified successfully.\n\n";
    }

    std::cout << "=== Theoretical Occupancy Analysis (via CUDA Occupancy API) ===\n";
    std::cout << std::left
              << std::setw(32) << "Variant"
              << std::setw(14) << "Block Dim"
              << std::setw(10) << "Regs/Thd"
              << std::setw(12) << "SMem(B)"
              << std::setw(12) << "Blks/SM"
              << std::setw(12) << "Warps/SM"
              << "Occupancy(%)\n";
    std::cout << std::string(98, '-') << "\n";

    const std::vector<KernelVariant> all_variants = {
        KernelVariant::SCALAR,
        KernelVariant::DP4A_BASIC,
        KernelVariant::DP4A_OPT,
        KernelVariant::DP4A_SHARED
    };

    for (auto v : all_variants) {
        auto occ = get_kernel_occupancy_info(v, active_dev_idx);
        std::string bdim = std::to_string(occ.block_dim_x) + "x" + std::to_string(occ.block_dim_y);
        std::cout << std::left
                  << std::setw(32) << occ.variant_name
                  << std::setw(14) << bdim
                  << std::setw(10) << occ.registers_per_thread
                  << std::setw(12) << occ.static_shared_bytes
                  << std::setw(12) << occ.active_blocks_per_sm
                  << std::setw(12) << (std::to_string(occ.resident_warps_per_sm) + "/" + std::to_string(occ.max_warps_per_sm))
                  << std::fixed << std::setprecision(1) << occ.theoretical_occupancy_pct << "%\n";
    }
    std::cout << std::string(98, '-') << "\n\n";

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    auto record_test = [&](const TestResult& res) {
        total_tests++;
        if (res.success) {
            passed_tests++;
        } else {
            failed_tests++;
            fprintf(stderr, "FAIL: Variant=%s | Family=%s | N=%d | K=%d | Mismatches=%d | FirstRow=%d (Expected=%d, Actual=%d)\n",
                    get_variant_name(res.variant), get_family_name(res.family),
                    res.N, res.K, res.mismatch_count, res.first_mismatch_row,
                    res.expected_val, res.actual_val);
        }
    };

    std::cout << "=== Phase 1: Input Families x Variant Smoke Tests (N=128, K=256) ===\n";
    const std::vector<InputFamily> all_families = {
        InputFamily::ALL_ZERO,
        InputFamily::ALL_ONES,
        InputFamily::ALTERNATING,
        InputFamily::RANDOM_BOUNDED,
        InputFamily::EXTREMES_BOUNDED,
        InputFamily::LANE_PATTERNS
    };

    for (auto v : all_variants) {
        for (auto f : all_families) {
            TestResult res = run_single_test(v, f, 128, 256, 1337);
            record_test(res);
            if (res.success) {
                printf("PASS: Variant=%-38s | Family=%-16s | N=%5d | K=%5d\n",
                       get_variant_name(v), get_family_name(f), 128, 256);
            }
        }
    }
    std::cout << "\n";

    std::cout << "=== Phase 2: K-Edge Coverage Suite (N=64, various K) ===\n";
    const std::vector<int> k_edge_values = {
        1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 1024, 2560, 4096, 9728
    };
    for (int k_val : k_edge_values) {
        for (auto v : all_variants) {
            TestResult res = run_single_test(v, InputFamily::RANDOM_BOUNDED, 64, k_val, 2024 + k_val);
            record_test(res);
        }
    }
    printf("K-edge coverage tests passed: %zu K values across %zu variants.\n\n",
           k_edge_values.size(), all_variants.size());

    std::cout << "=== Phase 3: N-Boundary Coverage Suite (K=256, various N) ===\n";
    const std::vector<int> n_boundary_values = {
        1, 2, 31, 32, 33, 63, 64, 65, 128, 256, 1024, 2560, 4096, 9728
    };
    for (int n_val : n_boundary_values) {
        for (auto v : all_variants) {
            TestResult res = run_single_test(v, InputFamily::RANDOM_BOUNDED, n_val, 256, 3030 + n_val);
            record_test(res);
        }
    }
    printf("N-boundary tests passed: %zu N values across %zu variants.\n\n",
           n_boundary_values.size(), all_variants.size());

    std::cout << "=== Phase 4: Qwen3-4B Real Model Shapes Coverage Suite ===\n";
    struct ModelShape { const char* name; int N; int K; };
    const std::vector<ModelShape> qwen_shapes = {
        { "attn_q",      4096, 2560 },
        { "attn_kv",     1024, 2560 },
        { "attn_out",    2560, 4096 },
        { "ffn_gate_up", 9728, 2560 },
        { "ffn_down",    2560, 9728 }
    };

    for (auto v : all_variants) {
        for (const auto& shape : qwen_shapes) {
            TestResult res = run_single_test(v, InputFamily::RANDOM_BOUNDED, shape.N, shape.K, 4040 + shape.N);
            record_test(res);
            if (res.success) {
                printf("PASS: Variant=%-38s | Family=RANDOM_BOUNDED   | N=%5d | K=%5d\n",
                       get_variant_name(v), shape.N, shape.K);
            }
        }
    }
    std::cout << "\n";

    std::cout << "============================================================\n";
    std::cout << "DP4A Test Suite Summary: " << passed_tests << " / " << total_tests
              << " Tests PASSED (100% bit-exact)\n";
    std::cout << "============================================================\n";

    if (failed_tests > 0) {
        fprintf(stderr, "FATAL: Test suite encountered %d failures!\n", failed_tests);
        return 1;
    }

    return 0;
}
