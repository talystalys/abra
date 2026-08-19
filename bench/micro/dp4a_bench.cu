#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <climits>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "kernels/dp4a/dp4a_gemv.cuh"
#include "smoke/smoke.h"

using namespace legacy_ai;
using namespace legacy_ai::dp4a;

// CPU Oracle Reference for validation
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
            fprintf(stderr, "FATAL: CPU Oracle overflow at row %d: %lld outside int32 range!\n", n, (long long)acc64);
            exit(1);
        }
        y[n] = static_cast<int32_t>(acc64);
    }
}

struct ShapeSpec {
    std::string name;
    int N;
    int K;
    bool is_required_qwen;
};

struct BenchmarkResult {
    std::string shape_name;
    int N;
    int K;
    std::string variant_name;
    int variant_id;
    int warmup_iterations;
    int measured_iterations;
    std::vector<double> latencies_us; // microseconds
    double median_us;
    double mean_us;
    double min_us;
    double max_us;
    double stddev_us;
    double spread_us;
    double gmac_per_sec;
    double effective_gops;
    double effective_throughput_gbps;
    bool verified_correct;
};

void fill_random_data(std::vector<int8_t>& W, std::vector<int8_t>& x, int N, int K, uint32_t seed) {
    W.resize((size_t)N * K);
    x.resize(K);
    uint32_t state = seed;
    auto lcg = [&state]() -> int8_t {
        state = state * 1664525u + 1013904223u;
        int val = (int)(state >> 26) - 32; // [-32, 31]
        return static_cast<int8_t>(val);
    };
    for (size_t i = 0; i < W.size(); ++i) W[i] = lcg();
    for (int k = 0; k < K; ++k) x[k] = lcg();
}

BenchmarkResult run_microbenchmark(
    KernelVariant variant,
    const ShapeSpec& shape,
    int warmup_reps = 10,
    int measured_reps = 100
) {
    int N = shape.N;
    int K = shape.K;

    std::vector<int8_t> h_W;
    std::vector<int8_t> h_x;
    fill_random_data(h_W, h_x, N, K, 1337 + N + K);

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

    // Warmup
    for (int i = 0; i < warmup_reps; ++i) {
        launch_gemv(variant, d_W, d_x, d_y, N, K, 0);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measured repetitions with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> latencies_us(measured_reps, 0.0);

    for (int r = 0; r < measured_reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        launch_gemv(variant, d_W, d_x, d_y, N, K, 0);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        latencies_us[r] = static_cast<double>(ms) * 1000.0; // convert ms to us
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Verify bit-for-bit correctness after timing
    std::vector<int32_t> h_y_gpu(N, 0);
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int n = 0; n < N; ++n) {
        if (h_y_gpu[n] != h_y_ref[n]) {
            correct = false;
            fprintf(stderr, "Correctness FAIL in benchmark %s (%s) at row %d: GPU=%d, CPU=%d\n",
                    shape.name.c_str(), get_variant_name(variant), n, h_y_gpu[n], h_y_ref[n]);
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Compute stats
    std::vector<double> sorted_lat = latencies_us;
    std::sort(sorted_lat.begin(), sorted_lat.end());

    double min_val = sorted_lat.front();
    double max_val = sorted_lat.back();
    double sum = 0.0;
    for (double v : sorted_lat) sum += v;
    double mean_val = sum / measured_reps;

    double median_val = (measured_reps % 2 == 1)
        ? sorted_lat[measured_reps / 2]
        : (sorted_lat[measured_reps / 2 - 1] + sorted_lat[measured_reps / 2]) / 2.0;

    double var = 0.0;
    if (measured_reps > 1) {
        for (double v : sorted_lat) var += (v - mean_val) * (v - mean_val);
        var /= (measured_reps - 1); // Sample SD (N-1)
    }
    double stddev_val = std::sqrt(var);

    // Metrics (based on median latency)
    double median_sec = median_val * 1e-6;
    double total_macs = static_cast<double>(N) * static_cast<double>(K);
    double gmac_per_sec = (total_macs / median_sec) / 1e9;
    double effective_gops = (2.0 * total_macs / median_sec) / 1e9; // 1 MAC = 2 ops (1 mul + 1 add)

    // Effective bytes: W (N*K int8) + x (K int8) + y (N int32)
    double effective_bytes = static_cast<double>(N * K) + static_cast<double>(K) + static_cast<double>(N * 4);
    double effective_throughput_gbps = (effective_bytes / median_sec) / (1024.0 * 1024.0 * 1024.0);

    BenchmarkResult res;
    res.shape_name = shape.name;
    res.N = N;
    res.K = K;
    res.variant_name = get_variant_name(variant);
    res.variant_id = static_cast<int>(variant);
    res.warmup_iterations = warmup_reps;
    res.measured_iterations = measured_reps;
    res.latencies_us = latencies_us;
    res.median_us = median_val;
    res.mean_us = mean_val;
    res.min_us = min_val;
    res.max_us = max_val;
    res.stddev_us = stddev_val;
    res.spread_us = max_val - min_val;
    res.gmac_per_sec = gmac_per_sec;
    res.effective_gops = effective_gops;
    res.effective_throughput_gbps = effective_throughput_gbps;
    res.verified_correct = correct;

    return res;
}

void print_json_results(
    const std::vector<BenchmarkResult>& results,
    const std::string& out_filepath,
    const GpuDevice& dev
) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"benchmark\": \"dp4a_gemv_microbenchmark\",\n";
    ss << "  \"device\": {\n";
    ss << "    \"name\": \"" << dev.name << "\",\n";
    ss << "    \"compute_capability\": \"" << dev.compute_capability_str() << "\",\n";
    ss << "    \"pci_bus_id\": \"" << dev.pci_bus_id << "\",\n";
    ss << "    \"uuid\": \"" << dev.uuid << "\",\n";
    ss << "    \"total_vram_bytes\": " << dev.total_memory_bytes << "\n";
    ss << "  },\n";
    ss << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "    {\n";
        ss << "      \"shape_name\": \"" << r.shape_name << "\",\n";
        ss << "      \"N\": " << r.N << ",\n";
        ss << "      \"K\": " << r.K << ",\n";
        ss << "      \"variant_name\": \"" << r.variant_name << "\",\n";
        ss << "      \"variant_id\": " << r.variant_id << ",\n";
        ss << "      \"warmup_iterations\": " << r.warmup_iterations << ",\n";
        ss << "      \"measured_iterations\": " << r.measured_iterations << ",\n";
        ss << "      \"median_us\": " << r.median_us << ",\n";
        ss << "      \"mean_us\": " << r.mean_us << ",\n";
        ss << "      \"min_us\": " << r.min_us << ",\n";
        ss << "      \"max_us\": " << r.max_us << ",\n";
        ss << "      \"stddev_us\": " << r.stddev_us << ",\n";
        ss << "      \"spread_us\": " << r.spread_us << ",\n";
        ss << "      \"gmac_per_sec\": " << r.gmac_per_sec << ",\n";
        ss << "      \"effective_gops\": " << r.effective_gops << ",\n";
        ss << "      \"effective_throughput_gbps\": " << r.effective_throughput_gbps << ",\n";
        ss << "      \"verified_correct\": " << (r.verified_correct ? "true" : "false") << ",\n";
        ss << "      \"latencies_us\": [";
        for (size_t j = 0; j < r.latencies_us.size(); ++j) {
            ss << r.latencies_us[j] << (j + 1 < r.latencies_us.size() ? ", " : "");
        }
        ss << "]\n";
        ss << "    }" << (i + 1 < results.size() ? ",\n" : "\n");
    }
    ss << "  ]\n";
    ss << "}\n";

    if (!out_filepath.empty()) {
        std::ofstream ofs(out_filepath);
        if (ofs.is_open()) {
            ofs << ss.str();
            ofs.close();
            printf("Saved benchmark results to: %s\n", out_filepath.c_str());
        } else {
            fprintf(stderr, "Failed to open output file: %s\n", out_filepath.c_str());
        }
    }
    printf("%s", ss.str().c_str());
}

void print_bench_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [out_file] [warmup_reps] [measured_reps] [options]\n"
              << "Options:\n"
              << "  --arch <pascal|turing>    : Assert expected architecture preset\n"
              << "  --assert-name <str>       : Assert exact GPU name substring\n"
              << "  --assert-cc <major.minor> : Assert compute capability\n"
              << "  --assert-pci <pci_id>     : Assert PCI bus ID\n"
              << "  --assert-uuid <uuid_str>  : Assert GPU UUID\n"
              << "  --help                    : Show help\n";
}

int main(int argc, char** argv) {
    std::string out_file = "";
    int warmup_reps = 10;
    int measured_reps = 100;

    std::string assert_name = "";
    int assert_cc_major = 0;
    int assert_cc_minor = 0;
    std::string assert_pci = "";
    std::string assert_uuid = "";

    int positional_idx = 0;
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
            print_bench_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            if (positional_idx == 0) out_file = arg;
            else if (positional_idx == 1) warmup_reps = std::atoi(arg.c_str());
            else if (positional_idx == 2) measured_reps = std::atoi(arg.c_str());
            positional_idx++;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_bench_usage(argv[0]);
            return 1;
        }
    }

    int active_dev_idx = 0;
    CUDA_CHECK(cudaGetDevice(&active_dev_idx));
    GpuDevice dev = get_device_info(active_dev_idx);

    std::cout << "============================================================\n";
    std::cout << "Milestone M2 DP4A Microbenchmark Suite\n";
    std::cout << "Device: " << dev.name << "\n";
    std::cout << "Compute Capability: " << dev.compute_capability_str() << "\n";
    std::cout << "PCI Bus ID: " << dev.pci_bus_id << "\n";
    std::cout << "UUID: " << dev.uuid << "\n";
    std::cout << "Warmup Reps: " << warmup_reps << " | Measured Reps: " << measured_reps << "\n";
    std::cout << "============================================================\n\n";

    // Strict Device Identity Assertion
    std::string identity_err;
    if (!verify_device_identity(dev, assert_name, assert_cc_major, assert_cc_minor, assert_pci, assert_uuid, identity_err)) {
        std::cerr << "BLOCKER: Physical GPU assertion failure on active device " << dev.index << ":\n" << identity_err;
        std::cerr << "Refusing benchmark execution due to device identity mismatch.\n";
        return 2;
    }
    if (!assert_name.empty() || assert_cc_major > 0 || !assert_pci.empty() || !assert_uuid.empty()) {
        std::cout << "PASS: Physical GPU identity assertions verified successfully.\n\n";
    }

    const std::vector<ShapeSpec> shapes = {
        // Required Real Qwen3-4B Shapes
        { "attn_q",      4096, 2560, true },
        { "attn_kv",     1024, 2560, true },
        { "attn_out",    2560, 4096, true },
        { "ffn_gate_up", 9728, 2560, true },
        { "ffn_down",    2560, 9728, true },
        // Secondary Synthetic Scaling Shapes
        { "synth_256",    256,  256, false },
        { "synth_1024",  1024, 1024, false },
        { "synth_4096",  4096, 4096, false }
    };

    const std::vector<KernelVariant> variants = {
        KernelVariant::SCALAR,
        KernelVariant::DP4A_BASIC,
        KernelVariant::DP4A_OPT,
        KernelVariant::DP4A_SHARED
    };

    std::vector<BenchmarkResult> all_results;

    for (const auto& shape : shapes) {
        printf("--- Benchmarking Shape: %-12s [N=%5d, K=%5d] ---\n", shape.name.c_str(), shape.N, shape.K);
        for (auto var : variants) {
            BenchmarkResult res = run_microbenchmark(var, shape, warmup_reps, measured_reps);
            if (!res.verified_correct) {
                fprintf(stderr, "FATAL: Correctness verification failed for %s on %s!\n",
                        shape.name.c_str(), get_variant_name(var));
                return 1;
            }
            printf("  %-42s: %8.2f us | %7.2f GMAC/s | %6.2f GiB/s\n",
                   res.variant_name.c_str(), res.median_us, res.gmac_per_sec, res.effective_throughput_gbps);
            all_results.push_back(res);
        }
        printf("\n");
    }

    print_json_results(all_results, out_file, dev);
    return 0;
}
