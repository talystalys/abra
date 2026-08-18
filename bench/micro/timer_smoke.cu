#include "smoke/smoke.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace legacy_ai;

int main(int argc, char **argv) {
    std::string target_spec = "";
    int iterations = 50;
    int warmup_iterations = 10;
    size_t n = 1048576; // 1M elements
    int block_size = 256;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--target-arch" && i + 1 < argc) {
            target_spec = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            target_spec = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            n = std::stoull(argv[++i]);
        }
    }

    auto all_devices = query_all_devices();
    if (all_devices.empty()) {
        std::cerr << "ERROR: No CUDA devices detected on host.\n";
        return 1;
    }

    int dev_idx = 0;
    if (!target_spec.empty()) {
        dev_idx = find_device_by_spec(target_spec);
        if (dev_idx < 0) {
            std::cerr << "BLOCKER: Requested target GPU '" << target_spec << "' not found.\n";
            return 2;
        }
    }

    CUDA_CHECK(cudaSetDevice(dev_idx));
    GpuDevice dev = get_device_info(dev_idx);

    std::cout << "\n============================================================\n";
    std::cout << "HARNESS VALIDATION ONLY\n";
    std::cout << "NOT A PERFORMANCE RESULT\n";
    std::cout << "============================================================\n";
    std::cout << "Selected GPU: " << dev.name << " (Index: " << dev.index << ")\n";
    std::cout << "Compute Capability: " << dev.compute_capability_str() << "\n";
    std::cout << "PCI Bus ID: " << dev.pci_bus_id << "\n";
    std::cout << "UUID: " << dev.uuid << "\n";
    std::cout << "Workload: " << n << " int32 elements (" << (n * sizeof(int32_t) * 3) / (1024.0 * 1024.0) << " MiB active data)\n";
    std::cout << "Warm-up Iterations: " << warmup_iterations << "\n";
    std::cout << "Measured Iterations: " << iterations << "\n";
    std::cout << "============================================================\n\n";

    // Allocate host and device memory
    std::vector<int32_t> h_a(n), h_b(n), h_c(n);
    generate_deterministic_inputs(h_a.data(), h_b.data(), n);

    int32_t *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = n * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // CUDA Events for timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warm-up phase
    std::cout << "Executing " << warmup_iterations << " warm-up iterations...\n";
    for (int i = 0; i < warmup_iterations; ++i) {
        launch_smoke_kernel(d_a, d_b, d_c, n, block_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Warm-up completed.\n\n";

    // Measurement phase
    std::vector<float> raw_times_ms;
    raw_times_ms.reserve(iterations);

    std::cout << "Recording " << iterations << " timing iterations (CUDA event measurement):\n";
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch_smoke_kernel(d_a, d_b, d_c, n, block_size);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        raw_times_ms.push_back(ms);
    }

    // Copy back and verify correctness
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h_c[i] != cpu_oracle_eval(h_a[i], h_b[i])) {
            mismatches++;
        }
    }

    // Statistical summary
    std::vector<float> sorted_times = raw_times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    float min_time = sorted_times.front();
    float max_time = sorted_times.back();
    float median_time = sorted_times[sorted_times.size() / 2];
    double sum = std::accumulate(raw_times_ms.begin(), raw_times_ms.end(), 0.0);
    double mean_time = sum / raw_times_ms.size();

    double sq_sum = 0.0;
    for (float t : raw_times_ms) {
        sq_sum += (t - mean_time) * (t - mean_time);
    }
    double stddev = std::sqrt(sq_sum / raw_times_ms.size());

    std::cout << "\n=== Raw Measurements (ms) ===\n";
    for (size_t i = 0; i < raw_times_ms.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << raw_times_ms[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n";
    }
    if (raw_times_ms.size() % 10 != 0) std::cout << "\n";

    std::cout << "\n=== Timing Statistics ===\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Min:        " << min_time << " ms\n";
    std::cout << "Median:     " << median_time << " ms\n";
    std::cout << "Mean:       " << mean_time << " ms\n";
    std::cout << "Max:        " << max_time << " ms\n";
    std::cout << "Spread:     " << (max_time - min_time) << " ms\n";
    std::cout << "Std Dev:    " << stddev << " ms\n";
    std::cout << "Correctness Mismatches: " << mismatches << " (expected 0)\n";

    std::cout << "\n============================================================\n";
    std::cout << "HARNESS VALIDATION COMPLETE — " << (mismatches == 0 ? "SUCCESS" : "FAIL") << "\n";
    std::cout << "============================================================\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return (mismatches == 0) ? 0 : 1;
}
