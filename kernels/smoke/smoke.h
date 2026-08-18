#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d - %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err), cudaGetErrorName(_err)); \
        exit(1); \
    } \
} while (0)

namespace legacy_ai {

struct GpuDevice {
    int index;
    std::string name;
    int cc_major;
    int cc_minor;
    size_t total_memory_bytes;
    std::string pci_bus_id;
    std::string uuid;
    int multi_processor_count;
    int clock_rate_khz;
    int memory_bus_width;

    std::string compute_capability_str() const {
        return std::to_string(cc_major) + "." + std::to_string(cc_minor);
    }
};

// Device querying and management
std::vector<GpuDevice> query_all_devices();
void print_device_table(const std::vector<GpuDevice> &devices);
GpuDevice get_device_info(int device_index);
int find_device_by_spec(const std::string &spec);

// CPU Oracle and Deterministic Input Generation
void generate_deterministic_inputs(int32_t *a, int32_t *b, size_t n);
int32_t cpu_oracle_eval(int32_t a, int32_t b);
void compute_cpu_oracle(const int32_t *a, const int32_t *b, int32_t *c, size_t n);

// GPU Smoke Kernel Launch and Validation
struct SmokeTestResult {
    bool success;
    int device_index;
    std::string device_name;
    std::string compute_capability;
    size_t dimension;
    size_t mismatches;
    std::string error_message;
};

SmokeTestResult execute_smoke_test(int device_index, size_t n, int block_size = 256);

// Raw Kernel invocation for timing harness
void launch_smoke_kernel(const int32_t *d_a, const int32_t *d_b, int32_t *d_c, size_t n, int block_size = 256, cudaStream_t stream = 0);

} // namespace legacy_ai
