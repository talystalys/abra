#include "smoke/smoke.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace legacy_ai {

__global__ void smoke_vector_kernel(const int32_t * __restrict__ a,
                                    const int32_t * __restrict__ b,
                                    int32_t * __restrict__ c,
                                    size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (a[idx] * 3) + (b[idx] * 5) + 42;
    }
}

std::vector<GpuDevice> query_all_devices() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::vector<GpuDevice> devices;
    devices.reserve(count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        GpuDevice dev;
        dev.index = i;
        dev.name = prop.name;
        dev.cc_major = prop.major;
        dev.cc_minor = prop.minor;
        dev.total_memory_bytes = prop.totalGlobalMem;
        dev.multi_processor_count = prop.multiProcessorCount;
        dev.clock_rate_khz = prop.clockRate;
        dev.memory_bus_width = prop.memoryBusWidth;

        char pci_bus_str[32];
        snprintf(pci_bus_str, sizeof(pci_bus_str), "%04x:%02x:%02x.0",
                 prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        dev.pci_bus_id = pci_bus_str;

        char uuid_str[64];
        const unsigned char *u = reinterpret_cast<const unsigned char *>(prop.uuid.bytes);
        snprintf(uuid_str, sizeof(uuid_str),
                 "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                 u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
        dev.uuid = uuid_str;

        devices.push_back(dev);
    }
    return devices;
}

void print_device_table(const std::vector<GpuDevice> &devices) {
    std::cout << "\n=== CUDA Device Enumeration ===\n";
    std::cout << std::left
              << std::setw(6)  << "Index"
              << std::setw(32) << "Name"
              << std::setw(10) << "CC"
              << std::setw(16) << "VRAM (MiB)"
              << std::setw(18) << "PCI Bus ID"
              << "UUID" << "\n";
    std::cout << std::string(130, '-') << "\n";

    for (const auto &dev : devices) {
        double vram_mib = static_cast<double>(dev.total_memory_bytes) / (1024.0 * 1024.0);
        std::cout << std::left
                  << std::setw(6)  << dev.index
                  << std::setw(32) << dev.name
                  << std::setw(10) << dev.compute_capability_str()
                  << std::setw(16) << std::fixed << std::setprecision(0) << vram_mib
                  << std::setw(18) << dev.pci_bus_id
                  << dev.uuid << "\n";
    }
    std::cout << std::string(130, '-') << "\n" << std::endl;
}

GpuDevice get_device_info(int device_index) {
    auto devs = query_all_devices();
    if (device_index < 0 || device_index >= static_cast<int>(devs.size())) {
        std::cerr << "ERROR: Invalid device index " << device_index << " (count: " << devs.size() << ")\n";
        exit(1);
    }
    return devs[device_index];
}

int find_device_by_uuid(const std::string &uuid) {
    auto devs = query_all_devices();
    for (const auto &d : devs) {
        if (d.uuid == uuid || d.uuid.find(uuid) != std::string::npos) {
            return d.index;
        }
    }
    return -1;
}

int find_device_by_pci(const std::string &pci) {
    auto devs = query_all_devices();
    for (const auto &d : devs) {
        if (d.pci_bus_id == pci || d.pci_bus_id.find(pci) != std::string::npos) {
            return d.index;
        }
    }
    return -1;
}

bool verify_device_identity(const GpuDevice &dev,
                            const std::string &expected_name,
                            int expected_cc_major,
                            int expected_cc_minor,
                            const std::string &expected_pci,
                            const std::string &expected_uuid,
                            std::string &error_msg) {
    std::ostringstream oss;
    bool match = true;

    if (!expected_name.empty() && dev.name.find(expected_name) == std::string::npos) {
        oss << "Device name mismatch: expected '" << expected_name << "', got '" << dev.name << "'\n";
        match = false;
    }
    if (expected_cc_major > 0 && (dev.cc_major != expected_cc_major || dev.cc_minor != expected_cc_minor)) {
        oss << "Compute capability mismatch: expected " << expected_cc_major << "." << expected_cc_minor
            << ", got " << dev.cc_major << "." << dev.cc_minor << "\n";
        match = false;
    }
    if (!expected_pci.empty() && dev.pci_bus_id.find(expected_pci) == std::string::npos) {
        oss << "PCI ID mismatch: expected '" << expected_pci << "', got '" << dev.pci_bus_id << "'\n";
        match = false;
    }
    if (!expected_uuid.empty() && dev.uuid.find(expected_uuid) == std::string::npos) {
        oss << "UUID mismatch: expected '" << expected_uuid << "', got '" << dev.uuid << "'\n";
        match = false;
    }

    if (!match) {
        error_msg = oss.str();
    }
    return match;
}

int find_device_by_spec(const std::string &spec) {
    auto devs = query_all_devices();
    if (devs.empty()) {
        std::cerr << "ERROR: No CUDA-capable devices found.\n";
        return -1;
    }

    // Check if numeric index
    char *endptr = nullptr;
    long idx = strtol(spec.c_str(), &endptr, 10);
    if (*endptr == '\0' && idx >= 0 && idx < static_cast<long>(devs.size())) {
        return static_cast<int>(idx);
    }

    // UUID match
    int by_uuid = find_device_by_uuid(spec);
    if (by_uuid >= 0) return by_uuid;

    // PCI match
    int by_pci = find_device_by_pci(spec);
    if (by_pci >= 0) return by_pci;

    // Check target arch strings
    if (spec == "sm_61" || spec == "6.1" || spec == "pascal" || spec == "Pascal") {
        for (const auto &d : devs) {
            if (d.cc_major == 6 && d.cc_minor == 1) return d.index;
        }
    }
    if (spec == "sm_75" || spec == "7.5" || spec == "turing" || spec == "Turing") {
        for (const auto &d : devs) {
            if (d.cc_major == 7 && d.cc_minor == 5) return d.index;
        }
    }

    // Substring match on name
    for (const auto &d : devs) {
        if (d.name.find(spec) != std::string::npos) {
            return d.index;
        }
    }

    return -1;
}

void generate_deterministic_inputs(int32_t *a, int32_t *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<int32_t>((i * 17 + 5) % 10007);
        b[i] = static_cast<int32_t>((i * 31 + 13) % 10007);
    }
}

int32_t cpu_oracle_eval(int32_t a, int32_t b) {
    return (a * 3) + (b * 5) + 42;
}

void compute_cpu_oracle(const int32_t *a, const int32_t *b, int32_t *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = cpu_oracle_eval(a[i], b[i]);
    }
}

void launch_smoke_kernel(const int32_t *d_a, const int32_t *d_b, int32_t *d_c, size_t n, int block_size, cudaStream_t stream) {
    size_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size == 0) grid_size = 1;
    smoke_vector_kernel<<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0, stream>>>(d_a, d_b, d_c, n);
}

SmokeTestResult execute_smoke_test(int device_index, size_t n, int block_size) {
    SmokeTestResult res;
    res.success = false;
    res.device_index = device_index;
    res.dimension = n;
    res.mismatches = 0;

    CUDA_CHECK(cudaSetDevice(device_index));
    auto dev_info = get_device_info(device_index);
    res.device_name = dev_info.name;
    res.compute_capability = dev_info.compute_capability_str();

    std::vector<int32_t> h_a(n);
    std::vector<int32_t> h_b(n);
    std::vector<int32_t> h_c_cpu(n);
    std::vector<int32_t> h_c_gpu(n, 0);

    generate_deterministic_inputs(h_a.data(), h_b.data(), n);
    compute_cpu_oracle(h_a.data(), h_b.data(), h_c_cpu.data(), n);

    int32_t *d_a = nullptr;
    int32_t *d_b = nullptr;
    int32_t *d_c = nullptr;
    size_t bytes = n * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    launch_smoke_kernel(d_a, d_b, d_c, n, block_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    for (size_t i = 0; i < n; ++i) {
        if (h_c_gpu[i] != h_c_cpu[i]) {
            res.mismatches++;
            if (res.mismatches <= 5) {
                std::ostringstream oss;
                oss << "Mismatch at index " << i << ": GPU=" << h_c_gpu[i] << " vs CPU=" << h_c_cpu[i] << " (inputs a=" << h_a[i] << ", b=" << h_b[i] << ")\n";
                res.error_message += oss.str();
            }
        }
    }

    if (res.mismatches == 0) {
        res.success = true;
    }

    return res;
}

} // namespace legacy_ai
