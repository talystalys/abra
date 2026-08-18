#include "smoke/smoke.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace legacy_ai;

void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --target-arch <sm_61|sm_75|pascal|turing> : Target specific GPU architecture\n"
              << "  --device <index>                         : Run on specific CUDA device index\n"
              << "  --pci <pci_bus_id>                       : Match device by PCI bus ID\n"
              << "  --uuid <uuid_str>                        : Match device by UUID\n"
              << "  --all-devices                            : Run test suite on all CUDA-visible devices\n"
              << "  --block-size <threads>                   : CUDA block size (default: 256)\n"
              << "  --help                                   : Display this help message\n";
}

int main(int argc, char **argv) {
    std::string target_spec = "";
    bool run_all = false;
    int block_size = 256;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--target-arch" && i + 1 < argc) {
            target_spec = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            target_spec = argv[++i];
        } else if (arg == "--pci" && i + 1 < argc) {
            target_spec = argv[++i];
        } else if (arg == "--uuid" && i + 1 < argc) {
            target_spec = argv[++i];
        } else if (arg == "--block-size" && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
        } else if (arg == "--all-devices") {
            run_all = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    auto all_devices = query_all_devices();
    if (all_devices.empty()) {
        std::cerr << "ERROR: No CUDA devices detected on host.\n";
        return 1;
    }

    print_device_table(all_devices);

    std::vector<int> target_devices;
    if (run_all) {
        for (const auto &d : all_devices) target_devices.push_back(d.index);
    } else if (!target_spec.empty()) {
        int dev_idx = find_device_by_spec(target_spec);
        if (dev_idx < 0) {
            std::cerr << "BLOCKER: Requested target GPU '" << target_spec << "' was not found among visible CUDA devices.\n";
            std::cerr << "Silent substitution is strictly prohibited.\n";
            return 2;
        }
        target_devices.push_back(dev_idx);
    } else {
        // Default to device 0 if nothing specified
        target_devices.push_back(0);
    }

    // Required test dimensions from specification
    const std::vector<size_t> test_dimensions = {
        1,
        31,
        32,
        33,
        255,
        256,
        257,
        1024,
        65536
    };

    bool global_pass = true;

    for (int dev_idx : target_devices) {
        GpuDevice dev = get_device_info(dev_idx);
        std::cout << "============================================================\n";
        std::cout << "TEST RUN FOR DEVICE " << dev.index << "\n";
        std::cout << "GPU: " << dev.name << "\n";
        std::cout << "CC:  " << dev.compute_capability_str() << "\n";
        std::cout << "PCI: " << dev.pci_bus_id << "\n";
        std::cout << "UUID: " << dev.uuid << "\n";
        std::cout << "Block Size: " << block_size << " threads/block\n";
        std::cout << "============================================================\n";

        bool device_pass = true;
        std::cout << std::left
                  << std::setw(12) << "Dimension"
                  << std::setw(16) << "Divisible(B)"
                  << std::setw(12) << "Mismatches"
                  << "Status\n";
        std::cout << std::string(55, '-') << "\n";

        for (size_t dim : test_dimensions) {
            bool is_div = (dim % block_size == 0);
            SmokeTestResult res = execute_smoke_test(dev.index, dim, block_size);

            std::cout << std::left
                      << std::setw(12) << dim
                      << std::setw(16) << (is_div ? "YES" : "NO")
                      << std::setw(12) << res.mismatches
                      << (res.success ? "[PASS]" : "[FAIL]") << "\n";

            if (!res.success) {
                device_pass = false;
                global_pass = false;
                std::cerr << "  Error detail:\n" << res.error_message;
            }
        }

        std::cout << std::string(55, '-') << "\n";
        std::cout << "DEVICE " << dev.index << " (" << dev.name << ", CC " << dev.compute_capability_str()
                  << ") RESULT: " << (device_pass ? "PASS" : "FAIL") << "\n\n";
    }

    std::cout << "============================================================\n";
    std::cout << "FINAL TEST SUITE OUTCOME: " << (global_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "============================================================\n";

    return global_pass ? 0 : 1;
}
