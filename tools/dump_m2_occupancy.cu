#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>
#include "kernels/dp4a/dp4a_gemv.cuh"
#include "smoke/smoke.h"

using namespace abra;
using namespace abra::dp4a;

int main(int argc, char** argv) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    const std::vector<KernelVariant> variants = {
        KernelVariant::SCALAR,
        KernelVariant::DP4A_BASIC,
        KernelVariant::DP4A_OPT,
        KernelVariant::DP4A_SHARED
    };

    for (int d = 0; d < device_count; ++d) {
        GpuDevice dev = get_device_info(d);
        cudaSetDevice(d);

        std::cout << "================================================================================\n";
        std::cout << "GPU " << d << ": " << dev.name << " (CC " << dev.compute_capability_str()
                  << ", PCI " << dev.pci_bus_id << ", UUID " << dev.uuid << ")\n";
        std::cout << "================================================================================\n";
        std::cout << std::left
                  << std::setw(32) << "Variant Name"
                  << std::setw(12) << "Block Dim"
                  << std::setw(12) << "Thds/Blk"
                  << std::setw(10) << "Regs/Thd"
                  << std::setw(10) << "SMem(B)"
                  << std::setw(10) << "Blks/SM"
                  << std::setw(12) << "Warps/SM"
                  << "Occupancy(%)\n";
        std::cout << std::string(98, '-') << "\n";

        for (auto v : variants) {
            auto occ = get_kernel_occupancy_info(v, d);
            std::string bdim = std::to_string(occ.block_dim_x) + "x" + std::to_string(occ.block_dim_y);
            std::string warps = std::to_string(occ.resident_warps_per_sm) + " / " + std::to_string(occ.max_warps_per_sm);
            std::cout << std::left
                      << std::setw(32) << occ.variant_name
                      << std::setw(12) << bdim
                      << std::setw(12) << occ.threads_per_block
                      << std::setw(10) << occ.registers_per_thread
                      << std::setw(10) << occ.static_shared_bytes
                      << std::setw(10) << occ.active_blocks_per_sm
                      << std::setw(12) << warps
                      << std::fixed << std::setprecision(1) << occ.theoretical_occupancy_pct << "%\n";
        }
        std::cout << "\n";
    }

    return 0;
}
