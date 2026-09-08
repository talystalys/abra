#include "kernels/q4k/q4k_family.cuh"
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char ** argv) {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    const std::vector<Q4KCandidateID> & candidates = get_all_candidate_ids();

    for (int d = 0; d < dev_count; ++d) {
        cudaSetDevice(d);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);

        std::cout << "====================================================================================================================\n";
        std::cout << "Device " << d << ": " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ", PCI " << prop.pciBusID << ")\n";
        std::cout << "Max Resident Warps / SM: " << prop.maxThreadsPerMultiProcessor / 32
                  << " | Max Threads / Block: " << prop.maxThreadsPerBlock
                  << " | Total Regs / SM: " << prop.regsPerMultiprocessor << "\n";
        std::cout << "====================================================================================================================\n";

        std::cout << std::left << std::setw(30) << "Candidate Name"
                  << std::setw(8)  << "WPR"
                  << std::setw(8)  << "WPB"
                  << std::setw(10) << "Block Dim"
                  << std::setw(10) << "Threads"
                  << std::setw(8)  << "Regs"
                  << std::setw(10) << "SMem(B)"
                  << std::setw(10) << "Blks/SM"
                  << std::setw(12) << "Warps/SM"
                  << std::setw(12) << "Max Warps"
                  << std::setw(14) << "Theoretical %"
                  << "Spills\n";
        std::cout << std::string(132, '-') << "\n";

        for (auto c : candidates) {
            Q4KCandidateSpec spec = get_candidate_spec(c);
            Q4KOccupancyInfo info = get_q4k_candidate_occupancy_info(c, d);
            std::string dim_str = std::to_string(info.block_dim_x) + "x" + std::to_string(info.block_dim_y);

            std::cout << std::left << std::setw(30) << spec.name
                      << std::setw(8)  << spec.warps_per_row
                      << std::setw(8)  << spec.warps_per_block
                      << std::setw(10) << dim_str
                      << std::setw(10) << info.threads_per_block
                      << std::setw(8)  << info.regs_per_thread
                      << std::setw(10) << info.static_shared_mem_bytes
                      << std::setw(10) << info.max_active_blocks_per_sm
                      << std::setw(12) << info.resident_warps_per_sm
                      << std::setw(12) << info.max_warps_per_sm
                      << std::fixed << std::setprecision(1) << std::setw(14) << info.theoretical_occupancy_pct
                      << info.spill_stores_bytes << " B\n";
        }
        std::cout << std::string(132, '-') << "\n\n";
    }

    return 0;
}
