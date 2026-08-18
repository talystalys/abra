#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "BigBoy Hardware & Toolchain Environment Inventory"
echo "Captured on: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo

run_cmd() {
    local title="$1"
    shift
    echo "--- [ ${title} ] ---"
    "$@" 2>&1 || true
    echo
}

run_cmd "Hostname" hostname
run_cmd "Working Directory" pwd
run_cmd "OS Release" cat /etc/os-release
run_cmd "Kernel" uname -a
run_cmd "CPU Architecture" lscpu
run_cmd "Memory" free -h
run_cmd "NUMA Topology" numactl --hardware
run_cmd "NVIDIA SMI Status" nvidia-smi
run_cmd "NVIDIA GPU List" nvidia-smi -L
run_cmd "NVIDIA GPU Query (CSV)" nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,driver_version,power.limit --format=csv
run_cmd "PCI Device Identification" sh -c "lspci -nn | grep -i nvidia"
run_cmd "CUDA Compiler (NVCC)" /usr/local/cuda-12.9/bin/nvcc --version
run_cmd "C Compiler (GCC)" gcc --version
run_cmd "C++ Compiler (G++)" g++ --version
run_cmd "Build System (CMake)" cmake --version
run_cmd "Build System (Ninja)" sh -c "ninja --version 2>/dev/null || ninja-build --version 2>/dev/null"
run_cmd "Version Control (Git)" git --version
run_cmd "C Standard Library (ldd / glibc)" ldd --version

echo "============================================================"
echo "Inventory Collection Complete"
echo "============================================================"
