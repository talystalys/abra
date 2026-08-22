#!/usr/bin/env python3
import json
import os
import sys

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench/results/m1"
    analysis_file = os.path.join(results_dir, "analysis-summary.json")
    manifest_file = os.path.join(results_dir, "manifest.json")
    
    if os.path.isfile(analysis_file):
        with open(analysis_file, "r") as f:
            analysis_data = json.load(f)
    else:
        analysis_data = {}
        
    manifest = {
        "manifest_version": "1.0",
        "milestone": "M1",
        "title": "llama.cpp Existing Inference Baseline on NVIDIA Pascal & Turing",
        "date_utc": "2026-08-18",
        "environment": {
            "host": "BigBoy",
            "os": "Debian GNU/Linux 13 (trixie) / Debian 13.6",
            "kernel": "Linux BigBoy 6.12.100+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100-1 (2026-07-30) x86_64",
            "cpu": "2 x Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 physical cores, 64 logical threads)",
            "ram_gib": 62,
            "numa_nodes": 2
        },
        "toolchain": {
            "nvidia_driver": "550.163.01",
            "cuda_toolkit": "12.9.2 (/usr/local/cuda-12.9)",
            "nvcc_version": "Cuda compilation tools, release 12.9, V12.9.86 (Build cuda_12.9.r12.9/compiler.36037853_0)",
            "host_compiler": "GCC 14.2.0 (Debian 14.2.0-19)",
            "c_library": "glibc 2.41 (Debian GLIBC 2.41-12+deb13u3)"
        },
        "gpus": {
            "primary_qualification_pascal": {
                "name": "NVIDIA GeForce GTX 1060 6GB",
                "architecture": "Pascal",
                "compute_capability": "6.1",
                "pci_bus_id": "0000:03:00.0",
                "uuid": "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd",
                "vram_total_bytes": 6360793088,
                "vram_usable_mib": 6065,
                "power_limit_w": 120.0,
                "numa_node": 0
            },
            "secondary_control_turing": {
                "name": "NVIDIA GeForce GTX 1660 SUPER",
                "architecture": "Turing",
                "compute_capability": "7.5",
                "pci_bus_id": "0000:81:00.0",
                "uuid": "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd",
                "vram_total_bytes": 6218055680,
                "vram_usable_mib": 5929,
                "power_limit_w": 125.0,
                "numa_node": 1
            }
        },
        "pinned_runtime": {
            "repository": "https://github.com/ggml-org/llama.cpp.git",
            "commit": "169e4a7ff201e666c2325bcd35afb64573e39d09",
            "cmake_configure_default": "CUDAFLAGS=\"-I.cuda_compat/include -Wno-deprecated-gpu-targets\" cmake -S .deps/llama.cpp -B .deps/llama.cpp/build-default -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"61;75\" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF",
            "cmake_configure_cublas": "CUDAFLAGS=\"-I.cuda_compat/include -Wno-deprecated-gpu-targets\" cmake -S .deps/llama.cpp -B .deps/llama.cpp/build-cublas -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"61;75\" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FORCE_CUBLAS=ON"
        },
        "pinned_model": {
            "repository": "Qwen/Qwen3-4B-GGUF",
            "filename": "Qwen3-4B-Q4_K_M.gguf",
            "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            "quantization": "Q4_K_M",
            "size_bytes": 2497280256
        },
        "workload_specification": {
            "n_prompt": 512,
            "n_gen": 128,
            "n_depths": [0, 2048],
            "n_batch": 2048,
            "n_ubatch": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "gpu_layers": 99,
            "flash_attn": "auto",
            "repetitions": 10
        },
        "benchmarks": analysis_data
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated manifest: {manifest_file}")

if __name__ == "__main__":
    main()
