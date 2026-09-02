#!/usr/bin/env python3
"""
tools/generate-m3-manifest.py

Generates the canonical environment, hardware, toolchain, and benchmark
provenance manifest for Milestone M3 in bench/results/m3/manifest.json.
"""

import json
import os
import sys
import subprocess
import datetime

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def get_os_info():
    if os.path.isfile("/etc/os-release"):
        info = {}
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    info[k] = v.strip('"')
        return info.get("PRETTY_NAME", info.get("NAME", "Linux"))
    return "Unknown Linux"

def get_cpu_info():
    model = "Unknown CPU"
    cores = 0
    threads = 0
    sockets = set()
    if os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    model = line.split(":", 1)[1].strip()
                elif "processor" in line:
                    threads += 1
                elif "physical id" in line:
                    sockets.add(line.split(":", 1)[1].strip())
                elif "cpu cores" in line:
                    cores = int(line.split(":", 1)[1].strip())
    num_sockets = max(len(sockets), 1)
    total_physical_cores = cores * num_sockets if cores > 0 else threads
    return f"{num_sockets} x {model} ({total_physical_cores} physical cores, {threads} logical threads)"

def get_ram_info():
    if os.path.isfile("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    return {
                        "total_kib": kb,
                        "total_mib": round(kb / 1024, 1),
                        "total_gib": round(kb / (1024 * 1024), 2)
                    }
    return {}

def get_numa_nodes_count():
    try:
        nodes = [d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]
        return len(nodes)
    except Exception:
        return 1

def get_gpu_numa_node(pci_bus_id):
    clean_pci = pci_bus_id.strip()
    if not clean_pci.startswith("0000:"):
        clean_pci = "0000:" + clean_pci.lstrip("0")
        if not clean_pci.startswith("0000:"):
            clean_pci = "0000:" + pci_bus_id
    numa_path = f"/sys/bus/pci/devices/{clean_pci}/numa_node"
    if os.path.isfile(numa_path):
        try:
            with open(numa_path) as f:
                node = int(f.read().strip())
                return node if node >= 0 else 0
        except Exception:
            pass
    return 0

def get_live_gpus():
    cmd = "nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,power.limit,compute_cap --format=csv,noheader,nounits"
    out = run_cmd(cmd)
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            pci_id = parts[3]
            if pci_id.startswith("00000000:"):
                pci_id = pci_id[4:]
            elif not pci_id.startswith("0000:"):
                pci_id = "0000:" + pci_id
                
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "pci_bus_id": pci_id,
                "vram_total_mib": float(parts[4]),
                "power_limit_w": float(parts[5]),
                "compute_capability": parts[6],
                "numa_node": get_gpu_numa_node(pci_id)
            })
    return gpus

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench/results/m3"
    pascal_bench_file = os.path.join(results_dir, "pascal-q4k-bench.json")
    turing_bench_file = os.path.join(results_dir, "turing-q4k-bench.json")
    manifest_file = os.path.join(results_dir, "manifest.json")
    
    pascal_bench = json.load(open(pascal_bench_file)) if os.path.isfile(pascal_bench_file) else {}
    turing_bench = json.load(open(turing_bench_file)) if os.path.isfile(turing_bench_file) else {}
    
    live_gpus = get_live_gpus()
    driver_version = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1")
    nvcc_raw = run_cmd("/usr/local/cuda-12.9/bin/nvcc --version 2>/dev/null || nvcc --version 2>/dev/null")
    gcc_raw = run_cmd("gcc --version | head -n 1")
    glibc_raw = run_cmd("ldd --version | head -n 1")
    kernel_raw = run_cmd("uname -s -r -v -m")
    hostname_raw = run_cmd("hostname")

    pascal_gpu = next((g for g in live_gpus if "1060" in g["name"] or g["uuid"] == "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"), None)
    turing_gpu = next((g for g in live_gpus if "1660" in g["name"] or g["uuid"] == "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"), None)

    manifest = {
        "manifest_version": "1.0",
        "milestone": "M3",
        "title": "Pascal Q4_K x Q8_1 Decode Microkernel Characterization",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        
        "mathematical_contract": {
            "operation": "y[n] = sum(k=0..K-1) dequantize(W[n,k]) * dequantize(x[k])",
            "operands": {
                "W": "block_q4_K (144 bytes per 256 weights = 4.5 bpw, asymmetric scales/mins)",
                "x": "block_q8_1 (36 bytes per 32 values, scale d + sum-scale s)",
                "y": "float32 output [N]"
            },
            "arithmetic": "Signed 8-bit integer dot products via IDP.4A.S8.S8 with FP32 dequantization scale accumulation"
        },

        "expected_pinned_specifications": {
            "m2_accepted_base_commit": "f10ead5fa94925e0a36980b1f3bdaa4a6998aa84",
            "pinned_llama_cpp_commit": "169e4a7ff201e666c2325bcd35afb64573e39d09",
            "canonical_model": {
                "name": "Qwen3-4B-Q4_K_M.gguf",
                "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
                "total_tensors": 398,
                "q4_k_tensors": 216,
                "q6_k_tensors": 37,
                "f32_tensors": 145
            },
            "required_qwen_shapes": [
                { "name": "attn_q", "N": 4096, "K": 2560, "model_quant": "Q4_K" },
                { "name": "attn_kv", "N": 1024, "K": 2560, "model_quant": "Q4_K (attn_k) / Q6_K (attn_v)" },
                { "name": "attn_out", "N": 2560, "K": 4096, "model_quant": "Q4_K" },
                { "name": "ffn_gate_up", "N": 9728, "K": 2560, "model_quant": "Q4_K" },
                { "name": "ffn_down", "N": 2560, "K": 9728, "model_quant": "Q6_K (evaluated as synthetic Q4_K shape)" }
            ],
            "synthetic_shapes": [
                { "name": "synth_256", "N": 256, "K": 256 },
                { "name": "synth_1024", "N": 1024, "K": 1024 },
                { "name": "synth_4096", "N": 4096, "K": 4096 }
            ],
            "kernel_variants": [
                { "id": 0, "name": "Variant U: Exact Pinned Upstream MMVQ (4-Warp Generic)" },
                { "id": 1, "name": "Variant A: Custom 4-Warp Control (1 Row/Block)" },
                { "id": 2, "name": "Variant B: Custom 2-Warp Split (2 Rows/Block)" },
                { "id": 3, "name": "Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem)" }
            ],
            "benchmark_config": {
                "warmup_iterations": 10,
                "measured_iterations": 100,
                "timing_method": "CUDA Events (cudaEventElapsedTime)",
                "timing_mode_1": "Multiply Only (M=1 Decode Microkernel)",
                "timing_mode_2": "Combined Activation Quantization (F32->Q8_1) + Multiply"
            },
            "pascal_assertions": {
                "expected_name": "NVIDIA GeForce GTX 1060 6GB",
                "expected_cc": "6.1",
                "expected_pci": "0000:03:00.0",
                "expected_uuid": "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd",
                "expected_numa_node": 0
            },
            "turing_assertions": {
                "expected_name": "NVIDIA GeForce GTX 1660 SUPER",
                "expected_cc": "7.5",
                "expected_pci": "0000:81:00.0",
                "expected_uuid": "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd",
                "expected_numa_node": 1
            }
        },

        "observed_execution_environment": {
            "host": {
                "hostname": hostname_raw,
                "os": get_os_info(),
                "kernel": kernel_raw,
                "cpu": get_cpu_info(),
                "ram": get_ram_info(),
                "numa_nodes_count": get_numa_nodes_count()
            },
            "toolchain": {
                "nvidia_driver": driver_version,
                "cuda_toolkit_path": "/usr/local/cuda-12.9",
                "nvcc_version": nvcc_raw,
                "host_compiler": gcc_raw,
                "glibc_version": glibc_raw
            },
            "live_gpus": live_gpus,
            "primary_qualification_pascal_observed": pascal_gpu,
            "secondary_control_turing_observed": turing_gpu
        },

        "benchmarks": {
            "pascal": pascal_bench,
            "turing": turing_bench
        }
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated M3 live manifest: {manifest_file}")

if __name__ == "__main__":
    main()
