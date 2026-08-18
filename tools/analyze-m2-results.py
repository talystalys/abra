#!/usr/bin/env python3
import json
import math
import sys
import os

def compute_stats(samples):
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    min_val = s[0]
    max_val = s[-1]
    mean_val = sum(s) / n
    
    if n % 2 == 1:
        median_val = s[n // 2]
    else:
        median_val = (s[n // 2 - 1] + s[n // 2]) / 2.0
        
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in s) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        variance = 0.0
        std_dev = 0.0
        
    return {
        "count": n,
        "min": min_val,
        "median": median_val,
        "mean": mean_val,
        "max": max_val,
        "std_dev": std_dev,
        "spread": max_val - min_val,
        "samples": s
    }

def process_benchmark_file(file_path):
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None

    raw_results = data.get("results", [])
    device_info = data.get("device", {})

    # Group by shape
    shapes = {}
    for item in raw_results:
        shape = item["shape_name"]
        if shape not in shapes:
            shapes[shape] = {
                "shape_name": shape,
                "N": item["N"],
                "K": item["K"],
                "variants": {}
            }
        
        var_id = item["variant_id"]
        latencies = item.get("latencies_us", [])
        stats = compute_stats(latencies)
        
        median_us = stats["median"]
        median_sec = median_us * 1e-6 if median_us > 0 else 1.0
        N = item["N"]
        K = item["K"]
        total_macs = N * K
        gmac_s = (total_macs / median_sec) / 1e9
        gops_s = (2.0 * total_macs / median_sec) / 1e9 # 1 MAC = 2 ops
        effective_bytes = (N * K * 1) + (K * 1) + (N * 4)
        throughput_gbps = (effective_bytes / median_sec) / (1024.0 * 1024.0 * 1024.0)

        shapes[shape]["variants"][var_id] = {
            "variant_id": var_id,
            "variant_name": item["variant_name"],
            "stats_us": stats,
            "gmac_per_sec": gmac_s,
            "effective_gops": gops_s,
            "effective_throughput_gbps": throughput_gbps,
            "verified_correct": item.get("verified_correct", False)
        }

    # Calculate speedups
    for shape, sdata in shapes.items():
        vars_map = sdata["variants"]
        scalar_med = vars_map.get(0, {}).get("stats_us", {}).get("median", None)
        basic_med = vars_map.get(1, {}).get("stats_us", {}).get("median", None)

        for var_id, vinfo in vars_map.items():
            cur_med = vinfo["stats_us"]["median"]
            speedup_vs_scalar = (scalar_med / cur_med) if scalar_med and cur_med > 0 else 1.0
            speedup_vs_basic = (basic_med / cur_med) if basic_med and cur_med > 0 else 1.0
            vinfo["speedup_vs_scalar"] = speedup_vs_scalar
            vinfo["speedup_vs_basic_dp4a"] = speedup_vs_basic

    return {
        "device": device_info,
        "shapes": shapes,
        "raw_results_count": len(raw_results)
    }

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench/results/m2"
    pascal_file = os.path.join(results_dir, "pascal-dp4a-bench.json")
    turing_file = os.path.join(results_dir, "turing-dp4a-bench.json")

    summary = {
        "pascal": process_benchmark_file(pascal_file),
        "turing": process_benchmark_file(turing_file)
    }

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
