#!/usr/bin/env python3
"""
tools/analyze-m3-results.py

Analyzes Milestone M3 raw benchmark results for Pascal and Turing GPUs,
computing speedup ratios, latency distributions, and throughput metrics.
"""

import json
import sys
import os

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_bench(json_data):
    device = json_data["device"]
    results = json_data["results"]
    arch = json_data.get("arch", "unknown")

    print(f"========================================================================================")
    print(f"Device: {device['name']} (CC {device['compute_capability']}, PCI {device['pci_bus_id']})")
    print(f"UUID: {device['uuid']}")
    print(f"========================================================================================")

    shapes = {}
    for r in results:
        s_name = r["shape_name"]
        if s_name not in shapes:
            shapes[s_name] = []
        shapes[s_name].append(r)

    print(f"\n--- TIMING MODE 1: MULTIPLY ONLY (M=1 Decode Microkernel) ---")
    print(f"{'Shape':<14} | {'Variant':<34} | {'Median (us)':<12} | {'Mean (us)':<10} | {'Spread (us)':<12} | {'GMAC/s':<8} | {'Eff GiB/s':<10} | {'Speedup vs U':<12}")
    print("-" * 125)

    for s_name, res_list in shapes.items():

        baseline = None
        for r in res_list:
            if r["variant_id"] == 0:
                baseline = r
                break
        if baseline is None:
            baseline = res_list[0]

        for r in res_list:
            speedup = baseline["median_us"] / r["median_us"] if r["median_us"] > 0 else 1.0
            var_short = r["variant_name"]
            if len(var_short) > 34:
                var_short = var_short[:31] + "..."
            print(f"{s_name:<14} | {var_short:<34} | {r['median_us']:<12.2f} | {r['mean_us']:<10.2f} | {r['spread_us']:<12.2f} | {r['gmac_per_sec']:<8.2f} | {r['effective_throughput_gbps']:<10.2f} | {speedup:<12.3f}x")
        print("-" * 125)

    print(f"\n--- TIMING MODE 2: COMBINED ACTIVATION QUANTIZATION (F32->Q8_1) + MULTIPLY ---")
    print(f"{'Shape':<14} | {'Variant':<34} | {'Comb Med (us)':<14} | {'Quant Med (us)':<14} | {'Comb Speedup vs U':<18}")
    print("-" * 105)

    for s_name, res_list in shapes.items():
        baseline = None
        for r in res_list:
            if r["variant_id"] == 0:
                baseline = r
                break
        if baseline is None:
            baseline = res_list[0]

        for r in res_list:
            speedup = baseline["combined_median_us"] / r["combined_median_us"] if r["combined_median_us"] > 0 else 1.0
            var_short = r["variant_name"]
            if len(var_short) > 34:
                var_short = var_short[:31] + "..."
            print(f"{s_name:<14} | {var_short:<34} | {r['combined_median_us']:<14.2f} | {r['quantize_only_median_us']:<14.2f} | {speedup:<18.3f}x")
        print("-" * 105)

def main():
    pascal_path = sys.argv[1] if len(sys.argv) > 1 else "bench/results/m3/pascal-q4k-bench.json"
    turing_path = sys.argv[2] if len(sys.argv) > 2 else "bench/results/m3/turing-q4k-bench.json"

    if os.path.isfile(pascal_path):
        pascal_data = load_json(pascal_path)
        analyze_bench(pascal_data)

    if os.path.isfile(turing_path):
        turing_data = load_json(turing_path)
        analyze_bench(turing_data)

if __name__ == "__main__":
    main()
