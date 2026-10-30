#!/usr/bin/env python3
"""
tools/generate_m3_summary.py

Computes statistical summaries and speedup ratios from raw Pascal and Turing
benchmark JSON files and writes bench/results/m3/analysis-m3-summary.json.
"""

import json
import os
import sys

def analyze_dataset(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    device = data["device"]
    results = data["results"]

    shapes_summary = {}
    for r in results:
        s_name = r["shape_name"]
        if s_name not in shapes_summary:
            shapes_summary[s_name] = {}

        var_id = r["variant_id"]
        shapes_summary[s_name][var_id] = {
            "variant_id": var_id,
            "variant_name": r["variant_name"],
            "N": r["N"],
            "K": r["K"],
            "median_us": r["median_us"],
            "mean_us": r["mean_us"],
            "min_us": r["min_us"],
            "max_us": r["max_us"],
            "stddev_us": r["stddev_us"],
            "spread_us": r["spread_us"],
            "gmac_per_sec": r["gmac_per_sec"],
            "effective_throughput_gbps": r["effective_throughput_gbps"],
            "quantize_only_median_us": r.get("quantize_only_median_us", 0.0),
            "combined_median_us": r.get("combined_median_us", 0.0),
            "combined_mean_us": r.get("combined_mean_us", 0.0),
            "combined_min_us": r.get("combined_min_us", 0.0),
            "combined_max_us": r.get("combined_max_us", 0.0),
            "combined_stddev_us": r.get("combined_stddev_us", 0.0),
            "combined_spread_us": r.get("combined_spread_us", 0.0),
            "verification": r.get("verification", {}),
            "verified_correct": r.get("verified_correct", True)
        }

    analyzed_shapes = []
    for s_name, var_dict in shapes_summary.items():
        u_bench = var_dict.get(0)
        a_bench = var_dict.get(1)

        variants_analyzed = []
        for vid in sorted(var_dict.keys()):
            v = var_dict[vid]
            mul_speedup_vs_u = (u_bench["median_us"] / v["median_us"]) if u_bench and v["median_us"] > 0 else 1.0
            mul_speedup_vs_a = (a_bench["median_us"] / v["median_us"]) if a_bench and v["median_us"] > 0 else 1.0
            comb_speedup_vs_u = (u_bench["combined_median_us"] / v["combined_median_us"]) if u_bench and v["combined_median_us"] > 0 else 1.0

            v_entry = dict(v)
            v_entry["speedup_mode1_vs_upstream"] = round(mul_speedup_vs_u, 3)
            v_entry["speedup_mode1_vs_variant_a"] = round(mul_speedup_vs_a, 3)
            v_entry["speedup_mode2_vs_upstream"] = round(comb_speedup_vs_u, 3)
            variants_analyzed.append(v_entry)

        analyzed_shapes.append({
            "shape_name": s_name,
            "N": variants_analyzed[0]["N"],
            "K": variants_analyzed[0]["K"],
            "variants": variants_analyzed
        })

    return {
        "device": device,
        "arch": data.get("arch", "unknown"),
        "shapes": analyzed_shapes
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(repo_dir, "bench", "results", "m3")

    pascal_raw = os.path.join(results_dir, "pascal-q4k-bench.json")
    turing_raw = os.path.join(results_dir, "turing-q4k-bench.json")
    summary_out = os.path.join(results_dir, "analysis-m3-summary.json")

    summary = {
        "title": "Milestone M3/M3R1 Benchmark Summary Analysis",
        "pascal": analyze_dataset(pascal_raw),
        "turing": analyze_dataset(turing_raw)
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote analysis summary to: {summary_out}")

if __name__ == "__main__":
    main()
