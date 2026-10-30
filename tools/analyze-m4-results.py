#!/usr/bin/env python3
"""
tools/analyze-m4-results.py

Comprehensive statistical analyzer for Milestone M4 Autotuning and Qualification.
Processes:
  1. bench/results/m4/pascal-final-bench.json (100 samples)
  2. bench/results/m4/turing-control-bench.json (100 samples)
  3. bench/results/m4/workload-manifest.json
  4. bench/results/m4/candidate-manifest.json
  5. bench/results/m4/tuning-table.json

Computes:
  - Shape-level comparison (Upstream MMVQ vs best static M3 vs best M4 vs autotuned selector)
  - Speedup metrics
  - Canonical Qwen3-4B Q4_K model-weighted aggregate latency and speedup across 216 tensors
  - Produces bench/results/m4/analysis-m4-summary.json
"""

import json
import math
import os
import sys

def compute_stats(lats):
    s_lats = sorted(lats)
    n = len(s_lats)
    med = s_lats[n // 2] if n % 2 == 1 else (s_lats[n // 2 - 1] + s_lats[n // 2]) / 2.0
    mean = sum(s_lats) / n
    mi = s_lats[0]
    ma = s_lats[-1]
    var = sum((x - mean) ** 2 for x in s_lats) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    spread = ma - mi
    return {
        "median_us": med,
        "mean_us": mean,
        "min_us": mi,
        "max_us": ma,
        "stddev_us": sd,
        "spread_us": spread
    }

def analyze_dataset(bench_json_path, workload_manifest, candidate_manifest):
    if not os.path.isfile(bench_json_path):
        raise FileNotFoundError(f"Missing benchmark file: {bench_json_path}")

    with open(bench_json_path, "r") as f:
        data = json.load(f)

    device_info = data.get("device", {})
    results = data.get("results", [])

    by_shape = {}
    for r in results:
        s = r["shape_name"]
        if s not in by_shape:
            by_shape[s] = {}
        by_shape[s][r["candidate_id"]] = r

    shape_analysis = {}

    weighted_upstream_us = 0.0
    weighted_best_m3_us = 0.0
    weighted_autotuned_us = 0.0
    total_weights = 0

    weighted_comb_upstream_us = 0.0
    weighted_comb_best_m3_us = 0.0
    weighted_comb_autotuned_us = 0.0

    workload_map = {}
    for w in workload_manifest.get("canonical_workloads", []):
        workload_map[w["shape_name"]] = w
    for w in workload_manifest.get("synthetic_guardrail_workloads", []):
        workload_map[w["shape_name"]] = w

    for s, c_map in by_shape.items():
        w_info = workload_map.get(s, {})
        n = next(iter(c_map.values()))["N"]
        k = next(iter(c_map.values()))["K"]
        weight = w_info.get("occurrence_count", 0)
        classification = w_info.get("classification", "unknown")

        u_run = c_map.get(0)
        u_stats = compute_stats(u_run["latencies_us"]) if u_run else {}
        u_comb_stats = compute_stats(u_run.get("combined_latencies_us", [])) if u_run and u_run.get("combined_latencies_us") else {}

        m3_candidates = [204, 104, 404]
        best_m3_id = None
        best_m3_stats = None
        best_m3_comb_stats = None
        min_m3_med = 1e9

        for cid in m3_candidates:
            if cid in c_map:
                c_st = compute_stats(c_map[cid]["latencies_us"])
                if c_st["median_us"] < min_m3_med:
                    min_m3_med = c_st["median_us"]
                    best_m3_id = cid
                    best_m3_stats = c_st
                    best_m3_comb_stats = compute_stats(c_map[cid].get("combined_latencies_us", []))

        best_m4_id = None
        best_m4_stats = None
        best_m4_comb_stats = None
        min_m4_med = 1e9

        for cid, r in c_map.items():
            if cid not in (0, 999):
                c_st = compute_stats(r["latencies_us"])
                if c_st["median_us"] < min_m4_med:
                    min_m4_med = c_st["median_us"]
                    best_m4_id = cid
                    best_m4_stats = c_st
                    best_m4_comb_stats = compute_stats(r.get("combined_latencies_us", []))

        auto_run = c_map.get(999)
        auto_stats = compute_stats(auto_run["latencies_us"]) if auto_run else (best_m4_stats or best_m3_stats)
        auto_comb_stats = compute_stats(auto_run.get("combined_latencies_us", [])) if auto_run and auto_run.get("combined_latencies_us") else (best_m4_comb_stats or best_m3_comb_stats)

        sp_auto_vs_upstream = u_stats["median_us"] / auto_stats["median_us"] if u_stats and auto_stats else 1.0
        sp_auto_vs_best_m3 = best_m3_stats["median_us"] / auto_stats["median_us"] if best_m3_stats and auto_stats else 1.0

        sp_comb_auto_vs_upstream = u_comb_stats["median_us"] / auto_comb_stats["median_us"] if u_comb_stats and auto_comb_stats else 1.0
        sp_comb_auto_vs_best_m3 = best_m3_comb_stats["median_us"] / auto_comb_stats["median_us"] if best_m3_comb_stats and auto_comb_stats else 1.0

        total_macs = float(n) * float(k)
        gmac_s = (total_macs / (auto_stats["median_us"] * 1e-6)) / 1e9
        q4_bytes = float(n) * (float(k) / 256.0) * 144.0
        q8_bytes = (float(k) / 32.0) * 36.0
        out_bytes = float(n) * 4.0
        eff_bytes = q4_bytes + q8_bytes + out_bytes
        eff_gbps = (eff_bytes / (auto_stats["median_us"] * 1e-6)) / (1024.0 ** 3)

        shape_analysis[s] = {
            "shape_name": s,
            "N": n,
            "K": k,
            "classification": classification,
            "occurrence_count": weight,
            "upstream_mmvq": {
                "median_us": u_stats.get("median_us", 0.0),
                "mean_us": u_stats.get("mean_us", 0.0),
                "min_us": u_stats.get("min_us", 0.0),
                "max_us": u_stats.get("max_us", 0.0),
                "stddev_us": u_stats.get("stddev_us", 0.0),
                "spread_us": u_stats.get("spread_us", 0.0),
                "combined_median_us": u_comb_stats.get("median_us", 0.0)
            },
            "best_m3_candidate": {
                "candidate_id": best_m3_id,
                "variant_name": c_map[best_m3_id]["variant_name"] if best_m3_id in c_map else "None",
                "median_us": best_m3_stats.get("median_us", 0.0),
                "combined_median_us": best_m3_comb_stats.get("median_us", 0.0)
            },
            "best_m4_candidate": {
                "candidate_id": best_m4_id,
                "variant_name": c_map[best_m4_id]["variant_name"] if best_m4_id in c_map else "None",
                "median_us": best_m4_stats.get("median_us", 0.0),
                "combined_median_us": best_m4_comb_stats.get("median_us", 0.0)
            },
            "autotuned_selector": {
                "median_us": auto_stats.get("median_us", 0.0),
                "mean_us": auto_stats.get("mean_us", 0.0),
                "min_us": auto_stats.get("min_us", 0.0),
                "max_us": auto_stats.get("max_us", 0.0),
                "stddev_us": auto_stats.get("stddev_us", 0.0),
                "spread_us": auto_stats.get("spread_us", 0.0),
                "gmac_per_sec": gmac_s,
                "effective_throughput_gbps": eff_gbps,
                "speedup_vs_upstream": sp_auto_vs_upstream,
                "speedup_vs_best_m3": sp_auto_vs_best_m3,
                "combined_median_us": auto_comb_stats.get("median_us", 0.0),
                "combined_speedup_vs_upstream": sp_comb_auto_vs_upstream,
                "combined_speedup_vs_best_m3": sp_comb_auto_vs_best_m3
            },
            "all_evaluated_candidates": {
                cid: {
                    "variant_name": r["variant_name"],
                    "median_us": compute_stats(r["latencies_us"])["median_us"],
                    "speedup_vs_upstream": u_stats["median_us"] / compute_stats(r["latencies_us"])["median_us"] if u_stats else 1.0
                } for cid, r in c_map.items()
            }
        }

        if classification == "canonical_q4k_workload" and weight > 0:
            weighted_upstream_us += u_stats["median_us"] * weight
            weighted_best_m3_us += best_m3_stats["median_us"] * weight
            weighted_autotuned_us += auto_stats["median_us"] * weight
            total_weights += weight

            weighted_comb_upstream_us += u_comb_stats["median_us"] * weight
            weighted_comb_best_m3_us += best_m3_comb_stats["median_us"] * weight
            weighted_comb_autotuned_us += auto_comb_stats["median_us"] * weight

    agg_speedup_vs_upstream = weighted_upstream_us / weighted_autotuned_us if weighted_autotuned_us > 0 else 1.0
    agg_speedup_vs_best_m3 = weighted_best_m3_us / weighted_autotuned_us if weighted_autotuned_us > 0 else 1.0

    agg_comb_speedup_vs_upstream = weighted_comb_upstream_us / weighted_comb_autotuned_us if weighted_comb_autotuned_us > 0 else 1.0
    agg_comb_speedup_vs_best_m3 = weighted_comb_best_m3_us / weighted_comb_autotuned_us if weighted_comb_autotuned_us > 0 else 1.0

    aggregate_summary = {
        "total_canonical_q4k_tensors": total_weights,
        "mode1_multiply_only": {
            "total_weighted_upstream_latency_us": weighted_upstream_us,
            "total_weighted_best_m3_latency_us": weighted_best_m3_us,
            "total_weighted_autotuned_latency_us": weighted_autotuned_us,
            "aggregate_speedup_vs_upstream": agg_speedup_vs_upstream,
            "aggregate_speedup_vs_best_m3": agg_speedup_vs_best_m3,
            "aggregate_latency_reduction_vs_upstream_pct": (1.0 - (weighted_autotuned_us / weighted_upstream_us)) * 100.0,
            "aggregate_latency_reduction_vs_best_m3_pct": (1.0 - (weighted_autotuned_us / weighted_best_m3_us)) * 100.0
        },
        "mode2_combined_pipeline": {
            "total_weighted_comb_upstream_latency_us": weighted_comb_upstream_us,
            "total_weighted_comb_best_m3_latency_us": weighted_comb_best_m3_us,
            "total_weighted_comb_autotuned_latency_us": weighted_comb_autotuned_us,
            "aggregate_speedup_mode2_vs_upstream": agg_comb_speedup_vs_upstream,
            "aggregate_speedup_mode2_vs_best_m3": agg_comb_speedup_vs_best_m3
        }
    }

    return {
        "device": device_info,
        "shape_analysis": shape_analysis,
        "canonical_model_aggregate": aggregate_summary
    }

def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(repo_root, "bench/results/m4")

    pascal_path = os.path.join(results_dir, "pascal-final-bench.json")
    turing_path = os.path.join(results_dir, "turing-control-bench.json")
    workload_path = os.path.join(results_dir, "workload-manifest.json")
    candidate_path = os.path.join(results_dir, "candidate-manifest.json")
    tuning_table_path = os.path.join(results_dir, "tuning-table.json")

    with open(workload_path, "r") as f:
        workload_manifest = json.load(f)
    with open(candidate_path, "r") as f:
        candidate_manifest = json.load(f)

    print("Analyzing Pascal SM 6.1 (GTX 1060) Final Results...")
    pascal_summary = analyze_dataset(pascal_path, workload_manifest, candidate_manifest)

    print("Analyzing Turing SM 7.5 (GTX 1660 SUPER) Control Results...")
    turing_summary = analyze_dataset(turing_path, workload_manifest, candidate_manifest)

    summary = {
        "milestone": "M4",
        "title": "Milestone M4: Pascal Autotuned Kernel Family Characterization & Aggregate Analysis",
        "primary_qualification_pascal": pascal_summary,
        "secondary_control_turing": turing_summary
    }

    summary_path = os.path.join(results_dir, "analysis-m4-summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote comprehensive statistical summary to: {summary_path}")

    print("\n" + "=" * 110)
    print("PASCAL SM 6.1 (GTX 1060 6GB) SHAPE-LEVEL RESULTS & SPEEDUP SUMMARY")
    print("=" * 110)
    print(f"{'Shape Name':<20} {'Dims (NxK)':<14} {'Upstream':<10} {'Best M3':<10} {'Autotuned':<10} {'Sp vs U':<10} {'Sp vs M3':<10} {'Eff GiB/s':<10}")
    print("-" * 110)

    for s, sa in pascal_summary["shape_analysis"].items():
        dim_str = f"{sa['N']}x{sa['K']}"
        u_med = sa["upstream_mmvq"]["median_us"]
        m3_med = sa["best_m3_candidate"]["median_us"]
        auto_med = sa["autotuned_selector"]["median_us"]
        sp_u = sa["autotuned_selector"]["speedup_vs_upstream"]
        sp_m3 = sa["autotuned_selector"]["speedup_vs_best_m3"]
        eff = sa["autotuned_selector"]["effective_throughput_gbps"]
        print(f"{s:<20} {dim_str:<14} {u_med:<10.2f} {m3_med:<10.2f} {auto_med:<10.2f} {sp_u:<10.3f}x {sp_m3:<10.3f}x {eff:<10.2f}")

    print("-" * 110)
    agg = pascal_summary["canonical_model_aggregate"]["mode1_multiply_only"]
    print(f"CANONICAL QWEN3-4B Q4_K MODEL-WEIGHTED AGGREGATE (216 Tensors):")
    print(f"  - Total Weighted Upstream MMVQ Latency:  {agg['total_weighted_upstream_latency_us']:.2f} us")
    print(f"  - Total Weighted Best Static M3 Latency: {agg['total_weighted_best_m3_latency_us']:.2f} us")
    print(f"  - Total Weighted M4 Autotuned Latency:   {agg['total_weighted_autotuned_latency_us']:.2f} us")
    print(f"  - Aggregate Speedup vs Pinned Upstream:  {agg['aggregate_speedup_vs_upstream']:.3f}x (+{agg['aggregate_latency_reduction_vs_upstream_pct']:.1f}% latency reduction)")
    print(f"  - Aggregate Speedup vs Best Static M3:   {agg['aggregate_speedup_vs_best_m3']:.3f}x (+{agg['aggregate_latency_reduction_vs_best_m3_pct']:.1f}% latency reduction)")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
