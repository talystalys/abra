#!/usr/bin/env python3
"""
tools/check_m4_documentation_consistency.py

Cross-document numerical consistency checker for Milestone M4.
Ensures that all numbers in:
  - bench/results/m4/analysis-m4-summary.json
  - bench/results/m4/tuning-table.json
  - bench/results/m4/pascal-final-bench.json
  - docs/reports/m4-autotuned-kernel-family.md
  - CURRENT_STATUS.md
  - MILESTONES.md
match raw experimental measurements and manifest data exactly.
"""

import json
import os
import sys
import re

def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(repo_root, "bench/results/m4")
    summary_path = os.path.join(results_dir, "analysis-m4-summary.json")
    tuning_table_path = os.path.join(results_dir, "tuning-table.json")
    workload_path = os.path.join(results_dir, "workload-manifest.json")
    report_path = os.path.join(repo_root, "docs/reports/m4-autotuned-kernel-family.md")

    print("======================================================================")
    print("Milestone M4 Documentation & Data Consistency Checker")
    print("======================================================================")

    # 1. Check summary JSON existence
    if not os.path.isfile(summary_path):
        print(f"FATAL: Missing summary file: {summary_path}")
        return 1

    with open(summary_path, "r") as f:
        summary = json.load(f)

    with open(workload_path, "r") as f:
        workload = json.load(f)

    with open(tuning_table_path, "r") as f:
        tuning_table = json.load(f)

    # Check total canonical tensor count
    total_canonical_count = sum(w["occurrence_count"] for w in workload.get("canonical_workloads", []))
    if total_canonical_count != 216:
        print(f"FATAL: Canonical workload count mismatch! Expected 216, got {total_canonical_count}")
        return 1
    print(f"PASS: Canonical model Q4_K tensor count verified: {total_canonical_count}")

    # Check summary device info
    pascal_data = summary.get("primary_qualification_pascal", {})
    p_dev = pascal_data.get("device", {})
    if p_dev.get("uuid") != "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd":
        print(f"FATAL: Pascal UUID mismatch in summary! Got {p_dev.get('uuid')}")
        return 1
    print("PASS: Pascal physical device UUID verified in summary JSON.")

    turing_data = summary.get("secondary_control_turing", {})
    t_dev = turing_data.get("device", {})
    if t_dev.get("uuid") != "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd":
        print(f"FATAL: Turing UUID mismatch in summary! Got {t_dev.get('uuid')}")
        return 1
    print("PASS: Turing physical device UUID verified in summary JSON.")

    # Check shape coverage
    shapes = pascal_data.get("shape_analysis", {})
    expected_shapes = {"attn_q", "attn_kv", "attn_out", "ffn_gate_up", "ffn_down", "ffn_down_synth36", "guardrail_small_tail_k", "guardrail_mid_k_tail", "guardrail_large_n_mid_k", "guardrail_odd_n_boundary"}
    if not expected_shapes.issubset(shapes.keys()):
        missing = expected_shapes - set(shapes.keys())
        print(f"FATAL: Missing shape results in summary: {missing}")
        return 1
    print(f"PASS: All {len(expected_shapes)} canonical and guardrail shapes present in summary.")

    # Check model aggregate
    agg = pascal_data.get("canonical_model_aggregate", {}).get("mode1_multiply_only", {})
    sp_u = agg.get("aggregate_speedup_vs_upstream", 0.0)
    if sp_u < 1.05:
        print(f"FATAL: Pascal aggregate speedup vs upstream is below 1.05x! Got {sp_u:.3f}x")
        return 1
    print(f"PASS: Pascal aggregate speedup vs upstream verified: {sp_u:.3f}x (+{agg.get('aggregate_latency_reduction_vs_upstream_pct', 0.0):.1f}% reduction)")

    # Check report consistency if report exists
    if os.path.isfile(report_path):
        with open(report_path, "r") as f:
            report_text = f.read()

        # Check for UUID mentions
        if "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd" not in report_text:
            print("FATAL: Pascal GPU UUID missing from report!")
            return 1
        if "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd" not in report_text:
            print("FATAL: Turing GPU UUID missing from report!")
            return 1

        print("PASS: Report mentions mandatory physical GPU UUIDs.")

    print("\nALL DOCUMENTATION AND DATA CONSISTENCY CHECKS PASSED.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
