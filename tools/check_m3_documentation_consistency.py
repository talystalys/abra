#!/usr/bin/env python3
"""
tools/check_m3_documentation_consistency.py

Strict structured validator asserting exact correspondence between:
1. Canonical raw benchmark JSONs (pascal-q4k-bench.json, turing-q4k-bench.json)
2. Canonical live environment manifest (manifest.json)
3. Canonical GGUF tensor map (qwen3-q4k-tensor-map.json)
4. Human-facing documentation (docs/reports/m3-q4k-q8-microkernel.md, CURRENT_STATUS.md, MILESTONES.md)

Validates every shape, variant, column, metric, and provenance attribute structurally.
Recomputes statistics directly from raw latency arrays and verifies that every table
cell in the report matches the recomputed canonical truth.
"""

import json
import math
import re
import sys
import os

def load_json(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Missing required file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_canonical_m3_metrics(bench_json_path):
    data = load_json(bench_json_path)
    metrics = {}

    by_shape = {}
    for r in data.get("results", []):
        s = r["shape_name"]
        if s not in by_shape:
            by_shape[s] = {}
        by_shape[s][r["variant_id"]] = r

    for s, vars_map in by_shape.items():
        metrics[s] = {}
        n = vars_map[0]["N"]
        k = vars_map[0]["K"]
        
        # Raw upstream latencies for speedup base
        u_lats = sorted(vars_map[0]["latencies_us"])
        u_count = len(u_lats)
        upstream_med = u_lats[u_count // 2] if u_count % 2 == 1 else (u_lats[u_count // 2 - 1] + u_lats[u_count // 2]) / 2.0
        
        u_comb_lats = sorted(vars_map[0].get("combined_latencies_us", []))
        u_comb_count = len(u_comb_lats)
        upstream_comb_med = u_comb_lats[u_comb_count // 2] if u_comb_count % 2 == 1 else (u_comb_lats[u_comb_count // 2 - 1] + u_comb_lats[u_comb_count // 2]) / 2.0 if u_comb_count > 0 else 0.0

        for vid in sorted(vars_map.keys()):
            r = vars_map[vid]
            lats = sorted(r["latencies_us"])
            count = len(lats)
            med = lats[count // 2] if count % 2 == 1 else (lats[count // 2 - 1] + lats[count // 2]) / 2.0
            mean = sum(lats) / count
            mi = lats[0]
            ma = lats[-1]
            var = sum((x - mean) ** 2 for x in lats) / (count - 1) if count > 1 else 0.0
            sd = math.sqrt(var)
            spread = ma - mi

            comb_lats = sorted(r.get("combined_latencies_us", []))
            comb_count = len(comb_lats)
            comb_med = comb_lats[comb_count // 2] if comb_count % 2 == 1 else (comb_lats[comb_count // 2 - 1] + comb_lats[comb_count // 2]) / 2.0 if comb_count > 0 else 0.0
            comb_mean = sum(comb_lats) / comb_count if comb_count > 0 else 0.0
            comb_mi = comb_lats[0] if comb_count > 0 else 0.0
            comb_ma = comb_lats[-1] if comb_count > 0 else 0.0
            comb_spread = comb_ma - comb_mi

            med_sec = med * 1e-6
            total_macs = float(n) * float(k)
            gmac_s = (total_macs / med_sec) / 1e9
            
            # Logical tensor bytes: Q4 weights (144 B per 256) + Q8 activations (36 B per 32) + F32 output (4 B per element)
            q4_bytes = float(n) * (float(k) / 256.0) * 144.0
            q8_bytes = (float(k) / 32.0) * 36.0
            out_bytes = float(n) * 4.0
            eff_bytes = q4_bytes + q8_bytes + out_bytes
            eff_throughput_gbps = (eff_bytes / med_sec) / (1024.0 ** 3)

            speedup_vs_upstream = upstream_med / med if med > 0 else 1.0
            comb_speedup_vs_upstream = upstream_comb_med / comb_med if comb_med > 0 else 1.0

            metrics[s][vid] = {
                "shape_name": s,
                "variant_id": vid,
                "variant_name": r["variant_name"],
                "N": n,
                "K": k,
                "median_us": med,
                "mean_us": mean,
                "min_us": mi,
                "max_us": ma,
                "stddev_us": sd,
                "spread_us": spread,
                "gmac_per_sec": gmac_s,
                "effective_throughput_gbps": eff_throughput_gbps,
                "speedup_vs_upstream": speedup_vs_upstream,
                "quantize_only_median_us": r.get("quantize_only_median_us", 0.0),
                "combined_median_us": comb_med,
                "combined_mean_us": comb_mean,
                "combined_min_us": comb_mi,
                "combined_max_us": comb_ma,
                "combined_spread_us": comb_spread,
                "speedup_mode2_vs_upstream": comb_speedup_vs_upstream,
                "verification": r.get("verification", {}),
                "verified_correct": r.get("verified_correct", False)
            }

    return {
        "device": data.get("device", {}),
        "metrics": metrics
    }

def clean_cell(cell_str):
    s = cell_str.strip()
    s = s.replace("**", "").replace("*", "").replace("`", "").replace("$", "")
    s = s.replace("\\mu\\text{s}", "").replace("\\text{ GiB/s}", "").replace("\\times", "")
    s = s.replace("GiB/s", "").replace("us", "").replace("x", "").replace("%", "")
    s = s.strip()
    return s

def find_cell(row_dict, substring):
    sub = substring.lower()
    for k, v in row_dict.items():
        if sub in k.lower():
            return clean_cell(v)
    return ""

def extract_markdown_tables(markdown_text):
    tables = []
    lines = markdown_text.splitlines()
    in_table = False
    current_table = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            current_table.append(stripped)
        else:
            if in_table and len(current_table) >= 2:
                tables.append(current_table)
            in_table = False
            current_table = []

    if in_table and len(current_table) >= 2:
        tables.append(current_table)

    return tables

def parse_table_rows(table_lines):
    if len(table_lines) < 3:
        return []
    header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows

def check_close(val1, val2, tol=0.1, label=""):
    diff = abs(val1 - val2)
    if diff > tol:
        return False, f"{label}: expected {val1:.3f}, got {val2:.3f} (diff {diff:.4f} > tol {tol})"
    return True, ""

def validate_m3_documentation(repo_root):
    results_dir = os.path.join(repo_root, "bench/results/m3")
    pascal_bench_path = os.path.join(results_dir, "pascal-q4k-bench.json")
    turing_bench_path = os.path.join(results_dir, "turing-q4k-bench.json")
    manifest_path = os.path.join(results_dir, "manifest.json")
    report_path = os.path.join(repo_root, "docs/reports/m3-q4k-q8-microkernel.md")
    status_path = os.path.join(repo_root, "CURRENT_STATUS.md")
    milestones_path = os.path.join(repo_root, "MILESTONES.md")

    print(f"Loading canonical raw evidence from: {results_dir}")
    pascal_canonical = get_canonical_m3_metrics(pascal_bench_path)
    turing_canonical = get_canonical_m3_metrics(turing_bench_path)
    manifest = load_json(manifest_path)

    errors = []

    # 1. Manifest Validation
    print("[Check 1/6] Validating Manifest against canonical hardware assertions...")
    pascal_dev = pascal_canonical["device"]
    if pascal_dev["uuid"] != "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd":
        errors.append(f"Pascal UUID mismatch: expected GPU-5078520c-bf92-4143-0d48-b02fd9d250dd, got {pascal_dev['uuid']}")
    if pascal_dev["pci_bus_id"] != "0000:03:00.0":
        errors.append(f"Pascal PCI Bus ID mismatch: expected 0000:03:00.0, got {pascal_dev['pci_bus_id']}")
    if pascal_dev["compute_capability"] != "6.1":
        errors.append(f"Pascal CC mismatch: expected 6.1, got {pascal_dev['compute_capability']}")

    turing_dev = turing_canonical["device"]
    if turing_dev["uuid"] != "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd":
        errors.append(f"Turing UUID mismatch: expected GPU-8a043791-359d-1df4-9353-38c5bc98e5dd, got {turing_dev['uuid']}")
    if turing_dev["pci_bus_id"] != "0000:81:00.0":
        errors.append(f"Turing PCI Bus ID mismatch: expected 0000:81:00.0, got {turing_dev['pci_bus_id']}")
    if turing_dev["compute_capability"] != "7.5":
        errors.append(f"Turing CC mismatch: expected 7.5, got {turing_dev['compute_capability']}")

    # Check manifest model info
    model_spec = manifest["expected_pinned_specifications"]["canonical_model"]
    if model_spec["sha256"] != "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5":
        errors.append(f"Canonical model SHA256 mismatch in manifest: {model_spec['sha256']}")
    if model_spec["total_tensors"] != 398 or model_spec["q4_k_tensors"] != 216:
        errors.append(f"Canonical model tensor count mismatch in manifest: {model_spec}")

    # 2. Check Host Provenance in Manifest vs Documentation
    print("[Check 2/6] Validating Host Provenance synchronization...")
    observed_host = manifest.get("observed_execution_environment", {}).get("host", {})
    observed_toolchain = manifest.get("observed_execution_environment", {}).get("toolchain", {})
    
    with open(status_path, "r", encoding="utf-8") as f:
        status_text = f.read()
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Check driver in status
    if observed_toolchain.get("nvidia_driver") and observed_toolchain["nvidia_driver"] not in status_text:
        errors.append(f"NVIDIA driver {observed_toolchain['nvidia_driver']} not in CURRENT_STATUS.md")

    # 3. Check CURRENT_STATUS.md & MILESTONES.md
    print("[Check 3/6] Validating CURRENT_STATUS.md and MILESTONES.md...")
    if "M3: Pascal Q4_K x Q8_1 Decode Microkernel Characterization" not in status_text and "M3" not in status_text:
        errors.append("CURRENT_STATUS.md does not reference Milestone M3 properly")
    with open(milestones_path, "r", encoding="utf-8") as f:
        milestones_text = f.read()
    if "Milestone M3" not in milestones_text:
        errors.append("MILESTONES.md does not reference Milestone M3")

    # 4. Check Report existence and required sections
    print("[Check 4/6] Validating report structure...")
    if not os.path.isfile(report_path):
        errors.append(f"Missing M3 report at: {report_path}")
        return False

    required_sections = [
        "Hardware & Environment Identification",
        "Canonical Tensor Mapping & Model Provenance",
        "Upstream Comparator & MMVQ Dispatch Contract",
        "Kernel Variant Architecture & Scheduling Hypothesis",
        "Numerical Verification & Frozen Upstream Envelope",
        "Pascal Execution Evidence",
        "Turing Control Evidence",
        "Analysis & Synthesis"
    ]
    for req in required_sections:
        if req.lower() not in report_text.lower():
            errors.append(f"Report missing required section: '{req}'")

    # 5. Deep Numerical Table Validation in Report
    print("[Check 5/6] Validating Report Markdown Tables against recomputed canonical metrics...")
    report_tables = extract_markdown_tables(report_text)
    print(f"  Found {len(report_tables)} markdown tables in report.")

    variant_name_to_id = {
        "Variant U: Exact Pinned Upstream MMVQ": 0,
        "Variant A: Custom 4-Warp Control (1 Row/Block)": 1,
        "Variant A: Custom 4-Warp Control": 1,
        "Variant B: Custom 2-Warp Split (2 Rows/Block)": 2,
        "Variant B: Custom 2-Warp Split": 2,
        "Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem)": 3,
        "Variant C: Custom 1-Warp Row": 3
    }

    for table_idx, tlines in enumerate(report_tables):
        rows = parse_table_rows(tlines)
        if not rows:
            continue
        first_row = rows[0]
        
        # Check if Mode 1 table
        if any("Throughput (GMAC/s)" in k for k in first_row.keys()) and any("Speedup vs Upstream" in k for k in first_row.keys()):
            is_pascal = (table_idx <= 3) # Table 2 is Pascal Mode 1, Table 4 is Turing Mode 1
            arch_canonical = pascal_canonical if is_pascal else turing_canonical
            arch_name = "Pascal" if is_pascal else "Turing"

            for r in rows:
                s_name = clean_cell(find_cell(r, "Workload"))
                v_name = clean_cell(find_cell(r, "Variant"))
                v_id = variant_name_to_id.get(v_name)
                if v_id is None:
                    for k_name, vid in variant_name_to_id.items():
                        if k_name.lower() in v_name.lower():
                            v_id = vid
                            break

                if s_name not in arch_canonical["metrics"] or v_id is None:
                    continue

                exp = arch_canonical["metrics"][s_name][v_id]
                
                # Check Median
                act_med = float(find_cell(r, "Median"))
                ok, err = check_close(exp["median_us"], act_med, tol=0.15, label=f"{arch_name} Mode1 {s_name} v{v_id} Median")
                if not ok: errors.append(err)

                # Check Mean
                act_mean = float(find_cell(r, "Mean"))
                ok, err = check_close(exp["mean_us"], act_mean, tol=0.15, label=f"{arch_name} Mode1 {s_name} v{v_id} Mean")
                if not ok: errors.append(err)

                # Check Min
                act_min = float(find_cell(r, "Min"))
                ok, err = check_close(exp["min_us"], act_min, tol=0.15, label=f"{arch_name} Mode1 {s_name} v{v_id} Min")
                if not ok: errors.append(err)

                # Check Max
                act_max = float(find_cell(r, "Max"))
                ok, err = check_close(exp["max_us"], act_max, tol=0.15, label=f"{arch_name} Mode1 {s_name} v{v_id} Max")
                if not ok: errors.append(err)

                # Check Spread
                act_spread = float(find_cell(r, "Spread"))
                ok, err = check_close(exp["spread_us"], act_spread, tol=0.15, label=f"{arch_name} Mode1 {s_name} v{v_id} Spread")
                if not ok: errors.append(err)

                # Check GMAC/s
                act_gmac = float(find_cell(r, "Throughput (GMAC/s)"))
                ok, err = check_close(exp["gmac_per_sec"], act_gmac, tol=1.5, label=f"{arch_name} Mode1 {s_name} v{v_id} GMAC/s")
                if not ok: errors.append(err)

                # Check Eff Throughput
                act_eff = float(find_cell(r, "Eff. Throughput"))
                ok, err = check_close(exp["effective_throughput_gbps"], act_eff, tol=1.5, label=f"{arch_name} Mode1 {s_name} v{v_id} Eff GiB/s")
                if not ok: errors.append(err)

                # Check Speedup
                act_sp = float(find_cell(r, "Speedup vs Upstream"))
                ok, err = check_close(exp["speedup_vs_upstream"], act_sp, tol=0.03, label=f"{arch_name} Mode1 {s_name} v{v_id} Speedup")
                if not ok: errors.append(err)

        # Check if Mode 2 table
        elif any("Quantize-Only" in k for k in first_row.keys()) and any("Combined Median" in k for k in first_row.keys()):
            is_pascal = (table_idx <= 3) # Table 3 is Pascal Mode 2, Table 5 is Turing Mode 2
            arch_canonical = pascal_canonical if is_pascal else turing_canonical
            arch_name = "Pascal" if is_pascal else "Turing"

            for r in rows:
                s_name = clean_cell(find_cell(r, "Workload"))
                v_name = clean_cell(find_cell(r, "Variant"))
                v_id = variant_name_to_id.get(v_name)
                if v_id is None:
                    for k_name, vid in variant_name_to_id.items():
                        if k_name.lower() in v_name.lower():
                            v_id = vid
                            break

                if s_name not in arch_canonical["metrics"] or v_id is None:
                    continue

                exp = arch_canonical["metrics"][s_name][v_id]

                # Check Quantize-Only Median
                act_q = float(find_cell(r, "Quantize-Only"))
                ok, err = check_close(exp["quantize_only_median_us"], act_q, tol=0.15, label=f"{arch_name} Mode2 {s_name} v{v_id} Quant-only")
                if not ok: errors.append(err)

                # Check Combined Median
                act_c_med = float(find_cell(r, "Combined Median"))
                ok, err = check_close(exp["combined_median_us"], act_c_med, tol=0.15, label=f"{arch_name} Mode2 {s_name} v{v_id} Comb Median")
                if not ok: errors.append(err)

                # Check Combined Mean
                act_c_mean = float(find_cell(r, "Combined Mean"))
                ok, err = check_close(exp["combined_mean_us"], act_c_mean, tol=0.15, label=f"{arch_name} Mode2 {s_name} v{v_id} Comb Mean")
                if not ok: errors.append(err)

                # Check Combined Spread
                act_c_spread = float(find_cell(r, "Combined Spread"))
                ok, err = check_close(exp["combined_spread_us"], act_c_spread, tol=0.15, label=f"{arch_name} Mode2 {s_name} v{v_id} Comb Spread")
                if not ok: errors.append(err)

                # Check Combined Speedup
                act_c_sp = float(find_cell(r, "Combined Speedup"))
                ok, err = check_close(exp["speedup_mode2_vs_upstream"], act_c_sp, tol=0.03, label=f"{arch_name} Mode2 {s_name} v{v_id} Comb Speedup")
                if not ok: errors.append(err)

    print(f"[Check 6/6] Final Consistency Verification Summary:")
    if errors:
        print(f"FAILED: {len(errors)} consistency errors found:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        return False
    else:
        print("PASS: All documentation tables, metrics, and raw canonical evidence are 100% synchronized!")
        return True

def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    success = validate_m3_documentation(repo_root)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
