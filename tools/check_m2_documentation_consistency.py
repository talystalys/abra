#!/usr/bin/env python3
import json
import math
import sys
import os

def load_json(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)

def get_canonical_metrics(bench_json_path):
    data = load_json(bench_json_path)
    metrics = {}
    
    # Group results by shape
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
        scalar_med = vars_map[0]["median_us"]
        basic_med = vars_map[1]["median_us"]
        
        for vid, r in vars_map.items():
            lats = sorted(r["latencies_us"])
            count = len(lats)
            med = lats[count // 2] if count % 2 == 1 else (lats[count // 2 - 1] + lats[count // 2]) / 2.0
            mean = sum(lats) / count
            mi = lats[0]
            ma = lats[-1]
            var = sum((x - mean) ** 2 for x in lats) / (count - 1) if count > 1 else 0.0
            sd = math.sqrt(var)
            spread = ma - mi
            
            med_sec = med * 1e-6
            total_macs = float(n) * float(k)
            gmac_s = (total_macs / med_sec) / 1e9
            eff_bytes = float(n * k) + float(k) + float(n * 4)
            eff_throughput_gbps = (eff_bytes / med_sec) / (1024.0 ** 3)
            
            speedup_vs_scalar = scalar_med / med
            speedup_vs_basic = basic_med / med
            
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
                "speedup_vs_scalar": speedup_vs_scalar,
                "speedup_vs_basic": speedup_vs_basic,
                "verified_correct": r.get("verified_correct", False)
            }
            
    return {
        "device": data.get("device", {}),
        "metrics": metrics
    }

def verify_report_synchronization(repo_dir):
    pascal_path = os.path.join(repo_dir, "bench/results/m2/pascal-dp4a-bench.json")
    turing_path = os.path.join(repo_dir, "bench/results/m2/turing-dp4a-bench.json")
    report_path = os.path.join(repo_dir, "docs/reports/m2-dp4a-microkernel.md")
    status_path = os.path.join(repo_dir, "CURRENT_STATUS.md")
    milestones_path = os.path.join(repo_dir, "MILESTONES.md")
    
    pascal_data = get_canonical_metrics(pascal_path)
    turing_data = get_canonical_metrics(turing_path)
    
    with open(report_path) as f:
        report_text = f.read()
    with open(status_path) as f:
        status_text = f.read()
    with open(milestones_path) as f:
        milestones_text = f.read()
        
    errors = []
    
    # Check all shapes in Pascal
    p_metrics = pascal_data["metrics"]
    for s, vmap in p_metrics.items():
        for vid, m in vmap.items():
            med_str = f"{m['median_us']:.2f}"
            if med_str not in report_text:
                errors.append(f"Pascal {s} Var {vid} median ({med_str}) not found in report")
                
    # Check all shapes in Turing
    t_metrics = turing_data["metrics"]
    for s, vmap in t_metrics.items():
        for vid, m in vmap.items():
            med_str = f"{m['median_us']:.2f}"
            if med_str not in report_text:
                errors.append(f"Turing {s} Var {vid} median ({med_str}) not found in report")
                
    # Key check values in CURRENT_STATUS.md
    for s in ["attn_q", "attn_kv", "attn_out", "ffn_gate_up", "ffn_down"]:
        p_c_eff = f"{p_metrics[s][2]['effective_throughput_gbps']:.2f}"
        if p_c_eff not in status_text:
            errors.append(f"Pascal {s} Var C throughput ({p_c_eff}) not found in CURRENT_STATUS.md")
            
        t_b_med = f"{t_metrics[s][1]['median_us']:.2f}"
        if t_b_med not in status_text:
            errors.append(f"Turing {s} Var B median ({t_b_med}) not found in CURRENT_STATUS.md")
            
    # Key check in MILESTONES.md
    if "176.13" not in milestones_text:
        errors.append("Pascal max throughput 176.13 not found in MILESTONES.md")
        
    if errors:
        print("FAIL: Documentation consistency errors detected:")
        for e in errors:
            print(f"  - {e}")
        return False
        
    print("PASS: All human-facing documentation numbers are mechanically synchronized with raw benchmark JSON.")
    return True

if __name__ == "__main__":
    repo_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    success = verify_report_synchronization(repo_dir)
    sys.exit(0 if success else 1)
