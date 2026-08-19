#!/usr/bin/env python3
import json
import math
import sys
import os

from check_m2_documentation_consistency import get_canonical_metrics

def main():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pascal_path = os.path.join(repo_dir, "bench/results/m2/pascal-dp4a-bench.json")
    turing_path = os.path.join(repo_dir, "bench/results/m2/turing-dp4a-bench.json")
    
    p_data = get_canonical_metrics(pascal_path)
    t_data = get_canonical_metrics(turing_path)
    
    print("=================== PASCAL QWEN TABLE ===================")
    print("| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\\mu\\text{s}$) | Variant B: Basic DP4A Median ($\\mu\\text{s}$) | Variant C: Opt DP4A Median ($\\mu\\text{s}$) | Variant D: Shared DP4A Median ($\\mu\\text{s}$) | DP4A Speedup vs Scalar (Var C) | Var C vs Var B Ratio | Effective Requested Throughput (Var C) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in ["attn_q", "attn_kv", "attn_out", "ffn_gate_up", "ffn_down"]:
        m = p_data["metrics"][s]
        n = m[0]["N"]
        k = m[0]["K"]
        v0 = f"{m[0]['median_us']:.2f}"
        v1 = f"{m[1]['median_us']:.2f}"
        v2 = f"{m[2]['median_us']:.2f}"
        v3 = f"{m[3]['median_us']:.2f}"
        
        # bold best
        if m[1]['median_us'] < m[2]['median_us']:
            v1_str = f"**{v1}**"
            v2_str = v2
        else:
            v1_str = v1
            v2_str = f"**{v2}**"
            
        sp_scalar = f"**${m[2]['speedup_vs_scalar']:.2f}\\times$**"
        sp_basic = f"${m[2]['speedup_vs_basic']:.2f}\\times$"
        eff = f"${m[2]['effective_throughput_gbps']:.2f}\\text{{ GiB/s}}$"
        print(f"| `{s}` | {n} | {k} | {v0} | {v1_str} | {v2_str} | {v3} | {sp_scalar} | {sp_basic} | {eff} |")

    print("\n=================== PASCAL SYNTHETIC TABLE ===================")
    print("| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\\mu\\text{s}$) | Variant B: Basic DP4A Median ($\\mu\\text{s}$) | Variant C: Opt DP4A Median ($\\mu\\text{s}$) | Variant D: Shared DP4A Median ($\\mu\\text{s}$) | DP4A Speedup vs Scalar (Best) | Effective Requested Throughput (Best) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in ["synth_256", "synth_1024", "synth_4096"]:
        m = p_data["metrics"][s]
        n = m[0]["N"]
        k = m[0]["K"]
        v0 = f"{m[0]['median_us']:.2f}"
        v1 = f"{m[1]['median_us']:.2f}"
        v2 = f"{m[2]['median_us']:.2f}"
        v3 = f"{m[3]['median_us']:.2f}"
        
        best_vid = min(range(4), key=lambda i: m[i]["median_us"])
        v_strs = [v0, v1, v2, v3]
        v_strs[best_vid] = f"**{v_strs[best_vid]}**"
        
        best_sp = f"**${m[0]['median_us'] / m[best_vid]['median_us']:.2f}\\times$** (Var {chr(65+best_vid)})"
        best_eff = f"${m[best_vid]['effective_throughput_gbps']:.2f}\\text{{ GiB/s}}$"
        print(f"| `{s}` | {n} | {k} | {v_strs[0]} | {v_strs[1]} | {v_strs[2]} | {v_strs[3]} | {best_sp} | {best_eff} |")

    print("\n=================== TURING QWEN TABLE ===================")
    print("| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\\mu\\text{s}$) | Variant B: Basic DP4A Median ($\\mu\\text{s}$) | Variant C: Opt DP4A Median ($\\mu\\text{s}$) | Variant D: Shared DP4A Median ($\\mu\\text{s}$) | DP4A Speedup vs Scalar (Var C) | Var C vs Var B Ratio | Effective Requested Throughput (Var C) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in ["attn_q", "attn_kv", "attn_out", "ffn_gate_up", "ffn_down"]:
        m = t_data["metrics"][s]
        n = m[0]["N"]
        k = m[0]["K"]
        v0 = f"{m[0]['median_us']:.2f}"
        v1 = f"{m[1]['median_us']:.2f}"
        v2 = f"{m[2]['median_us']:.2f}"
        v3 = f"{m[3]['median_us']:.2f}"
        
        if m[1]['median_us'] < m[2]['median_us']:
            v1_str = f"**{v1}**"
            v2_str = v2
        elif m[2]['median_us'] < m[1]['median_us']:
            v1_str = v1
            v2_str = f"**{v2}**"
        else:
            v1_str = f"**{v1}**"
            v2_str = f"**{v2}**"
            
        sp_scalar = f"**${m[2]['speedup_vs_scalar']:.2f}\\times$**"
        sp_basic = f"${m[2]['speedup_vs_basic']:.2f}\\times$"
        eff = f"${m[2]['effective_throughput_gbps']:.2f}\\text{{ GiB/s}}$"
        print(f"| `{s}` | {n} | {k} | {v0} | {v1_str} | {v2_str} | {v3} | {sp_scalar} | {sp_basic} | {eff} |")

    print("\n=================== TURING SYNTHETIC TABLE ===================")
    print("| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\\mu\\text{s}$) | Variant B: Basic DP4A Median ($\\mu\\text{s}$) | Variant C: Opt DP4A Median ($\\mu\\text{s}$) | Variant D: Shared DP4A Median ($\\mu\\text{s}$) | DP4A Speedup vs Scalar (Best) | Effective Requested Throughput (Best) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in ["synth_256", "synth_1024", "synth_4096"]:
        m = t_data["metrics"][s]
        n = m[0]["N"]
        k = m[0]["K"]
        v0 = f"{m[0]['median_us']:.2f}"
        v1 = f"{m[1]['median_us']:.2f}"
        v2 = f"{m[2]['median_us']:.2f}"
        v3 = f"{m[3]['median_us']:.2f}"
        
        best_vid = min(range(4), key=lambda i: m[i]["median_us"])
        v_strs = [v0, v1, v2, v3]
        v_strs[best_vid] = f"**{v_strs[best_vid]}**"
        
        best_sp = f"**${m[0]['median_us'] / m[best_vid]['median_us']:.2f}\\times$** (Var {chr(65+best_vid)})"
        best_eff = f"${m[best_vid]['effective_throughput_gbps']:.2f}\\text{{ GiB/s}}$"
        print(f"| `{s}` | {n} | {k} | {v_strs[0]} | {v_strs[1]} | {v_strs[2]} | {v_strs[3]} | {best_sp} | {best_eff} |")

if __name__ == "__main__":
    main()
