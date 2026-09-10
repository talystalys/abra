#!/usr/bin/env python3
"""
tools/run-m4-autotune.py

Automated orchestrator for Milestone M4 Autotuning and Qualification Benchmarks.
1. Physical device identity verification (by immutable UUID).
2. Continuous background GPU telemetry (clock, power, temp, utilization).
3. Coarse parameter space autotuning search (17 candidates x all shapes).
4. Deterministic winner selection and tuning table synthesis.
5. Qualification-grade final benchmarking (100 samples) on Pascal GP106.
6. Control benchmarking (100 samples) on Turing TU116.
7. Model-weighted aggregation across the canonical 216 Q4_K tensors.
"""

import subprocess
import os
import sys
import time
import json
import signal

PASCAL_UUID = "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
PASCAL_PCI = "0000:03:00.0"
PASCAL_NAME = "NVIDIA GeForce GTX 1060 6GB"
PASCAL_CC = "6.1"

TURING_UUID = "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
TURING_PCI = "0000:81:00.0"
TURING_NAME = "NVIDIA GeForce GTX 1660 SUPER"
TURING_CC = "7.5"

def start_telemetry(uuid, output_csv_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cmd = [
        "nvidia-smi",
        f"--id={uuid}",
        "--query-gpu=timestamp,gpu_name,gpu_uuid,pci.bus_id,pstate,clocks.gr_mhz,clocks.mem_mhz,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv",
        "-lms", "100"
    ]
    f = open(output_csv_path, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
    return proc, f

def stop_telemetry(proc, f):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    if f:
        f.close()

def run_cmd(cmd, env=None):
    print(f"Executing: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, check=True)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f}s (exit code {res.returncode})\n")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(repo_root, "build")
    results_dir = os.path.join(repo_root, "bench/results/m4")
    bench_bin = os.path.join(build_dir, "q4k_q8_bench")
    os.makedirs(results_dir, exist_ok=True)

    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Step 1: Coarse Autotuning Search on Pascal GTX 1060
    if action in ("all", "autotune", "pascal-autotune"):
        print("======================================================================")
        print("PHASE 1: Coarse Autotuning Search on Pascal GTX 1060 (All Candidates)")
        print("======================================================================")
        raw_json = os.path.join(results_dir, "pascal-autotune-raw.json")
        telem_csv = os.path.join(results_dir, "pascal-telemetry-autotune.csv")
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = PASCAL_UUID

        t_proc, t_file = start_telemetry(PASCAL_UUID, telem_csv)
        try:
            cmd = [
                bench_bin,
                "--arch", "pascal",
                "--mode", "autotune",
                "--assert-name", PASCAL_NAME,
                "--assert-cc", PASCAL_CC,
                "--assert-pci", PASCAL_PCI,
                "--assert-uuid", PASCAL_UUID,
                "--warmup", "15",
                "--iters", "30",
                "--output", raw_json
            ]
            run_cmd(cmd, env=env)
        finally:
            stop_telemetry(t_proc, t_file)

    # Step 2: Winner Selection & Tuning Table Synthesis
    if action in ("all", "autotune", "select", "table"):
        print("======================================================================")
        print("PHASE 2: Winner Selection & Deterministic Tuning Table Synthesis")
        print("======================================================================")
        synthesize_tuning_table(results_dir)

    # Step 3: Qualification Final Benchmarking on Pascal GTX 1060 (100 samples)
    if action in ("all", "final", "pascal-final"):
        print("======================================================================")
        print("PHASE 3: Final Qualification Benchmarking on Pascal GTX 1060 (100 Samples)")
        print("======================================================================")
        final_json = os.path.join(results_dir, "pascal-final-bench.json")
        telem_csv = os.path.join(results_dir, "pascal-telemetry.csv")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = PASCAL_UUID

        t_proc, t_file = start_telemetry(PASCAL_UUID, telem_csv)
        try:
            cmd = [
                bench_bin,
                "--arch", "pascal",
                "--mode", "m4-final",
                "--assert-name", PASCAL_NAME,
                "--assert-cc", PASCAL_CC,
                "--assert-pci", PASCAL_PCI,
                "--assert-uuid", PASCAL_UUID,
                "--warmup", "20",
                "--iters", "100",
                "--output", final_json
            ]
            run_cmd(cmd, env=env)
        finally:
            stop_telemetry(t_proc, t_file)

    # Step 4: Control Benchmarking on Turing GTX 1660 SUPER (100 samples)
    if action in ("all", "final", "turing-final", "control"):
        print("======================================================================")
        print("PHASE 4: Control Benchmarking on Turing GTX 1660 SUPER (100 Samples)")
        print("======================================================================")
        turing_json = os.path.join(results_dir, "turing-control-bench.json")
        telem_csv = os.path.join(results_dir, "turing-telemetry.csv")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = TURING_UUID

        t_proc, t_file = start_telemetry(TURING_UUID, telem_csv)
        try:
            cmd = [
                bench_bin,
                "--arch", "turing",
                "--mode", "m4-final",
                "--assert-name", TURING_NAME,
                "--assert-cc", TURING_CC,
                "--assert-pci", TURING_PCI,
                "--assert-uuid", TURING_UUID,
                "--warmup", "20",
                "--iters", "100",
                "--output", turing_json
            ]
            run_cmd(cmd, env=env)
        finally:
            stop_telemetry(t_proc, t_file)

    # Step 5: Statistical Analysis & Aggregation Summary
    if action in ("all", "analyze", "summary"):
        print("======================================================================")
        print("PHASE 5: Statistical Analysis & Model-Weighted Aggregate Synthesis")
        print("======================================================================")
        analyze_script = os.path.join(repo_root, "tools/analyze-m4-results.py")
        subprocess.run([sys.executable, analyze_script, repo_root], check=True)

def synthesize_tuning_table(results_dir):
    raw_path = os.path.join(results_dir, "pascal-autotune-raw.json")
    if not os.path.isfile(raw_path):
        print(f"Warning: {raw_path} not found. Skipping table synthesis.")
        return

    with open(raw_path, "r") as f:
        data = json.load(f)

    # Group by shape
    by_shape = {}
    for r in data.get("results", []):
        s = r["shape_name"]
        if s not in by_shape:
            by_shape[s] = []
        by_shape[s].append(r)

    tuning_table_entries = []
    print(f"{'Shape':<26} {'N':<6} {'K':<6} {'Upstream':<10} {'Winner Candidate':<28} {'Winner Med':<12} {'Speedup':<10}")
    print("-" * 102)

    for s, runs in by_shape.items():
        n = runs[0]["N"]
        k = runs[0]["K"]
        
        # Upstream median
        u_run = next((r for r in runs if r["candidate_id"] == 0), None)
        u_med = u_run["median_us"] if u_run else 1.0

        # Exclude upstream from candidate winners
        custom_runs = [r for r in runs if r["candidate_id"] != 0 and r["candidate_id"] != 999]
        
        # Sort by median latency ascending
        custom_runs.sort(key=lambda r: r["median_us"])
        
        # Pick winner (if top candidates are within 0.5% noise tie, prefer lower WPR / zero smem)
        best = custom_runs[0]
        threshold = best["median_us"] * 1.005
        tied = [r for r in custom_runs if r["median_us"] <= threshold]
        
        # Tie-breaking priority: 1) zero shared memory (WPR=1), 2) lower warps/block
        tied.sort(key=lambda r: (
            0 if "WPR1" in r["variant_name"] else (1 if "WPR2" in r["variant_name"] else 2),
            r["candidate_id"]
        ))
        winner = tied[0]

        speedup = u_med / winner["median_us"]

        entry = {
            "shape_name": s,
            "N": n,
            "K": k,
            "architecture": 61,
            "winning_candidate_id": winner["candidate_id"],
            "winning_candidate_name": winner["variant_name"],
            "winning_median_us": winner["median_us"],
            "upstream_median_us": u_med,
            "speedup_vs_upstream": speedup,
            "selection_rationale": f"Fastest verified candidate (median {winner['median_us']:.2f} us vs upstream {u_med:.2f} us, +{(speedup-1)*100:.1f}%)"
        }
        tuning_table_entries.append(entry)

        print(f"{s:<26} {n:<6} {k:<6} {u_med:<10.2f} {winner['variant_name']:<28} {winner['median_us']:<12.2f} {speedup:<10.3f}x")

    table_path = os.path.join(results_dir, "tuning-table.json")
    with open(table_path, "w") as f:
        json.dump({
            "milestone": "M4",
            "device": data.get("device", {}),
            "safe_fallback": {
                "rule": "For shapes not explicitly tuned: if K % 256 == 0 and K == 2560 and N >= 8192 use Q4K_PASCAL_WPR1_WPB4; otherwise use Q4K_PASCAL_WPR2_WPB4; otherwise fallback to VARIANT_UPSTREAM_MMVQ.",
                "default_candidate_id": 204,
                "default_candidate_name": "Q4K_PASCAL_WPR2_WPB4"
            },
            "tuning_table": tuning_table_entries
        }, f, indent=2)

    print(f"\nSaved deterministic tuning table to: {table_path}\n")

if __name__ == "__main__":
    main()
