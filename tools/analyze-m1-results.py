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
    
    # Conventional median
    if n % 2 == 1:
        median_val = s[n // 2]
    else:
        median_val = (s[n // 2 - 1] + s[n // 2]) / 2.0
        
    variance = sum((x - mean_val) ** 2 for x in s) / n
    std_dev = math.sqrt(variance)
    
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

def process_bench_file(file_path):
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None

    results = []
    # llama-bench outputs a list of benchmark objects
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "results" in data:
        items = data["results"]
    else:
        items = [data]

    for item in items:
        n_prompt = item.get("n_prompt", 0)
        n_gen = item.get("n_gen", 0)
        n_depth = item.get("n_depth", 0)
        
        # Determine test type: pp (prompt processing) or tg (token generation)
        test_type = "pp" if n_prompt > 0 and n_gen == 0 else ("tg" if n_gen > 0 else "unknown")
        
        # samples_ts is tokens per second array; samples_ns is nanoseconds array
        samples_ts = item.get("samples_ts", [])
        samples_ns = item.get("samples_ns", [])
        
        # If samples_ts not directly present, compute from samples_ns and token count
        if not samples_ts and samples_ns:
            tokens = n_prompt if test_type == "pp" else n_gen
            samples_ts = [(tokens / (ns / 1e9)) for ns in samples_ns if ns > 0]
            
        stats_ts = compute_stats(samples_ts)
        stats_ns = compute_stats(samples_ns)
        
        results.append({
            "test_type": test_type,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "n_depth": n_depth,
            "n_batch": item.get("n_batch", 0),
            "n_ubatch": item.get("n_ubatch", 0),
            "n_threads": item.get("n_threads", 0),
            "n_gpu_layers": item.get("n_gpu_layers", 0),
            "tokens_per_second": stats_ts,
            "duration_ns": stats_ns,
            "raw_item": item
        })
    return results

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench/results/m1"
    
    files = {
        "pascal_default_depth0": os.path.join(results_dir, "pascal-default-depth0.json"),
        "pascal_default_depth2048": os.path.join(results_dir, "pascal-default-depth2048.json"),
        "pascal_cublas_depth0": os.path.join(results_dir, "pascal-cublas-depth0.json"),
        "turing_default_depth0": os.path.join(results_dir, "turing-default-depth0.json"),
        "turing_default_depth2048": os.path.join(results_dir, "turing-default-depth2048.json"),
        "cpu_context_depth0": os.path.join(results_dir, "cpu-context-depth0.json"),
    }
    
    summary = {}
    for key, path in files.items():
        res = process_bench_file(path)
        summary[key] = res
        
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
