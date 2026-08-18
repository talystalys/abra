#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/bench/results/m1"
MODELS_DIR="${REPO_ROOT}/models"
MODEL_FILE="${MODELS_DIR}/Qwen3-4B-Q4_K_M.gguf"
EXPECTED_SHA256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"

LLAMA_DIR="${REPO_ROOT}/.deps/llama.cpp"
PINNED_COMMIT="169e4a7ff201e666c2325bcd35afb64573e39d09"
LLAMA_BENCH_DEFAULT="${LLAMA_DIR}/build-default/bin/llama-bench"
LLAMA_CLI_DEFAULT="${LLAMA_DIR}/build-default/bin/llama-cli"
LLAMA_BENCH_CUBLAS="${LLAMA_DIR}/build-cublas/bin/llama-bench"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
PASCAL_PCI="0000:03:00.0"
PASCAL_NUMA=0

TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
TURING_PCI="0000:81:00.0"
TURING_NUMA=1

echo "============================================================"
echo "Milestone M1: llama.cpp Baseline Benchmark Suite"
echo "Host: $(hostname)"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# Verify pinned llama.cpp commit and clean source
if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    echo "llama.cpp repository missing at ${LLAMA_DIR}" >&2
    echo "Running setup script..."
    "${REPO_ROOT}/tools/setup-llama-cpp.sh"
fi

cd "${LLAMA_DIR}"
ACTUAL_LLAMA_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_LLAMA_COMMIT}" != "${PINNED_COMMIT}" ]]; then
    echo "FATAL: llama.cpp commit mismatch!" >&2
    echo "Expected: ${PINNED_COMMIT}" >&2
    echo "Actual:   ${ACTUAL_LLAMA_COMMIT}" >&2
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    echo "FATAL: Tracked source files in ${LLAMA_DIR} are modified!" >&2
    git status --short
    exit 1
fi
echo "Verified clean llama.cpp source at pinned commit: ${ACTUAL_LLAMA_COMMIT}"
cd "${REPO_ROOT}"

# Verify model hash
if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "Model file missing: ${MODEL_FILE}" >&2
    echo "Running model download script..."
    "${REPO_ROOT}/tools/download-m1-model.sh"
fi

ACTUAL_SHA256="$(sha256sum "${MODEL_FILE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "FATAL: Model hash mismatch!" >&2
    echo "Expected: ${EXPECTED_SHA256}" >&2
    echo "Actual:   ${ACTUAL_SHA256}" >&2
    exit 1
fi
echo "Verified model: ${MODEL_FILE} (SHA-256: ${ACTUAL_SHA256})"

mkdir -p "${RESULTS_DIR}"

run_with_telemetry() {
    local gpu_uuid="$1"
    local output_telemetry="$2"
    local cmd_log="$3"
    shift 3
    
    # Start telemetry background sampler
    local sampler_pid
    "${REPO_ROOT}/tools/sample-gpu-telemetry.sh" "${gpu_uuid}" "${output_telemetry}" 0.25 &
    sampler_pid=$!
    
    # Let sampler start
    sleep 1
    
    # Run command and capture output
    echo "Executing: $*"
    if ! "$@" > "${cmd_log}"; then
        echo "Command failed: $*" >&2
        kill -9 "${sampler_pid}" 2>/dev/null || true
        return 1
    fi
    
    # Stop sampler
    sleep 1
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
    echo "Completed. Log saved to: ${cmd_log}"
}

# Idle status inspection
echo
echo "--- GPU Pre-Benchmark State ---"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,temperature.gpu,power.limit,clocks.sm,clocks.mem --format=csv
echo

# -----------------------------------------------------------------------------
# 1. Pascal Default Baseline (GTX 1060 6GB)
# -----------------------------------------------------------------------------
echo "============================================================"
echo "1. Pascal Default Baseline (GTX 1060 6GB, CC 6.1)"
echo "   Isolation: ${PASCAL_UUID} | NUMA: ${PASCAL_NUMA}"
echo "============================================================"

# Pascal depth 0
echo "--> Running Pascal Default @ depth 0 (pp512, tg128, 10 reps)..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" \
run_with_telemetry "${PASCAL_UUID}" \
    "${RESULTS_DIR}/pascal-default-depth0-telemetry.csv" \
    "${RESULTS_DIR}/pascal-default-depth0.json" \
    numactl --cpunodebind="${PASCAL_NUMA}" --membind="${PASCAL_NUMA}" \
    "${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -d 0 \
    -b 2048 -ub 512 \
    -ctk f16 -ctv f16 \
    -ngl 99 \
    -r 10 \
    -o json

# Pascal depth 2048
echo "--> Running Pascal Default @ depth 2048 (pp512, tg128, 10 reps)..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" \
run_with_telemetry "${PASCAL_UUID}" \
    "${RESULTS_DIR}/pascal-default-depth2048-telemetry.csv" \
    "${RESULTS_DIR}/pascal-default-depth2048.json" \
    numactl --cpunodebind="${PASCAL_NUMA}" --membind="${PASCAL_NUMA}" \
    "${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -d 2048 \
    -b 2048 -ub 512 \
    -ctk f16 -ctv f16 \
    -ngl 99 \
    -r 10 \
    -o json

# Pascal Forced-cuBLAS Diagnostic
echo "--> Running Pascal Forced-cuBLAS Diagnostic @ depth 0 (pp512, tg128, 10 reps)..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" \
run_with_telemetry "${PASCAL_UUID}" \
    "${RESULTS_DIR}/pascal-cublas-depth0-telemetry.csv" \
    "${RESULTS_DIR}/pascal-cublas-depth0.json" \
    numactl --cpunodebind="${PASCAL_NUMA}" --membind="${PASCAL_NUMA}" \
    "${LLAMA_BENCH_CUBLAS}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -d 0 \
    -b 2048 -ub 512 \
    -ctk f16 -ctv f16 \
    -ngl 99 \
    -r 10 \
    -o json

# Pascal llama-cli End-to-End Sanity Check
echo "--> Running Pascal llama-cli End-to-End Sanity Check..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" \
numactl --cpunodebind="${PASCAL_NUMA}" --membind="${PASCAL_NUMA}" \
"${LLAMA_CLI_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p "Explain in one short paragraph why memory bandwidth matters for neural-network inference." \
    -n 128 -s 42 --temp 0 \
    -ngl 99 \
    -st --simple-io > "${RESULTS_DIR}/pascal-cli-sanity.txt" 2>&1
echo "Pascal CLI Sanity output written to: ${RESULTS_DIR}/pascal-cli-sanity.txt"

# Pascal Offload Proof Diagnostics
echo "--> Generating Pascal Offload Proof Diagnostics..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" \
numactl --cpunodebind="${PASCAL_NUMA}" --membind="${PASCAL_NUMA}" \
"${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -ngl 99 -r 1 -v > "${RESULTS_DIR}/pascal-offload-proof.txt" 2>&1
echo "Pascal Offload Proof written to: ${RESULTS_DIR}/pascal-offload-proof.txt"

# -----------------------------------------------------------------------------
# 2. Turing Control Baseline (GTX 1660 SUPER)
# -----------------------------------------------------------------------------
echo "============================================================"
echo "2. Turing Control Baseline (GTX 1660 SUPER, CC 7.5)"
echo "   Isolation: ${TURING_UUID} | NUMA: ${TURING_NUMA}"
echo "============================================================"

# Turing depth 0
echo "--> Running Turing Default @ depth 0 (pp512, tg128, 10 reps)..."
CUDA_VISIBLE_DEVICES="${TURING_UUID}" \
run_with_telemetry "${TURING_UUID}" \
    "${RESULTS_DIR}/turing-default-depth0-telemetry.csv" \
    "${RESULTS_DIR}/turing-default-depth0.json" \
    numactl --cpunodebind="${TURING_NUMA}" --membind="${TURING_NUMA}" \
    "${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -d 0 \
    -b 2048 -ub 512 \
    -ctk f16 -ctv f16 \
    -ngl 99 \
    -r 10 \
    -o json

# Turing depth 2048
echo "--> Running Turing Default @ depth 2048 (pp512, tg128, 10 reps)..."
CUDA_VISIBLE_DEVICES="${TURING_UUID}" \
run_with_telemetry "${TURING_UUID}" \
    "${RESULTS_DIR}/turing-default-depth2048-telemetry.csv" \
    "${RESULTS_DIR}/turing-default-depth2048.json" \
    numactl --cpunodebind="${TURING_NUMA}" --membind="${TURING_NUMA}" \
    "${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -d 2048 \
    -b 2048 -ub 512 \
    -ctk f16 -ctv f16 \
    -ngl 99 \
    -r 10 \
    -o json

# Turing CLI Sanity Check
echo "--> Running Turing llama-cli End-to-End Sanity Check..."
CUDA_VISIBLE_DEVICES="${TURING_UUID}" \
numactl --cpunodebind="${TURING_NUMA}" --membind="${TURING_NUMA}" \
"${LLAMA_CLI_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p "Explain in one short paragraph why memory bandwidth matters for neural-network inference." \
    -n 128 -s 42 --temp 0 \
    -ngl 99 \
    -st --simple-io > "${RESULTS_DIR}/turing-cli-sanity.txt" 2>&1
echo "Turing CLI Sanity output written to: ${RESULTS_DIR}/turing-cli-sanity.txt"

# Turing Offload Proof Diagnostics
echo "--> Generating Turing Offload Proof Diagnostics..."
CUDA_VISIBLE_DEVICES="${TURING_UUID}" \
numactl --cpunodebind="${TURING_NUMA}" --membind="${TURING_NUMA}" \
"${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -ngl 99 -r 1 -v > "${RESULTS_DIR}/turing-offload-proof.txt" 2>&1
echo "Turing Offload Proof written to: ${RESULTS_DIR}/turing-offload-proof.txt"

# -----------------------------------------------------------------------------
# 3. CPU Context Baseline (Context Only)
# -----------------------------------------------------------------------------
echo "============================================================"
echo "3. CPU Context Baseline (CONTEXT ONLY — NOT PRIMARY COMPETITOR)"
echo "============================================================"
echo "--> Running CPU Context @ depth 0 (pp512, tg128, 3 reps, ngl 0)..."
numactl --cpunodebind=0 --membind=0 \
"${LLAMA_BENCH_DEFAULT}" \
    -m "${MODEL_FILE}" \
    -p 512 -n 128 -d 0 \
    -b 2048 -ub 512 \
    -ctk f16 -ctv f16 \
    -ngl 0 \
    -t 32 \
    -r 3 \
    -o json > "${RESULTS_DIR}/cpu-context-depth0.json"
echo "CPU Context output written to: ${RESULTS_DIR}/cpu-context-depth0.json"

# -----------------------------------------------------------------------------
# 4. Processing Statistics and Manifest
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo "4. Processing Benchmark Statistics and Manifest"
echo "============================================================"
python3 "${REPO_ROOT}/tools/analyze-m1-results.py" "${RESULTS_DIR}" > "${RESULTS_DIR}/analysis-summary.json"
python3 "${REPO_ROOT}/tools/generate-m1-manifest.py" "${RESULTS_DIR}"
echo "Saved analysis summary to: ${RESULTS_DIR}/analysis-summary.json"

echo
echo "============================================================"
echo "Milestone M1 Baseline Run Completed Successfully"
echo "============================================================"
