#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
RESULTS_DIR="${REPO_DIR}/bench/results/m2"
mkdir -p "${RESULTS_DIR}"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"

start_telemetry() {
    local target_uuid="$1"
    local output_csv="$2"
    echo "timestamp,gpu_name,gpu_uuid,pci_bus_id,pstate,clocks.gr_mhz,clocks.mem_mhz,power.draw_w,temperature.gpu,utilization.gpu_pct,utilization.memory_pct,memory.used_mib,memory.total_mib" > "${output_csv}"
    (
        while true; do
            nvidia-smi -i "${target_uuid}" \
                --query-gpu=timestamp,name,uuid,pci.bus_id,pstate,clocks.gr,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> "${output_csv}" || true
            sleep 0.25
        done
    ) &
    echo $!
}

cleanup_process() {
    local pid="$1"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        kill -KILL "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

TEL_PID_PASCAL=""
TEL_PID_TURING=""

cleanup_all() {
    cleanup_process "${TEL_PID_PASCAL}"
    cleanup_process "${TEL_PID_TURING}"
}
trap cleanup_all EXIT INT TERM

echo "======================================================================"
echo "Starting Canonical M2 Microbenchmark on Pascal GTX 1060 (NUMA node 0)"
echo "======================================================================"
PASCAL_JSON="${RESULTS_DIR}/pascal-dp4a-bench.json"
PASCAL_CSV="${RESULTS_DIR}/pascal-telemetry.csv"

TEL_PID_PASCAL=$(start_telemetry "${PASCAL_UUID}" "${PASCAL_CSV}")
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" numactl --cpunodebind=0 --membind=0 \
    "${BUILD_DIR}/dp4a_bench" "${PASCAL_JSON}" 10 100 --arch pascal
cleanup_process "${TEL_PID_PASCAL}"
TEL_PID_PASCAL=""
echo "Completed Pascal benchmark. Output saved to ${PASCAL_JSON}"

echo "======================================================================"
echo "Starting Canonical M2 Microbenchmark on Turing GTX 1660 SUPER (NUMA node 1)"
echo "======================================================================"
TURING_JSON="${RESULTS_DIR}/turing-dp4a-bench.json"
TURING_CSV="${RESULTS_DIR}/turing-telemetry.csv"

TEL_PID_TURING=$(start_telemetry "${TURING_UUID}" "${TURING_CSV}")
CUDA_VISIBLE_DEVICES="${TURING_UUID}" numactl --cpunodebind=1 --membind=1 \
    "${BUILD_DIR}/dp4a_bench" "${TURING_JSON}" 10 100 --arch turing

CUDA_VISIBLE_DEVICES="${TURING_UUID}" numactl --cpunodebind=1 --membind=1 \
    "${BUILD_DIR}/dp4a_bench" "" 10 100 --arch turing > /dev/null 2>&1 || true

cleanup_process "${TEL_PID_TURING}"
TEL_PID_TURING=""
echo "Completed Turing benchmark. Output saved to ${TURING_JSON}"

echo "======================================================================"
echo "Analyzing benchmark results and generating live M2 manifest"
echo "======================================================================"
python3 "${REPO_DIR}/tools/analyze-m2-results.py" "${RESULTS_DIR}" > "${RESULTS_DIR}/analysis-m2-summary.json"
python3 "${REPO_DIR}/tools/generate-m2-manifest.py" "${RESULTS_DIR}"

echo "M2 canonical benchmark run completed successfully."
