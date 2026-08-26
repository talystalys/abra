#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
RESULTS_DIR="${REPO_DIR}/bench/results/m2"
EVIDENCE_DIR="${REPO_DIR}/docs/evidence"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"

echo "======================================================================"
echo "Milestone M2 Automated Qualification & Verification Suite"
echo "======================================================================"

# Step 1: Check Lineage
echo "[1/7] Verifying Git branch and commit lineage..."
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "${CURRENT_BRANCH}" != "m2/dp4a-microkernel" ]]; then
    echo "WARNING: Expected branch m2/dp4a-microkernel, but currently on ${CURRENT_BRANCH}"
fi
echo "Branch: ${CURRENT_BRANCH} (HEAD: $(git rev-parse HEAD))"

# Step 2: Build Binaries
echo "[2/7] Checking build and compiler artifacts..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "Build succeeded for all targets (legacy_dp4a, dp4a_test, dp4a_bench)."

# Step 3: SASS & Resource Usage Verification
echo "[3/7] Verifying Pascal SM 6.1 SASS disassembly..."
OBJ_FILE="${BUILD_DIR}/CMakeFiles/legacy_dp4a.dir/kernels/dp4a/dp4a_gemv.cu.o"
/usr/local/cuda-12.9/bin/cuobjdump -sass -arch sm_61 "${OBJ_FILE}" > /tmp/verify_sass.txt

python3 -c '
with open("/tmp/verify_sass.txt") as f:
    text = f.read()
funcs = text.split("Function : ")
for fn in funcs[1:]:
    lines = fn.splitlines()
    name = lines[0].strip()
    idp_count = sum(1 for l in lines if "IDP.4A" in l)
    if "gemv_scalar_kernel" in name:
        if idp_count != 0:
            raise RuntimeError(f"FAIL: gemv_scalar_kernel contains {idp_count} IDP.4A instructions! Must be 0.")
        print("PASS: gemv_scalar_kernel contains 0 IDP.4A instructions (pure scalar arithmetic).")
    elif "gemv_dp4a_basic_kernel" in name:
        if idp_count == 0:
            raise RuntimeError("FAIL: gemv_dp4a_basic_kernel missing IDP.4A instructions!")
        print(f"PASS: gemv_dp4a_basic_kernel contains {idp_count} IDP.4A instructions.")
    elif "gemv_dp4a_opt_kernel" in name:
        if idp_count == 0:
            raise RuntimeError("FAIL: gemv_dp4a_opt_kernel missing IDP.4A instructions!")
        print(f"PASS: gemv_dp4a_opt_kernel contains {idp_count} IDP.4A instructions.")
    elif "gemv_dp4a_shared_kernel" in name:
        if idp_count == 0:
            raise RuntimeError("FAIL: gemv_dp4a_shared_kernel missing IDP.4A instructions!")
        print(f"PASS: gemv_dp4a_shared_kernel contains {idp_count} IDP.4A instructions.")
'

# Step 4: Correctness Suite on Pascal GTX 1060
echo "[4/7] Running 156-test Correctness Suite on Pascal GTX 1060 (NUMA 0)..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" numactl --cpunodebind=0 --membind=0 \
    "${BUILD_DIR}/dp4a_test"

# Step 5: Correctness Suite on Turing GTX 1660 SUPER
echo "[5/7] Running 156-test Correctness Suite on Turing GTX 1660 SUPER (NUMA 1)..."
CUDA_VISIBLE_DEVICES="${TURING_UUID}" numactl --cpunodebind=1 --membind=1 \
    "${BUILD_DIR}/dp4a_test"

# Step 6: Evidence & Benchmark Files Integrity
echo "[6/7] Verifying benchmark results and evidence artifacts..."
test -f "${RESULTS_DIR}/pascal-dp4a-bench.json"
test -f "${RESULTS_DIR}/turing-dp4a-bench.json"
test -f "${RESULTS_DIR}/pascal-telemetry.csv"
test -f "${RESULTS_DIR}/turing-telemetry.csv"
test -f "${RESULTS_DIR}/analysis-m2-summary.json"
test -f "${RESULTS_DIR}/manifest.json"
test -f "${EVIDENCE_DIR}/m2-pascal-sass.txt"
test -f "${EVIDENCE_DIR}/m2-resource-usage.txt"
test -f "${EVIDENCE_DIR}/m2-profiler-attempt.txt"
echo "All evidence and benchmark artifacts verified present."

# Step 7: Manifest Re-generation Check
echo "[7/7] Verifying live manifest generator..."
python3 "${REPO_DIR}/tools/generate-m2-manifest.py" "${RESULTS_DIR}"

echo "======================================================================"
echo "MILESTONE M2 QUALIFICATION COMPLETE: ALL CHECKS PASSED"
echo "======================================================================"
