#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
EVIDENCE_DIR="${REPO_ROOT}/docs/evidence"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
NVCC="${CUDA_ROOT}/bin/nvcc"
CUOBJDUMP="${CUDA_ROOT}/bin/cuobjdump"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
PASCAL_PCI="0000:03:00.0"
PASCAL_NAME="GeForce GTX 1060 6GB"
PASCAL_CC="6.1"

TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
TURING_PCI="0000:81:00.0"
TURING_NAME="GeForce GTX 1660 SUPER"
TURING_CC="7.5"

echo "============================================================"
echo "Milestone M0: Reproducible Baseline Verification"
echo "Host: $(hostname)"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo

mkdir -p "${EVIDENCE_DIR}"

echo "--- Step 1: Collecting Live Environment Inventory ---"
"${REPO_ROOT}/tools/collect-hardware.sh" > "${EVIDENCE_DIR}/hardware-inventory-raw.txt" 2>&1
echo "Saved raw hardware inventory to: docs/evidence/hardware-inventory-raw.txt"
echo

echo "--- Step 2: Applying Canonical CUDA Compatibility Patch ---"
"${REPO_ROOT}/tools/setup-cuda-compat.sh"
echo

echo "--- Step 3: Clean Configure and Build (sm_61 + sm_75) ---"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
CUDAFLAGS="-I${REPO_ROOT}/.cuda_compat/include -Wno-deprecated-gpu-targets" \
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --parallel
echo "Build completed successfully."
echo

echo "--- Step 4: Inspecting Generated Binary Architectures ---"
SM61_ARTIFACT_VERIFIED="NO"
SM75_ARTIFACT_VERIFIED="NO"

CUOBJDUMP_OUTPUT="$("${CUOBJDUMP}" -lelf "${BUILD_DIR}/smoke_test" 2>&1)"
echo "${CUOBJDUMP_OUTPUT}"

if echo "${CUOBJDUMP_OUTPUT}" | grep -q "sm_61"; then
    SM61_ARTIFACT_VERIFIED="YES"
    echo "Verified sm_61 artifact present in binary."
else
    echo "ERROR: sm_61 artifact missing from binary!" >&2
    exit 1
fi

if echo "${CUOBJDUMP_OUTPUT}" | grep -q "sm_75"; then
    SM75_ARTIFACT_VERIFIED="YES"
    echo "Verified sm_75 artifact present in binary."
else
    echo "ERROR: sm_75 artifact missing from binary!" >&2
    exit 1
fi
echo

echo "--- Step 5: Enumerating Visible CUDA Devices ---"
"${BUILD_DIR}/list_cuda_devices"
echo

echo "--- Step 6: Pascal Correctness Qualification (Pinned: ${PASCAL_UUID} / ${PASCAL_PCI}) ---"
PASCAL_DETECTED="NO"
PASCAL_CC_VERIFIED="NO"
PASCAL_EXECUTION_VERIFIED="NO"

if "${BUILD_DIR}/smoke_test" \
    --uuid "${PASCAL_UUID}" \
    --assert-uuid "${PASCAL_UUID}" \
    --assert-pci "${PASCAL_PCI}" \
    --assert-name "${PASCAL_NAME}" \
    --assert-cc "${PASCAL_CC}"; then
    PASCAL_DETECTED="YES"
    PASCAL_CC_VERIFIED="YES"
    PASCAL_EXECUTION_VERIFIED="YES"
    echo "Pascal physical CUDA correctness (GTX 1060 6GB): PASS"
else
    echo "ERROR: Pascal correctness qualification failed on ${PASCAL_UUID}" >&2
    exit 1
fi
echo

echo "--- Step 7: Turing Control Correctness (Pinned: ${TURING_UUID} / ${TURING_PCI}) ---"
TURING_DETECTED="NO"
TURING_CC_VERIFIED="NO"
TURING_EXECUTION_VERIFIED="NO"

if "${BUILD_DIR}/smoke_test" \
    --uuid "${TURING_UUID}" \
    --assert-uuid "${TURING_UUID}" \
    --assert-pci "${TURING_PCI}" \
    --assert-name "${TURING_NAME}" \
    --assert-cc "${TURING_CC}"; then
    TURING_DETECTED="YES"
    TURING_CC_VERIFIED="YES"
    TURING_EXECUTION_VERIFIED="YES"
    echo "Turing physical CUDA correctness (GTX 1660 SUPER): PASS"
else
    echo "ERROR: Turing control correctness failed on ${TURING_UUID}" >&2
    exit 1
fi
echo

echo "--- Step 8: Timing Harness Infrastructure Smoke Validation on Pascal GPU ---"
"${BUILD_DIR}/timer_smoke" \
    --uuid "${PASCAL_UUID}" \
    --assert-uuid "${PASCAL_UUID}" \
    --assert-pci "${PASCAL_PCI}" \
    --assert-name "${PASCAL_NAME}" \
    --assert-cc "${PASCAL_CC}" \
    --iterations 20 \
    --warmup 5
echo

M0_STATUS="FAIL"
if [[ "${PASCAL_DETECTED}" == "YES" && \
      "${PASCAL_CC_VERIFIED}" == "YES" && \
      "${SM61_ARTIFACT_VERIFIED}" == "YES" && \
      "${PASCAL_EXECUTION_VERIFIED}" == "YES" && \
      "${TURING_DETECTED}" == "YES" && \
      "${TURING_CC_VERIFIED}" == "YES" && \
      "${SM75_ARTIFACT_VERIFIED}" == "YES" && \
      "${TURING_EXECUTION_VERIFIED}" == "YES" ]]; then
    M0_STATUS="SUCCESS"
fi

echo "============================================================"
echo "M0 Environment Verification Summary"
echo "============================================================"
echo "M0 environment verification: ${M0_STATUS}"
echo
echo "GTX 1060 detected: ${PASCAL_DETECTED}"
echo "GTX 1060 CC 6.1 verified: ${PASCAL_CC_VERIFIED}"
echo "sm_61 artifact verified: ${SM61_ARTIFACT_VERIFIED}"
echo "Pascal physical CUDA execution: ${PASCAL_EXECUTION_VERIFIED}"
echo
echo "GTX 1660 SUPER detected: ${TURING_DETECTED}"
echo "GTX 1660 SUPER CC 7.5 verified: ${TURING_CC_VERIFIED}"
echo "sm_75 artifact verified: ${SM75_ARTIFACT_VERIFIED}"
echo "Turing control execution: ${TURING_EXECUTION_VERIFIED}"
echo "============================================================"

if [[ "${M0_STATUS}" != "SUCCESS" ]]; then
    exit 1
fi
