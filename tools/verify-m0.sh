#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
NVCC="${CUDA_ROOT}/bin/nvcc"
CUOBJDUMP="${CUDA_ROOT}/bin/cuobjdump"

echo "============================================================"
echo "Milestone M0: Reproducible Baseline Verification"
echo "Host: $(hostname)"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo

# Gate Tracking
GTX_1060_DETECTED="NO"
GTX_1060_CC_VERIFIED="NO"
SM61_ARTIFACT_VERIFIED="NO"
PASCAL_EXECUTION_VERIFIED="NO"

GTX_1660_DETECTED="NO"
GTX_1660_CC_VERIFIED="NO"
SM75_ARTIFACT_VERIFIED="NO"
TURING_EXECUTION_VERIFIED="NO"

M0_STATUS="FAIL"

# Step 1: Establish CUDA compatibility layer
echo "--- Step 1: Setting up glibc compatibility overlay ---"
"${REPO_ROOT}/tools/setup-cuda-compat.sh"
echo

# Step 2: Configure & Build
echo "--- Step 2: Configuring and Building Project (sm_61 + sm_75) ---"
mkdir -p "${BUILD_DIR}"
CUDAFLAGS="-I${REPO_ROOT}/.cuda_compat/include -Wno-deprecated-gpu-targets" \
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --parallel
echo "Build completed successfully."
echo

# Step 3: Inspect Binary Architectures with cuobjdump
echo "--- Step 3: Inspecting Generated Binary Architectures ---"
CUOBJDUMP_OUTPUT="$("${CUOBJDUMP}" -lelf "${BUILD_DIR}/smoke_test" 2>&1)"
echo "${CUOBJDUMP_OUTPUT}"

if echo "${CUOBJDUMP_OUTPUT}" | grep -q "sm_61"; then
    SM61_ARTIFACT_VERIFIED="YES"
    echo "Verified sm_61 artifact present in binary."
else
    echo "ERROR: sm_61 artifact missing from binary!" >&2
fi

if echo "${CUOBJDUMP_OUTPUT}" | grep -q "sm_75"; then
    SM75_ARTIFACT_VERIFIED="YES"
    echo "Verified sm_75 artifact present in binary."
else
    echo "ERROR: sm_75 artifact missing from binary!" >&2
fi
echo

# Step 4: Enumerate Hardware Devices
echo "--- Step 4: Enumerating Visible CUDA Devices ---"
"${BUILD_DIR}/list_cuda_devices"

# Check physical presence of GPUs via nvidia-smi / device list
if nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | grep -qi "GTX 1060"; then
    GTX_1060_DETECTED="YES"
    if nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | grep -i "GTX 1060" | grep -q "6.1"; then
        GTX_1060_CC_VERIFIED="YES"
    fi
fi

if nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | grep -qi "GTX 1660"; then
    GTX_1660_DETECTED="YES"
    if nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | grep -i "GTX 1660" | grep -q "7.5"; then
        GTX_1660_CC_VERIFIED="YES"
    fi
fi
echo

# Step 5: Execute Correctness Suite on GTX 1060 (Primary Pascal Qualification)
echo "--- Step 5: Correctness Test on NVIDIA GeForce GTX 1060 6GB (Pascal SM 6.1) ---"
if [[ "${GTX_1060_DETECTED}" != "YES" ]]; then
    echo "BLOCKER: Physical GTX 1060 not detected. Refusing silent substitution." >&2
else
    if "${BUILD_DIR}/smoke_test" --target-arch sm_61; then
        PASCAL_EXECUTION_VERIFIED="YES"
        echo "Pascal physical CUDA correctness: PASS"
    else
        echo "Pascal physical CUDA correctness: FAIL" >&2
    fi
fi
echo

# Step 6: Execute Correctness Suite on GTX 1660 SUPER (Turing SM 7.5 Control)
echo "--- Step 6: Correctness Test on NVIDIA GeForce GTX 1660 SUPER (Turing SM 7.5) ---"
if [[ "${GTX_1660_DETECTED}" != "YES" ]]; then
    echo "BLOCKER: Physical GTX 1660 SUPER not detected. Refusing silent substitution." >&2
else
    if "${BUILD_DIR}/smoke_test" --target-arch sm_75; then
        TURING_EXECUTION_VERIFIED="YES"
        echo "Turing physical CUDA correctness: PASS"
    else
        echo "Turing physical CUDA correctness: FAIL" >&2
    fi
fi
echo

# Step 7: Benchmark Timing Harness Smoke Validation
echo "--- Step 7: Timing Harness Infrastructure Smoke Validation ---"
"${BUILD_DIR}/timer_smoke" --target-arch sm_61 --iterations 20 --warmup 5
echo

# Determine Overall M0 Status
if [[ "${GTX_1060_DETECTED}" == "YES" && \
      "${GTX_1060_CC_VERIFIED}" == "YES" && \
      "${SM61_ARTIFACT_VERIFIED}" == "YES" && \
      "${PASCAL_EXECUTION_VERIFIED}" == "YES" && \
      "${GTX_1660_DETECTED}" == "YES" && \
      "${GTX_1660_CC_VERIFIED}" == "YES" && \
      "${SM75_ARTIFACT_VERIFIED}" == "YES" && \
      "${TURING_EXECUTION_VERIFIED}" == "YES" ]]; then
    M0_STATUS="SUCCESS"
fi

echo "============================================================"
echo "M0 Environment Verification Summary"
echo "============================================================"
echo "M0 environment verification: ${M0_STATUS}"
echo
echo "GTX 1060 detected: ${GTX_1060_DETECTED}"
echo "GTX 1060 CC 6.1 verified: ${GTX_1060_CC_VERIFIED}"
echo "sm_61 artifact verified: ${SM61_ARTIFACT_VERIFIED}"
echo "Pascal physical CUDA execution: ${PASCAL_EXECUTION_VERIFIED}"
echo
echo "GTX 1660 SUPER detected: ${GTX_1660_DETECTED}"
echo "GTX 1660 SUPER CC 7.5 verified: ${GTX_1660_CC_VERIFIED}"
echo "sm_75 artifact verified: ${SM75_ARTIFACT_VERIFIED}"
echo "Turing control execution: ${TURING_EXECUTION_VERIFIED}"
echo "============================================================"

if [[ "${M0_STATUS}" != "SUCCESS" ]]; then
    exit 1
fi
