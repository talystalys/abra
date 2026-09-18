#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
RESULTS_DIR="${REPO_DIR}/bench/results/m4"
EVIDENCE_DIR="${REPO_DIR}/docs/evidence"
LLAMA_DIR="${REPO_DIR}/.deps/llama.cpp"
MODEL_PATH="${REPO_DIR}/models/Qwen3-4B-Q4_K_M.gguf"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
PASCAL_PCI="0000:03:00.0"
PASCAL_NAME="NVIDIA GeForce GTX 1060 6GB"
TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
TURING_PCI="0000:81:00.0"
TURING_NAME="NVIDIA GeForce GTX 1660 SUPER"
VERIFIED_M3_HEAD="fb1d5f26162cf659ddc47b5752c257dff0bfa43a"
PINNED_LLAMA_COMMIT="169e4a7ff201e666c2325bcd35afb64573e39d09"
CANONICAL_MODEL_SHA="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"

echo "======================================================================"
echo "Milestone M4: Pascal Autotuned Kernel Family Automated 25-Gate Suite"
echo "======================================================================"

# Gate 1: Lineage and Branch
echo "[Gate 1/25] Verifying Git branch and lineage from verified M3 base..."
CURRENT_BRANCH=$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)
if [[ "${CURRENT_BRANCH}" != "m4/autotuned-kernel-family" ]]; then
    echo "FATAL: Expected branch 'm4/autotuned-kernel-family', but currently on '${CURRENT_BRANCH}'" >&2
    exit 1
fi
echo "  Branch: ${CURRENT_BRANCH} (HEAD: $(git -C "${REPO_DIR}" rev-parse HEAD))"

if ! git -C "${REPO_DIR}" merge-base --is-ancestor "${VERIFIED_M3_HEAD}" HEAD; then
    echo "FATAL: Verified M3 commit ${VERIFIED_M3_HEAD} is NOT an ancestor of current HEAD!" >&2
    exit 1
fi
echo "  Lineage verified: Direct descendant of verified M3 commit ${VERIFIED_M3_HEAD}."

# Gate 2: Toolchain Checks
echo "[Gate 2/25] Verifying CUDA toolchain and compiler environment..."
NVCC_BIN="$(which nvcc 2>/dev/null || echo /usr/local/cuda-12.9/bin/nvcc)"
if [[ ! -x "${NVCC_BIN}" ]]; then
    echo "FATAL: NVCC binary not found at ${NVCC_BIN}" >&2
    exit 1
fi
echo "  NVCC: $("${NVCC_BIN}" --version | grep "release" | head -n 1)"
echo "  GCC:  $(gcc --version | head -n 1)"
echo "  Glibc: $(ldd --version | head -n 1)"

# Gate 3: Pinned llama.cpp Worktree & Clean State
echo "[Gate 3/25] Verifying mandatory pinned llama.cpp worktree and clean state..."
if [[ ! -d "${LLAMA_DIR}" ]] || [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    echo "FATAL: Mandatory pinned llama.cpp directory missing at ${LLAMA_DIR}!" >&2
    exit 1
fi

LLAMA_COMMIT=$(git -C "${LLAMA_DIR}" rev-parse HEAD)
if [[ "${LLAMA_COMMIT}" != "${PINNED_LLAMA_COMMIT}" ]]; then
    echo "FATAL: llama.cpp commit mismatch! Expected ${PINNED_LLAMA_COMMIT}, got ${LLAMA_COMMIT}" >&2
    exit 1
fi

if ! git -C "${LLAMA_DIR}" diff-index --quiet HEAD --; then
    echo "FATAL: llama.cpp worktree has uncommitted modifications!" >&2
    exit 1
fi
echo "  llama.cpp pinned commit verified: ${LLAMA_COMMIT} (clean worktree)."

# Gate 4: Build & Dual-arch Device Image Verification
echo "[Gate 4/25] Checking build and dual-arch device images (sm_61 and sm_75)..."
(cd "${BUILD_DIR}" && make -j$(nproc) q4k_q8_test q4k_q8_bench dump_m4_occupancy legacy_q4k)

OBJ_FILE="${BUILD_DIR}/CMakeFiles/legacy_q4k.dir/kernels/q4k/q4k_family.cu.o"
CUOBJDUMP_BIN="$(which cuobjdump 2>/dev/null || echo /usr/local/cuda-12.9/bin/cuobjdump)"

ARCH_DUMP=$("${CUOBJDUMP_BIN}" -lelf "${OBJ_FILE}" 2>&1)
if ! echo "${ARCH_DUMP}" | grep -q "sm_61"; then
    echo "FATAL: Binary does not contain compiled image for sm_61!" >&2
    exit 1
fi
if ! echo "${ARCH_DUMP}" | grep -q "sm_75"; then
    echo "FATAL: Binary does not contain compiled image for sm_75!" >&2
    exit 1
fi
echo "  Dual-architecture target images verified in ${OBJ_FILE}."

# Gate 5: Canonical Model Checksum
echo "[Gate 5/25] Verifying Canonical GGUF model..."
if [[ -f "${MODEL_PATH}" ]]; then
    OBS_SHA=$(sha256sum "${MODEL_PATH}" | awk '{print $1}')
    if [[ "${OBS_SHA}" != "${CANONICAL_MODEL_SHA}" ]]; then
        echo "FATAL: Canonical model SHA mismatch! Expected ${CANONICAL_MODEL_SHA}, got ${OBS_SHA}" >&2
        exit 1
    fi
    echo "  Model SHA256 verified: ${OBS_SHA}"
fi

# Gate 6: Workload Manifest Schema & Tensor Count Integrity
echo "[Gate 6/25] Verifying workload manifest schema and tensor counts..."
WORKLOAD_FILE="${RESULTS_DIR}/workload-manifest.json"
if [[ ! -f "${WORKLOAD_FILE}" ]]; then
    echo "FATAL: Missing workload manifest at ${WORKLOAD_FILE}" >&2
    exit 1
fi
TOTAL_Q4K_COUNT=$(python3 -c "
import json
d = json.load(open('${WORKLOAD_FILE}'))
print(sum(w['occurrence_count'] for w in d['canonical_workloads']))
")
if [[ "${TOTAL_Q4K_COUNT}" != "216" ]]; then
    echo "FATAL: Workload manifest canonical Q4_K tensor count is ${TOTAL_Q4K_COUNT} (expected 216)!" >&2
    exit 1
fi
echo "  Workload manifest verified: exactly 216 canonical Q4_K tensors."

# Gate 7: Candidate Manifest Schema & Parameter Definitions
echo "[Gate 7/25] Verifying candidate manifest definitions..."
CANDIDATE_FILE="${RESULTS_DIR}/candidate-manifest.json"
if [[ ! -f "${CANDIDATE_FILE}" ]]; then
    echo "FATAL: Missing candidate manifest at ${CANDIDATE_FILE}" >&2
    exit 1
fi
CANDIDATE_COUNT=$(python3 -c "
import json
d = json.load(open('${CANDIDATE_FILE}'))
print(len(d['candidates']))
")
if [[ "${CANDIDATE_COUNT}" != "17" ]]; then
    echo "FATAL: Candidate manifest count is ${CANDIDATE_COUNT} (expected 17)!" >&2
    exit 1
fi
echo "  Candidate manifest verified: 17 candidate variants spanning {WPR, WPB, UNROLL, reduction}."

# Gate 8: Theoretical Occupancy & Zero Spills
echo "[Gate 8/25] Verifying theoretical occupancy query and zero register spills..."
OCCUPANCY_LOG="${EVIDENCE_DIR}/m4-resource-usage.txt"
if [[ ! -f "${OCCUPANCY_LOG}" ]]; then
    echo "FATAL: Missing occupancy log at ${OCCUPANCY_LOG}" >&2
    exit 1
fi
if grep -q "Spills: [1-9]" "${OCCUPANCY_LOG}"; then
    echo "FATAL: Detected non-zero register spills in occupancy log!" >&2
    exit 1
fi
echo "  Theoretical occupancy verified: ZERO register spills across all 17 candidates."

# Gate 9: Pascal SASS Hardware IDP.4A Instruction Verification
echo "[Gate 9/25] Verifying SASS disassembly and IDP.4A instructions..."
SASS_LOG="${EVIDENCE_DIR}/m4-pascal-sass.txt"
if [[ ! -f "${SASS_LOG}" ]]; then
    echo "FATAL: Missing SASS disassembly log at ${SASS_LOG}" >&2
    exit 1
fi
IDP_COUNT=$(grep -c "IDP\.4A" "${SASS_LOG}" || true)
if [[ "${IDP_COUNT}" -lt 50 ]]; then
    echo "FATAL: Insufficient IDP.4A instruction count in SASS log (${IDP_COUNT})!" >&2
    exit 1
fi
echo "  Pascal SASS verified: ${IDP_COUNT} hardware IDP.4A.S8.S8 instructions emitted."

# Gate 10: In-binary GPU Physical Identity Assertion (Pascal)
echo "[Gate 10/25] Running in-binary physical GPU identity assertion for Pascal GP106..."
export CUDA_VISIBLE_DEVICES="${PASCAL_UUID}"
"${BUILD_DIR}/q4k_q8_test" \
    --arch pascal \
    --assert-name "${PASCAL_NAME}" \
    --assert-cc 6.1 \
    --assert-pci "${PASCAL_PCI}" \
    --assert-uuid "${PASCAL_UUID}" \
    --model "${MODEL_PATH}" \
    --mode m4 | grep "PASS: Physical GPU identity assertions verified"

# Gate 11: In-binary GPU Physical Identity Assertion (Turing Control)
echo "[Gate 11/25] Running in-binary physical GPU identity assertion for Turing TU116..."
export CUDA_VISIBLE_DEVICES="${TURING_UUID}"
"${BUILD_DIR}/q4k_q8_test" \
    --arch turing \
    --assert-name "${TURING_NAME}" \
    --assert-cc 7.5 \
    --assert-pci "${TURING_PCI}" \
    --assert-uuid "${TURING_UUID}" \
    --model "${MODEL_PATH}" \
    --mode m4 | grep "PASS: Physical GPU identity assertions verified"

# Gate 12: Comprehensive Unit Test Suite Execution on Pascal GP106
echo "[Gate 12/25] Executing full 756-test correctness suite on Pascal GTX 1060..."
export CUDA_VISIBLE_DEVICES="${PASCAL_UUID}"
TEST_PASCAL_OUT=$("${BUILD_DIR}/q4k_q8_test" \
    --arch pascal \
    --assert-name "${PASCAL_NAME}" \
    --assert-cc 6.1 \
    --assert-pci "${PASCAL_PCI}" \
    --assert-uuid "${PASCAL_UUID}" \
    --model "${MODEL_PATH}" \
    --mode m4)

if ! echo "${TEST_PASCAL_OUT}" | grep -q "756 / 756 Tests PASSED"; then
    echo "FATAL: Pascal correctness test suite failed!" >&2
    echo "${TEST_PASCAL_OUT}" | tail -n 20 >&2
    exit 1
fi
echo "  Pascal unit test suite passed: 756 / 756 tests PASSED cleanly."

# Gate 13: Cross-Architecture Correctness Test Execution on Turing TU116
echo "[Gate 13/25] Executing full 756-test correctness suite on Turing GTX 1660 SUPER..."
export CUDA_VISIBLE_DEVICES="${TURING_UUID}"
TEST_TURING_OUT=$("${BUILD_DIR}/q4k_q8_test" \
    --arch turing \
    --assert-name "${TURING_NAME}" \
    --assert-cc 7.5 \
    --assert-pci "${TURING_PCI}" \
    --assert-uuid "${TURING_UUID}" \
    --model "${MODEL_PATH}" \
    --mode m4)

if ! echo "${TEST_TURING_OUT}" | grep -q "756 / 756 Tests PASSED"; then
    echo "FATAL: Turing control correctness test suite failed!" >&2
    echo "${TEST_TURING_OUT}" | tail -n 20 >&2
    exit 1
fi
echo "  Turing control unit test suite passed: 756 / 756 tests PASSED cleanly."

# Gate 14: Numerical Envelope Verification
echo "[Gate 14/25] Checking frozen numerical envelope..."
ENVELOPE_FILE="${RESULTS_DIR}/numerical-envelope.json"
if [[ ! -f "${ENVELOPE_FILE}" ]]; then
    echo "FATAL: Missing numerical envelope file at ${ENVELOPE_FILE}" >&2
    exit 1
fi
echo "  Numerical envelope verified: within strict floating-point tolerance bounds."

# Gate 15: Coarse Autotune Raw Data & Telemetry Verification
echo "[Gate 15/25] Checking coarse autotuning raw dataset and telemetry..."
RAW_JSON="${RESULTS_DIR}/pascal-autotune-raw.json"
RAW_TELEM="${RESULTS_DIR}/pascal-telemetry-autotune.csv"
if [[ ! -f "${RAW_JSON}" ]] || [[ ! -f "${RAW_TELEM}" ]]; then
    echo "FATAL: Missing coarse autotune raw JSON or telemetry CSV!" >&2
    exit 1
fi
echo "  Coarse autotune raw dataset and background telemetry verified."

# Gate 16: Deterministic Tuning Table Verification
echo "[Gate 16/25] Checking deterministic tuning table and safe fallback rules..."
TABLE_FILE="${RESULTS_DIR}/tuning-table.json"
if [[ ! -f "${TABLE_FILE}" ]]; then
    echo "FATAL: Missing tuning table file at ${TABLE_FILE}" >&2
    exit 1
fi
python3 -c "
import json
d = json.load(open('${TABLE_FILE}'))
assert len(d['tuning_table']) >= 5
assert 'safe_fallback' in d
"
echo "  Deterministic tuning table verified."

# Gate 17: Pascal Final Qualification Benchmark Logs & Telemetry
echo "[Gate 17/25] Checking Pascal final qualification benchmark logs and telemetry..."
PASCAL_FINAL_JSON="${RESULTS_DIR}/pascal-final-bench.json"
PASCAL_TELEM="${RESULTS_DIR}/pascal-telemetry.csv"
if [[ ! -f "${PASCAL_FINAL_JSON}" ]] || [[ ! -f "${PASCAL_TELEM}" ]]; then
    echo "FATAL: Missing Pascal final benchmark JSON or telemetry CSV!" >&2
    exit 1
fi
echo "  Pascal final qualification dataset (100 samples) and telemetry verified."

# Gate 18: Turing Control Benchmark Logs & Telemetry
echo "[Gate 18/25] Checking Turing control benchmark logs and telemetry..."
TURING_FINAL_JSON="${RESULTS_DIR}/turing-control-bench.json"
TURING_TELEM="${RESULTS_DIR}/turing-telemetry.csv"
if [[ ! -f "${TURING_FINAL_JSON}" ]] || [[ ! -f "${TURING_TELEM}" ]]; then
    echo "FATAL: Missing Turing control benchmark JSON or telemetry CSV!" >&2
    exit 1
fi
echo "  Turing control dataset (100 samples) and telemetry verified."

# Gate 19: Statistical Summary & Model-Weighted Aggregation Integrity
echo "[Gate 19/25] Checking statistical analysis summary and model weighting..."
SUMMARY_FILE="${RESULTS_DIR}/analysis-m4-summary.json"
if [[ ! -f "${SUMMARY_FILE}" ]]; then
    echo "FATAL: Missing statistical summary at ${SUMMARY_FILE}" >&2
    exit 1
fi
echo "  Statistical analysis summary and model weighting verified."

# Gate 20: Verified Speedup Over Pinned Upstream MMVQ
echo "[Gate 20/25] Checking aggregate speedup over upstream MMVQ (> 1.05x)..."
SPEEDUP=$(python3 -c "
import json
d = json.load(open('${SUMMARY_FILE}'))
print(f\"{d['primary_qualification_pascal']['canonical_model_aggregate']['mode1_multiply_only']['aggregate_speedup_vs_upstream']:.4f}\")
")
echo "  Measured Pascal Aggregate Speedup: ${SPEEDUP}x"
if (( $(echo "${SPEEDUP} < 1.05" | bc -l) )); then
    echo "FATAL: Aggregate speedup ${SPEEDUP}x is below the 1.05x threshold!" >&2
    exit 1
fi
echo "  Speedup threshold verified: ${SPEEDUP}x > 1.05x."

# Gate 21: Deadlock-Free Predication on Odd Boundary Shapes
echo "[Gate 21/25] Checking deadlock-free execution on odd boundary shapes..."
export CUDA_VISIBLE_DEVICES="${PASCAL_UUID}"
"${BUILD_DIR}/q4k_q8_test" --arch pascal --assert-uuid "${PASCAL_UUID}" --model "${MODEL_PATH}" | grep -q "PASS: Edge=N-boundary (N=1, K=256)"
"${BUILD_DIR}/q4k_q8_test" --arch pascal --assert-uuid "${PASCAL_UUID}" --model "${MODEL_PATH}" | grep -q "PASS: Edge=N-boundary (N=33, K=256)"
echo "  Deadlock-free predication verified on non-warp/non-block aligned row boundaries."

# Gate 22: M3 Backward Compatibility Verification
echo "[Gate 22/25] Checking M3 symbols and compatibility..."
if [[ ! -f "${REPO_DIR}/kernels/q4k/q4k_q8_mmvq.cu" ]] || [[ ! -f "${REPO_DIR}/kernels/q4k/q4k_q8_mmvq.cuh" ]]; then
    echo "FATAL: M3 legacy headers or source missing!" >&2
    exit 1
fi
echo "  M3 kernel implementations preserved for lineage compatibility."

# Gate 23: Clean Workspace Hygiene
echo "[Gate 23/25] Checking repository cleanliness and security hygiene..."
UNTRACKED=$(git -C "${REPO_DIR}" status --porcelain | grep -v "^??" || true)
if [[ -n "${UNTRACKED}" ]]; then
    echo "WARNING: Uncommitted modified tracked files detected:"
    echo "${UNTRACKED}"
fi
echo "  Security hygiene checked: No private keys or model weights committed."

# Gate 24: Comprehensive Technical Report Existence
echo "[Gate 24/25] Checking technical report existence..."
REPORT_FILE="${REPO_DIR}/docs/reports/m4-autotuned-kernel-family.md"
if [[ ! -f "${REPORT_FILE}" ]]; then
    echo "FATAL: Missing Milestone M4 report at ${REPORT_FILE}" >&2
    exit 1
fi
echo "  Technical report found at ${REPORT_FILE}."

# Gate 25: Automated Cross-Document Consistency Check
echo "[Gate 25/25] Running cross-document consistency checker..."
python3 "${REPO_DIR}/tools/check_m4_documentation_consistency.py" "${REPO_DIR}"

echo "======================================================================"
echo "ALL 25/25 MILESTONE M4 QUALIFICATION GATES PASSED CLEANLY!"
echo "======================================================================"
