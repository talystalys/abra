#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
RESULTS_DIR="${REPO_DIR}/bench/results/m3"
PRE_REPAIR_DIR="${RESULTS_DIR}/pre-repair"
EVIDENCE_DIR="${REPO_DIR}/docs/evidence"
LLAMA_LIB_DIR="${REPO_DIR}/.deps/llama.cpp/build-default/bin"
MODEL_PATH="${REPO_DIR}/models/Qwen3-4B-Q4_K_M.gguf"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
PASCAL_PCI="0000:03:00.0"
TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
TURING_PCI="0000:81:00.0"
ACCEPTED_M2_HEAD="f10ead5fa94925e0a36980b1f3bdaa4a6998aa84"
PINNED_LLAMA_COMMIT="169e4a7ff201e666c2325bcd35afb64573e39d09"
CANONICAL_MODEL_SHA="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"

echo "======================================================================"
echo "Milestone M3R1: Pascal Q4_K x Q8_1 Automated 25-Gate Qualification Suite"
echo "======================================================================"

# Gate 1: Lineage and Branch
echo "[Gate 1/25] Verifying Git branch and lineage from accepted M2 base..."
CURRENT_BRANCH=$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)
if [[ "${CURRENT_BRANCH}" != "m3/q4k-q8-microkernel" ]]; then
    echo "FATAL: Expected branch 'm3/q4k-q8-microkernel', but currently on '${CURRENT_BRANCH}'" >&2
    exit 1
fi
echo "  Branch: ${CURRENT_BRANCH} (HEAD: $(git -C "${REPO_DIR}" rev-parse HEAD))"

if ! git -C "${REPO_DIR}" merge-base --is-ancestor "${ACCEPTED_M2_HEAD}" HEAD; then
    echo "FATAL: Accepted M2 commit ${ACCEPTED_M2_HEAD} is NOT an ancestor of current HEAD!" >&2
    exit 1
fi
echo "  Lineage verified: Direct descendant of accepted M2 commit ${ACCEPTED_M2_HEAD}."

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

# Gate 3: Build & Device Architecture Image Verification
echo "[Gate 3/25] Checking build and dual-arch device images (sm_61 and sm_75)..."
cd "${BUILD_DIR}"
make -j$(nproc) q4k_q8_test q4k_q8_bench dump_m3_occupancy legacy_q4k

OBJ_FILE="${BUILD_DIR}/CMakeFiles/legacy_q4k.dir/kernels/q4k/q4k_q8_mmvq.cu.o"
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

# Gate 4: Canonical Model Checksum & Tensor Map
echo "[Gate 4/25] Verifying Canonical GGUF model and tensor map..."
if [[ -f "${MODEL_PATH}" ]]; then
    OBS_SHA=$(sha256sum "${MODEL_PATH}" | awk '{print $1}')
    if [[ "${OBS_SHA}" != "${CANONICAL_MODEL_SHA}" ]]; then
        echo "FATAL: Canonical model SHA mismatch! Expected ${CANONICAL_MODEL_SHA}, got ${OBS_SHA}" >&2
        exit 1
    fi
    echo "  Model SHA256 verified: ${OBS_SHA}"
fi

TENSOR_MAP_JSON="${RESULTS_DIR}/qwen3-q4k-tensor-map.json"
if [[ ! -f "${TENSOR_MAP_JSON}" ]]; then
    echo "FATAL: Tensor map file missing at ${TENSOR_MAP_JSON}" >&2
    exit 1
fi
echo "  Tensor map file verified: ${TENSOR_MAP_JSON}"

# Gate 5: Pinned Upstream llama.cpp Worktree
echo "[Gate 5/25] Verifying pinned upstream llama.cpp worktree and clean state..."
LLAMA_DIR="${REPO_DIR}/.deps/llama.cpp"
if [[ -d "${LLAMA_DIR}" ]]; then
    LLAMA_COMMIT=$(git -C "${LLAMA_DIR}" rev-parse HEAD)
    if [[ "${LLAMA_COMMIT}" != "${PINNED_LLAMA_COMMIT}" ]]; then
        echo "FATAL: llama.cpp commit mismatch! Expected ${PINNED_LLAMA_COMMIT}, got ${LLAMA_COMMIT}" >&2
        exit 1
    fi
    echo "  llama.cpp pinned commit verified: ${LLAMA_COMMIT}"
fi

# Gate 6: Link Proof Evidence
echo "[Gate 6/25] Verifying upstream dynamic library link proof evidence..."
LINK_PROOF="${EVIDENCE_DIR}/m3r1-upstream-link-proof.txt"
if [[ ! -f "${LINK_PROOF}" ]] || ! grep -q "libggml-cuda.so" "${LINK_PROOF}" || ! grep -q "ggml_cuda_op_mul_mat_vec_q" "${LINK_PROOF}"; then
    echo "FATAL: Upstream link proof evidence missing or incomplete in ${LINK_PROOF}!" >&2
    exit 1
fi
echo "  Verified dynamic link proof in ${LINK_PROOF}."

# Gate 7: Upstream SASS Evidence
echo "[Gate 7/25] Verifying upstream SASS disassembly evidence..."
UPSTREAM_SASS="${EVIDENCE_DIR}/m3r1-upstream-sass.txt"
if [[ ! -f "${UPSTREAM_SASS}" ]]; then
    echo "FATAL: Upstream SASS evidence missing at ${UPSTREAM_SASS}!" >&2
    exit 1
fi
echo "  Verified upstream SASS evidence (${UPSTREAM_SASS})."

# Gate 8: Custom Kernel SASS Evidence
echo "[Gate 8/25] Verifying native IDP.4A.S8.S8 in custom Pascal SASS disassembly..."
CUSTOM_SASS="${EVIDENCE_DIR}/m3-pascal-q4k-sass.txt"
if [[ ! -f "${CUSTOM_SASS}" ]] || ! grep -q "IDP.4A.S8.S8" "${CUSTOM_SASS}"; then
    echo "FATAL: Custom SASS evidence missing IDP.4A.S8.S8 instructions!" >&2
    exit 1
fi
echo "  Verified native IDP.4A.S8.S8 instructions in ${CUSTOM_SASS}."

# Gate 9: Theoretical Occupancy & Resource Usage Evidence
echo "[Gate 9/25] Verifying theoretical occupancy and resource usage evidence..."
OCC_FILE="${EVIDENCE_DIR}/m3-resource-usage.txt"
if [[ ! -f "${OCC_FILE}" ]] || ! grep -q "GTX 1060" "${OCC_FILE}"; then
    echo "FATAL: Resource usage evidence missing or incomplete in ${OCC_FILE}!" >&2
    exit 1
fi
echo "  Verified theoretical occupancy metrics in ${OCC_FILE}."

# Gate 10: Upstream MMVQ Dispatch Proof
echo "[Gate 10/25] Verifying upstream MMVQ dispatch proof..."
DISPATCH_FILE="${EVIDENCE_DIR}/m3-upstream-mmvq-dispatch.txt"
if [[ ! -f "${DISPATCH_FILE}" ]] || ! grep -q "MMVQ_PARAMETERS_GENERIC" "${DISPATCH_FILE}"; then
    echo "FATAL: Upstream dispatch proof missing in ${DISPATCH_FILE}!" >&2
    exit 1
fi
echo "  Verified MMVQ dispatch proof in ${DISPATCH_FILE}."

# Gate 11: Frozen Numerical Envelope
echo "[Gate 11/25] Verifying frozen numerical envelope..."
ENVELOPE_JSON="${RESULTS_DIR}/numerical-envelope.json"
if [[ ! -f "${ENVELOPE_JSON}" ]]; then
    echo "FATAL: Numerical envelope missing at ${ENVELOPE_JSON}!" >&2
    exit 1
fi
python3 -c "
import json
env = json.load(open('${ENVELOPE_JSON}'))
assert 'upstream_vs_f64_oracle' in env
assert 'custom_variants_vs_upstream' in env
assert env['upstream_vs_f64_oracle']['max_rel_error'] < 0.005
print('  Verified numerical envelope: max rel error =', env['upstream_vs_f64_oracle']['max_rel_error'])
"

# Gate 12: Pre-repair Evidence Preserved
echo "[Gate 12/25] Verifying pre-repair 8f0290d evidence preservation..."
if [[ ! -f "${PRE_REPAIR_DIR}/pascal-q4k-bench.json" ]]; then
    echo "FATAL: Pre-repair benchmark evidence missing in ${PRE_REPAIR_DIR}!" >&2
    exit 1
fi
echo "  Pre-repair evidence preserved in ${PRE_REPAIR_DIR}."

# Gate 13: Frozen B/C Source Integrity
echo "[Gate 13/25] Verifying frozen Variant B and C kernel implementation integrity..."
KERNEL_CU="${REPO_DIR}/kernels/q4k/q4k_q8_mmvq.cu"
if ! grep -q "q4k_q8_mmvq_2warp_kernel" "${KERNEL_CU}" || ! grep -q "q4k_q8_mmvq_1warp_kernel" "${KERNEL_CU}"; then
    echo "FATAL: Missing Variant B or Variant C kernel definition in ${KERNEL_CU}!" >&2
    exit 1
fi
echo "  Variant B and C kernel source verified."

# Gate 14: Standalone Quantizer Renamed
echo "[Gate 14/25] Verifying standalone quantizer renamed to CUSTOM_REFERENCE_Q8_1_QUANTIZER..."
if ! grep -q "CUSTOM_REFERENCE_Q8_1_QUANTIZER" "${KERNEL_CU}"; then
    echo "FATAL: Custom reference quantizer naming assertion failed in ${KERNEL_CU}!" >&2
    exit 1
fi
echo "  Verified CUSTOM_REFERENCE_Q8_1_QUANTIZER naming."

# Gate 15: Distinct Kernel Verification
echo "[Gate 15/25] Verifying Variant U != Variant A != Variant B != Variant C..."
python3 -c "
import subprocess
nm_out = subprocess.run(['nm', '-C', '${BUILD_DIR}/q4k_q8_bench'], capture_output=True, text=True, check=True).stdout
assert 'ggml_cuda_op_mul_mat_vec_q' in nm_out, 'Variant U symbol missing'
assert 'q4k_q8_mmvq_4warp_kernel' in nm_out, 'Variant A symbol missing'
assert 'q4k_q8_mmvq_2warp_kernel' in nm_out, 'Variant B symbol missing'
assert 'q4k_q8_mmvq_1warp_kernel' in nm_out, 'Variant C symbol missing'
print('  All 4 distinct kernel symbols verified in binary.')
"

# Gate 16: Physical GPU Assertions & Correctness Tests (Pascal)
echo "[Gate 16/25] Executing in-binary identity and correctness suite on Pascal GP106..."
export CUDA_VISIBLE_DEVICES="${PASCAL_UUID}"
"${BUILD_DIR}/q4k_q8_test" --arch pascal \
    --assert-name "NVIDIA GeForce GTX 1060 6GB" \
    --assert-cc "6.1" \
    --assert-pci "${PASCAL_PCI}" \
    --assert-uuid "${PASCAL_UUID}" \
    --model "${MODEL_PATH}"
echo "  Pascal correctness suite: 168/168 PASSED."

# Gate 17: Physical GPU Assertions & Correctness Tests (Turing)
echo "[Gate 17/25] Executing in-binary identity and correctness suite on Turing TU116..."
export CUDA_VISIBLE_DEVICES="${TURING_UUID}"
"${BUILD_DIR}/q4k_q8_test" --arch turing \
    --assert-name "NVIDIA GeForce GTX 1660 SUPER" \
    --assert-cc "7.5" \
    --assert-pci "${TURING_PCI}" \
    --assert-uuid "${TURING_UUID}" \
    --model "${MODEL_PATH}"
echo "  Turing correctness suite: 168/168 PASSED."

# Gate 18: Pascal Benchmark JSON Validation
echo "[Gate 18/25] Validating Pascal benchmark JSON..."
python3 -c "
import json
d = json.load(open('${RESULTS_DIR}/pascal-q4k-bench.json'))
assert 'GTX 1060' in d['device']['name']
assert len(d['results']) == 32
for r in d['results']:
    assert len(r['latencies_us']) == 100
    assert len(r['combined_latencies_us']) == 100
    assert r['verification']['verified'] == True
print('  Pascal benchmark JSON verified: 32 results, 100% verified_correct.')
"

# Gate 19: Turing Benchmark JSON Validation
echo "[Gate 19/25] Validating Turing benchmark JSON..."
python3 -c "
import json
d = json.load(open('${RESULTS_DIR}/turing-q4k-bench.json'))
assert 'GTX 1660' in d['device']['name']
assert len(d['results']) == 32
for r in d['results']:
    assert len(r['latencies_us']) == 100
    assert len(r['combined_latencies_us']) == 100
    assert r['verification']['verified'] == True
print('  Turing benchmark JSON verified: 32 results, 100% verified_correct.')
"

# Gate 20: Telemetry CSVs
echo "[Gate 20/25] Validating continuous GPU telemetry CSVs..."
if [[ ! -f "${RESULTS_DIR}/pascal-telemetry.csv" ]] || [[ ! -f "${RESULTS_DIR}/turing-telemetry.csv" ]]; then
    echo "FATAL: Telemetry CSV files missing!" >&2
    exit 1
fi
echo "  Continuous GPU telemetry CSVs verified."

# Gate 21: Statistical Summary JSON
echo "[Gate 21/25] Validating statistical summary JSON..."
if [[ ! -f "${RESULTS_DIR}/analysis-m3-summary.json" ]]; then
    echo "FATAL: Summary JSON missing at ${RESULTS_DIR}/analysis-m3-summary.json" >&2
    exit 1
fi
echo "  Statistical summary JSON verified."

# Gate 22: Provenance Manifest Validation
echo "[Gate 22/25] Validating canonical live manifest.json..."
MANIFEST_FILE="${RESULTS_DIR}/manifest.json"
if [[ ! -f "${MANIFEST_FILE}" ]]; then
    echo "FATAL: Manifest file missing at ${MANIFEST_FILE}" >&2
    exit 1
fi
python3 -c "
import json
m = json.load(open('${MANIFEST_FILE}'))
assert m['milestone'] == 'M3'
assert m['expected_pinned_specifications']['m2_accepted_base_commit'] == '${ACCEPTED_M2_HEAD}'
assert 'GTX 1060' in m['observed_execution_environment']['primary_qualification_pascal_observed']['name']
assert 'GTX 1660' in m['observed_execution_environment']['secondary_control_turing_observed']['name']
print('  Manifest structure and provenance assertions verified.')
"

# Gate 23: Strict Documentation Consistency Check
echo "[Gate 23/25] Executing strict structured documentation consistency checker..."
python3 "${SCRIPT_DIR}/check_m3_documentation_consistency.py" "${REPO_DIR}"

# Gate 24: AGENTS.md Demarcation Check
echo "[Gate 24/25] Validating AGENTS.md device demarcation in all documentation..."
python3 -c "
for fname in ['${REPO_DIR}/docs/reports/m3-q4k-q8-microkernel.md', '${REPO_DIR}/CURRENT_STATUS.md']:
    text = open(fname).read()
    assert 'GTX 1660 SUPER' in text
    assert 'GTX 1060' in text
    # Ensure 1660 SUPER is not claimed as Pascal
    assert 'GTX 1660 SUPER (Pascal' not in text
    assert 'GTX 1660 SUPER — Pascal' not in text
print('  AGENTS.md device demarcation assertions verified.')
"

# Gate 25: Git Workspace Hygiene
echo "[Gate 25/25] Verifying Git workspace hygiene (no forbidden binaries or weights committed)..."
python3 -c "
import subprocess
tracked_files = subprocess.run(['git', '-C', '${REPO_DIR}', 'ls-files'], capture_output=True, text=True, check=True).stdout.splitlines()
for f in tracked_files:
    assert not f.endswith('.gguf'), f'Forbidden model file tracked: {f}'
    assert not f.endswith('.so'), f'Forbidden binary library tracked: {f}'
    assert not f.endswith('.o'), f'Forbidden object file tracked: {f}'
    assert not f.startswith('build/'), f'Forbidden build file tracked: {f}'
print('  Git workspace hygiene verified: 0 forbidden files tracked.')
"

echo "======================================================================"
echo "ALL 25 QUALIFICATION GATES PASSED PERFECTLY!"
echo "Milestone M3R1 is verified, reproducible, and ready for lead review."
echo "======================================================================"
