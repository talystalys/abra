#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
RESULTS_DIR="${REPO_DIR}/bench/results/m2"
EVIDENCE_DIR="${REPO_DIR}/docs/evidence"

PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
ACCEPTED_M1_HEAD="3ef60d1f11ba6d9e867b6108dad33d0c96d2195f"

echo "======================================================================"
echo "Milestone M2 Automated Qualification & Verification Suite (Hardened)"
echo "======================================================================"

echo "[1/8] Verifying Git branch and lineage from accepted M1 base..."
CURRENT_BRANCH=$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)
if [[ "${CURRENT_BRANCH}" != "m2/dp4a-microkernel" ]]; then
    echo "FATAL: Expected branch 'm2/dp4a-microkernel', but currently on '${CURRENT_BRANCH}'" >&2
    exit 1
fi
echo "  Branch: ${CURRENT_BRANCH} (HEAD: $(git -C "${REPO_DIR}" rev-parse HEAD))"

if ! git -C "${REPO_DIR}" merge-base --is-ancestor "${ACCEPTED_M1_HEAD}" HEAD; then
    echo "FATAL: Accepted M1 commit ${ACCEPTED_M1_HEAD} is NOT an ancestor of current HEAD!" >&2
    exit 1
fi
MERGE_BASE=$(git -C "${REPO_DIR}" merge-base "${ACCEPTED_M1_HEAD}" HEAD)
if [[ "${MERGE_BASE}" != "${ACCEPTED_M1_HEAD}" ]]; then
    echo "FATAL: Merge base with M1 head (${MERGE_BASE}) does not match expected (${ACCEPTED_M1_HEAD})!" >&2
    exit 1
fi
echo "  Lineage verified: Direct descendant of accepted M1 commit ${ACCEPTED_M1_HEAD}."

echo "[2/8] Verifying toolchain gates..."
NVCC_BIN="$(which nvcc 2>/dev/null || echo /usr/local/cuda-12.9/bin/nvcc)"
if [[ ! -x "${NVCC_BIN}" ]]; then
    echo "FATAL: NVCC binary not found or not executable at ${NVCC_BIN}" >&2
    exit 1
fi
echo "  NVCC: $("${NVCC_BIN}" --version | grep "release" | head -n 1)"
echo "  Host Compiler: $(gcc --version | head -n 1)"
echo "  glibc: $(ldd --version | head -n 1)"

echo "[3/8] Checking build and verifying dual-arch device images (sm_61 and sm_75)..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

OBJ_FILE="${BUILD_DIR}/CMakeFiles/legacy_dp4a.dir/kernels/dp4a/dp4a_gemv.cu.o"
CUOBJDUMP_BIN="$(which cuobjdump 2>/dev/null || echo /usr/local/cuda-12.9/bin/cuobjdump)"

ARCH_DUMP=$("${CUOBJDUMP_BIN}" -lelf "${OBJ_FILE}" 2>&1)
if ! echo "${ARCH_DUMP}" | grep -q "sm_61"; then
    echo "FATAL: Compiled object ${OBJ_FILE} is missing sm_61 device code!" >&2
    exit 1
fi
if ! echo "${ARCH_DUMP}" | grep -q "sm_75"; then
    echo "FATAL: Compiled object ${OBJ_FILE} is missing sm_75 device code!" >&2
    exit 1
fi
echo "  Verified: Object contains both sm_61 (Pascal) and sm_75 (Turing) ELF images."

echo "[4/8] Verifying Pascal SM 6.1 SASS disassembly instruction gates..."
"${CUOBJDUMP_BIN}" -sass -arch sm_61 "${OBJ_FILE}" > /tmp/verify_sass.txt

python3 -c '
with open("/tmp/verify_sass.txt") as f:
    text = f.read()
funcs = text.split("Function : ")
scalar_checked = False
dp4a_checked = False
for fn in funcs[1:]:
    lines = fn.splitlines()
    name = lines[0].strip()
    idp_count = sum(1 for l in lines if "IDP.4A" in l)
    if "gemv_scalar_kernel" in name:
        if idp_count != 0:
            raise RuntimeError(f"FAIL: gemv_scalar_kernel contains {idp_count} IDP.4A instructions! Must be 0.")
        print("  PASS: gemv_scalar_kernel contains 0 IDP.4A instructions (pure scalar arithmetic).")
        scalar_checked = True
    elif "gemv_dp4a_basic_kernel" in name:
        if idp_count == 0:
            raise RuntimeError("FAIL: gemv_dp4a_basic_kernel missing IDP.4A instructions!")
        print(f"  PASS: gemv_dp4a_basic_kernel contains {idp_count} IDP.4A instructions.")
        dp4a_checked = True
    elif "gemv_dp4a_opt_kernel" in name:
        if idp_count == 0:
            raise RuntimeError("FAIL: gemv_dp4a_opt_kernel missing IDP.4A instructions!")
        print(f"  PASS: gemv_dp4a_opt_kernel contains {idp_count} IDP.4A instructions.")
    elif "gemv_dp4a_shared_kernel" in name:
        if idp_count == 0:
            raise RuntimeError("FAIL: gemv_dp4a_shared_kernel missing IDP.4A instructions!")
        print(f"  PASS: gemv_dp4a_shared_kernel contains {idp_count} IDP.4A instructions.")

if not scalar_checked or not dp4a_checked:
    raise RuntimeError("FAIL: Missing expected kernel symbols in SASS dump!")
'

echo "[5/8] Running 156-test Correctness Suite on Pascal GTX 1060 (NUMA 0)..."
CUDA_VISIBLE_DEVICES="${PASCAL_UUID}" numactl --cpunodebind=0 --membind=0 \
    "${BUILD_DIR}/dp4a_test" --arch pascal

echo "[6/8] Running 156-test Correctness Suite on Turing GTX 1660 SUPER (NUMA 1)..."
CUDA_VISIBLE_DEVICES="${TURING_UUID}" numactl --cpunodebind=1 --membind=1 \
    "${BUILD_DIR}/dp4a_test" --arch turing

echo "[7/8] Parsing and verifying benchmark JSON integrity and statistics..."
python3 -c '
import json
import math
import sys
import os

def verify_bench_json(filepath, expected_arch, expected_name, expected_cc, expected_pci, expected_uuid):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Missing benchmark file: {filepath}")
    with open(filepath) as f:
        data = json.load(f)

    dev = data.get("device", {})
    actual_name = dev.get("name", "")
    actual_cc = dev.get("compute_capability", "")
    actual_pci = dev.get("pci_bus_id", "")
    actual_uuid = dev.get("uuid", "")
    if actual_name != expected_name:
        raise ValueError(f"Device name mismatch in {filepath}: expected {expected_name}, got {actual_name}")
    if actual_cc != expected_cc:
        raise ValueError(f"CC mismatch in {filepath}: expected {expected_cc}, got {actual_cc}")
    if actual_pci != expected_pci:
        raise ValueError(f"PCI mismatch in {filepath}: expected {expected_pci}, got {actual_pci}")
    if actual_uuid != expected_uuid:
        raise ValueError(f"UUID mismatch in {filepath}: expected {expected_uuid}, got {actual_uuid}")

    results = data.get("results", [])
    required_shapes = ["attn_q", "attn_kv", "attn_out", "ffn_gate_up", "ffn_down"]
    found_shapes = set(r["shape_name"] for r in results)
    for req in required_shapes:
        if req not in found_shapes:
            raise ValueError(f"Missing required shape {req} in {filepath}")

    for r in results:
        shape = r["shape_name"]
        var_name = r["variant_name"]
        lats = r.get("latencies_us", [])
        if len(lats) != 100:
            raise ValueError(f"Expected 100 latency samples for {shape} ({var_name}), got {len(lats)}")
        if r.get("warmup_iterations", 0) < 10:
            raise ValueError(f"Warmup reps < 10 for {shape} ({var_name})")
        if not r.get("verified_correct", False):
            raise ValueError(f"Correctness verification flag false for {shape} ({var_name})")

        for val in lats:
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"Invalid non-positive latency sample in {shape} ({var_name}): {val}")

        s = sorted(lats)
        n = len(s)
        recomp_min = s[0]
        recomp_max = s[-1]
        recomp_mean = sum(s) / n
        recomp_med = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        recomp_var = sum((x - recomp_mean) ** 2 for x in s) / (n - 1)
        recomp_sd = math.sqrt(recomp_var)
        recomp_spread = recomp_max - recomp_min

        eps = 1e-3
        if abs(r["min_us"] - recomp_min) > eps:
            raise ValueError(f"Min mismatch in {shape}: reported {r['min_us']} vs recomp {recomp_min}")
        if abs(r["max_us"] - recomp_max) > eps:
            raise ValueError(f"Max mismatch in {shape}: reported {r['max_us']} vs recomp {recomp_max}")
        if abs(r["mean_us"] - recomp_mean) > eps:
            raise ValueError(f"Mean mismatch in {shape}: reported {r['mean_us']} vs recomp {recomp_mean}")
        if abs(r["median_us"] - recomp_med) > eps:
            raise ValueError(f"Median mismatch in {shape}: reported {r['median_us']} vs recomp {recomp_med}")
        if abs(r["stddev_us"] - recomp_sd) > eps:
            raise ValueError(f"StdDev mismatch in {shape}: reported {r['stddev_us']} vs recomp {recomp_sd}")
        if abs(r["spread_us"] - recomp_spread) > eps:
            raise ValueError(f"Spread mismatch in {shape}: reported {r['spread_us']} vs recomp {recomp_spread}")

    print(f"  PASS: {filepath} verified valid ({len(results)} configurations, 100 samples each, statistics exact).")

verify_bench_json(
    sys.argv[1],
    "pascal",
    "NVIDIA GeForce GTX 1060 6GB",
    "6.1",
    "0000:03:00.0",
    "GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
)
verify_bench_json(
    sys.argv[2],
    "turing",
    "NVIDIA GeForce GTX 1660 SUPER",
    "7.5",
    "0000:81:00.0",
    "GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
)
' "${RESULTS_DIR}/pascal-dp4a-bench.json" "${RESULTS_DIR}/turing-dp4a-bench.json"

echo "[8/9] Verifying evidence files, telemetry traces, and live manifest..."
test -s "${RESULTS_DIR}/pascal-telemetry.csv"
test -s "${RESULTS_DIR}/turing-telemetry.csv"
test -s "${RESULTS_DIR}/analysis-m2-summary.json"
test -s "${RESULTS_DIR}/manifest.json"
test -s "${EVIDENCE_DIR}/m2-pascal-sass.txt"
test -s "${EVIDENCE_DIR}/m2-resource-usage.txt"
test -s "${EVIDENCE_DIR}/m2-profiler-attempt.txt"
test -d "${RESULTS_DIR}/original-423ef719"
test -s "${RESULTS_DIR}/original-423ef719/pascal-dp4a-bench.json"

python3 "${REPO_DIR}/tools/generate-m2-manifest.py" "${RESULTS_DIR}"

echo "[9/9] Verifying mechanical synchronization between reports and raw benchmark JSON..."
python3 "${REPO_DIR}/tools/check_m2_documentation_consistency.py" "${REPO_DIR}"

echo "======================================================================"
echo "MILESTONE M2 QUALIFICATION COMPLETE: ALL HARDENED CHECKS PASSED"
echo "======================================================================"
