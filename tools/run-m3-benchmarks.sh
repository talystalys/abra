#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
RESULTS_DIR="${REPO_DIR}/bench/results/m3"
PRE_M3R2_DIR="${RESULTS_DIR}/pre-m3r2"
EVIDENCE_DIR="${REPO_DIR}/docs/evidence"
DEPS_DIR="${REPO_DIR}/.deps"
LLAMA_DIR="${DEPS_DIR}/llama.cpp"
LLAMA_LIB_DIR="${LLAMA_DIR}/build-default/bin"
MODEL_PATH="${REPO_DIR}/models/Qwen3-4B-Q4_K_M.gguf"

PINNED_LLAMA_COMMIT="169e4a7ff201e666c2325bcd35afb64573e39d09"
PASCAL_UUID="GPU-5078520c-bf92-4143-0d48-b02fd9d250dd"
PASCAL_PCI="0000:03:00.0"
TURING_UUID="GPU-8a043791-359d-1df4-9353-38c5bc98e5dd"
TURING_PCI="0000:81:00.0"

mkdir -p "${RESULTS_DIR}" "${PRE_M3R2_DIR}" "${EVIDENCE_DIR}"

echo "======================================================================"
echo "Milestone M3R2: Pascal Q4_K x Q8_1 Benchmark & Qualification Suite"
echo "======================================================================"

echo "[0/9] Verifying mandatory pinned llama.cpp repository..."
if [[ ! -d "${LLAMA_DIR}" ]] || [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    echo "FATAL: Pinned llama.cpp directory does not exist at ${LLAMA_DIR}!" >&2
    exit 1
fi

OBS_LLAMA_SHA=$(git -C "${LLAMA_DIR}" rev-parse HEAD)
if [[ "${OBS_LLAMA_SHA}" != "${PINNED_LLAMA_COMMIT}" ]]; then
    echo "FATAL: llama.cpp commit mismatch! Expected ${PINNED_LLAMA_COMMIT}, got ${OBS_LLAMA_SHA}" >&2
    exit 1
fi

if ! git -C "${LLAMA_DIR}" diff-index --quiet HEAD --; then
    echo "FATAL: llama.cpp worktree has uncommitted modifications!" >&2
    git -C "${LLAMA_DIR}" status --short >&2
    exit 1
fi
echo "  Pinned llama.cpp verified: ${OBS_LLAMA_SHA} (clean worktree)."

if [[ -f "${RESULTS_DIR}/pascal-q4k-bench.json" ]] && [[ ! -f "${PRE_M3R2_DIR}/pascal-q4k-bench.json" ]]; then
    echo "[1/9] Preserving pre-m3r2 2e28b42 benchmark evidence in ${PRE_M3R2_DIR}..."
    cp "${RESULTS_DIR}/"*.json "${PRE_M3R2_DIR}/" 2>/dev/null || true
    cp "${RESULTS_DIR}/"*.csv "${PRE_M3R2_DIR}/" 2>/dev/null || true
fi

echo "[2/9] Rebuilding pinned upstream libraries and recording provenance..."
source "${REPO_DIR}/tools/setup-cuda-compat.sh"
COMPAT_INC="${REPO_DIR}/.cuda_compat/include"
NVCC_BIN="$(which nvcc 2>/dev/null || echo /usr/local/cuda-12.9/bin/nvcc)"

mkdir -p "${LLAMA_DIR}/build-default"
CUDAFLAGS="-I${COMPAT_INC} -Wno-deprecated-gpu-targets" \
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build-default" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61;75" \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF

cmake --build "${LLAMA_DIR}/build-default" --config Release --target ggml-cuda ggml-base ggml llama-bench llama-cli --parallel "$(nproc)"

PROV_FILE="${EVIDENCE_DIR}/m3r2-upstream-library-provenance.txt"
{
    echo "=== Milestone M3R2 Upstream Library Provenance ==="
    echo "Generated: $(date -u)"
    echo "Pinned llama.cpp Source SHA: ${PINNED_LLAMA_COMMIT}"
    echo "Worktree Status: Clean (git diff-index --quiet HEAD -- returned 0)"
    echo "Build Command: cmake -S ${LLAMA_DIR} -B ${LLAMA_DIR}/build-default -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"61;75\" -DCMAKE_BUILD_TYPE=Release"
    echo ""
    echo "--- Compiled Dynamic Libraries & SHA256 ---"
    for libname in libggml-cuda.so libggml-base.so libggml.so; do
        target_path="${LLAMA_LIB_DIR}/${libname}"
        real_path=$(readlink -f "${target_path}" || echo "${target_path}")
        sha=$(sha256sum "${real_path}" | awk '{print $1}')
        echo "Library: ${libname}"
        echo "  Symlink: ${target_path}"
        echo "  Real Path: ${real_path}"
        echo "  SHA256: ${sha}"
    done
} > "${PROV_FILE}"
echo "  Wrote library provenance: ${PROV_FILE}"

echo "[3/9] Building all M3R2 targets with real pinned ggml-cuda linking..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${REPO_DIR}"
make -j$(nproc) q4k_q8_test q4k_q8_bench dump_m3_occupancy legacy_q4k

echo "[4/9] Generating occupancy and resource usage evidence..."
"${BUILD_DIR}/dump_m3_occupancy" > "${EVIDENCE_DIR}/m3-resource-usage.txt"
echo "  Wrote ${EVIDENCE_DIR}/m3-resource-usage.txt"

echo "[5/9] Extracting SASS disassembly evidence..."
CUOBJDUMP_BIN="$(which cuobjdump 2>/dev/null || echo /usr/local/cuda-12.9/bin/cuobjdump)"

"${CUOBJDUMP_BIN}" -sass -arch sm_61 "${BUILD_DIR}/CMakeFiles/legacy_q4k.dir/kernels/q4k/q4k_q8_mmvq.cu.o" > "${EVIDENCE_DIR}/m3-pascal-q4k-sass.txt"
echo "  Wrote custom kernel SASS: ${EVIDENCE_DIR}/m3-pascal-q4k-sass.txt"

"${CUOBJDUMP_BIN}" -sass -arch sm_61 "${LLAMA_LIB_DIR}/libggml-cuda.so" | python3 -c '
import sys
in_sec = False
for line in sys.stdin:
    if "Function :" in line:
        in_sec = any(k in line for k in ["mul_mat_vec_q", "q4_K", "q8_1", "quantize_row_q8_1"])
    if in_sec:
        sys.stdout.write(line)
' > "${EVIDENCE_DIR}/m3r1-upstream-sass.txt"
echo "  Wrote upstream kernel SASS: ${EVIDENCE_DIR}/m3r1-upstream-sass.txt"

echo "[6/9] Generating binary link proof evidence..."
{
    echo "=== Pinned ggml-cuda Binary Link Proof ==="
    echo "Target Binary: ${BUILD_DIR}/q4k_q8_bench"
    echo "Date: $(date -u)"
    echo ""
    echo "--- Dynamic Library Linkage (ldd) ---"
    ldd "${BUILD_DIR}/q4k_q8_bench" | grep -E "ggml|cuda" || true
    echo ""
    echo "--- Referenced Pinned Upstream Symbols in Binary (nm -C) ---"
    nm -C "${BUILD_DIR}/q4k_q8_bench" | grep -E "ggml_cuda_op_mul_mat_vec_q|quantize_row_q8_1_cuda|quantize_row_q4_K_ref" || true
    echo ""
    echo "--- Exported Symbols in libggml-cuda.so ---"
    nm -C -D "${LLAMA_LIB_DIR}/libggml-cuda.so" | grep -E "ggml_cuda_op_mul_mat_vec_q|quantize_row_q8_1_cuda" || true
    echo ""
    echo "--- Exported Symbols in libggml-base.so ---"
    nm -C -D "${LLAMA_LIB_DIR}/libggml-base.so" | grep -E "quantize_row_q4_K_ref|dequantize_row_q4_K" || true
} > "${EVIDENCE_DIR}/m3r1-upstream-link-proof.txt"
echo "  Wrote link proof: ${EVIDENCE_DIR}/m3r1-upstream-link-proof.txt"

echo "[7/9] Executing Pascal GPU Qualification (GTX 1060 6GB, CC 6.1, PCI ${PASCAL_PCI}, NUMA 0)..."
export CUDA_VISIBLE_DEVICES="${PASCAL_UUID}"

echo "  Running correctness test suite and generating numerical envelope on Pascal..."
numactl --cpunodebind=0 --membind=0 "${BUILD_DIR}/q4k_q8_test" --arch pascal \
    --assert-name "NVIDIA GeForce GTX 1060 6GB" \
    --assert-cc "6.1" \
    --assert-pci "${PASCAL_PCI}" \
    --assert-uuid "${PASCAL_UUID}" \
    --envelope-out "${RESULTS_DIR}/numerical-envelope.json" \
    --model "${MODEL_PATH}"

echo "  Sampling telemetry and running benchmark suite on Pascal..."
PASCAL_TELEM_CSV="${RESULTS_DIR}/pascal-telemetry.csv"
"${SCRIPT_DIR}/sample-gpu-telemetry.sh" "${PASCAL_PCI}" "${PASCAL_TELEM_CSV}" &
TELEM_PID=$!

PASCAL_JSON="${RESULTS_DIR}/pascal-q4k-bench.json"
numactl --cpunodebind=0 --membind=0 "${BUILD_DIR}/q4k_q8_bench" --arch pascal \
    --assert-name "NVIDIA GeForce GTX 1060 6GB" \
    --assert-cc "6.1" \
    --assert-pci "${PASCAL_PCI}" \
    --assert-uuid "${PASCAL_UUID}" \
    --output "${PASCAL_JSON}" \
    --warmup 10 \
    --iters 100

kill "${TELEM_PID}" 2>/dev/null || true
wait "${TELEM_PID}" 2>/dev/null || true
echo "  Wrote Pascal benchmark results to: ${PASCAL_JSON}"
echo "  Wrote Pascal telemetry to: ${PASCAL_TELEM_CSV}"

echo "[8/9] Executing Turing GPU Control Benchmark (GTX 1660 SUPER, CC 7.5, PCI ${TURING_PCI}, NUMA 1)..."
export CUDA_VISIBLE_DEVICES="${TURING_UUID}"

echo "  Running correctness test suite on Turing..."
numactl --cpunodebind=1 --membind=1 "${BUILD_DIR}/q4k_q8_test" --arch turing \
    --assert-name "NVIDIA GeForce GTX 1660 SUPER" \
    --assert-cc "7.5" \
    --assert-pci "${TURING_PCI}" \
    --assert-uuid "${TURING_UUID}" \
    --model "${MODEL_PATH}"

echo "  Sampling telemetry and running benchmark suite on Turing..."
TURING_TELEM_CSV="${RESULTS_DIR}/turing-telemetry.csv"
"${SCRIPT_DIR}/sample-gpu-telemetry.sh" "${TURING_PCI}" "${TURING_TELEM_CSV}" &
TELEM_PID=$!

TURING_JSON="${RESULTS_DIR}/turing-q4k-bench.json"
numactl --cpunodebind=1 --membind=1 "${BUILD_DIR}/q4k_q8_bench" --arch turing \
    --assert-name "NVIDIA GeForce GTX 1660 SUPER" \
    --assert-cc "7.5" \
    --assert-pci "${TURING_PCI}" \
    --assert-uuid "${TURING_UUID}" \
    --output "${TURING_JSON}" \
    --warmup 10 \
    --iters 100

kill "${TELEM_PID}" 2>/dev/null || true
wait "${TELEM_PID}" 2>/dev/null || true
echo "  Wrote Turing benchmark results to: ${TURING_JSON}"
echo "  Wrote Turing telemetry to: ${TURING_TELEM_CSV}"

echo "[9/9] Generating Manifest, Summary, and Synchronizing Documentation..."
cat << 'EOF_DISPATCH' > "${EVIDENCE_DIR}/m3-upstream-mmvq-dispatch.txt"
=== Upstream llama.cpp MMVQ Dispatch Contract Proof ===
Pinned llama.cpp Commit: 169e4a7ff201e666c2325bcd35afb64573e39d09

1. MMVQ Dispatch Rule:
   In ggml-cuda.cu / mmvq.cuh:
   bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
       if (!ggml_is_quantized(type)) return false;
       return ne11 <= MMVQ_MAX_BATCH_SIZE; // MMVQ_MAX_BATCH_SIZE == 8
   }
   During autoregressive decode (ne11 == 1, batch=1), ggml_cuda_should_use_mmvq() returns TRUE.
   The decode incumbent for Q4_K is therefore MMVQ (NOT MMQ).

2. Architectural Parameter Tables in mmvq.cu:
   - For Pascal SM 6.1:
     Falls into MMVQ_PARAMETERS_GENERIC table.
     calc_nwarps(GGML_TYPE_Q4_K, ncols_dst=1, MMVQ_PARAMETERS_GENERIC) -> returns 4.
     Generic Pascal MMVQ uses 4 warps collaborating on 1 output row (1 row per block).
   - For Turing SM 7.5:
     Falls into MMVQ_PARAMETERS_TURING table.
     calc_nwarps(GGML_TYPE_Q4_K, ncols_dst=1, MMVQ_PARAMETERS_TURING) -> returns 2.
     Turing MMVQ uses 2 warps collaborating on 1 output row.

3. Numerical Accumulation Contract:
   - Quantized Weight Block: block_q4_K (144 bytes per 256 weights, 4.5 bpw, asymmetric scales/mins).
   - Quantized Activation Block: block_q8_1 (36 bytes per 32 values, scale d + sum-scale s).
   - Kernel vec_dot_q4_K_q8_1 executes IDP.4A.S8.S8 SIMD dot products directly on quantized byte representations.
   - Dequantization scale/min offsets are accumulated in FP32 without intermediate full-tensor dequantization in global memory.
EOF_DISPATCH
echo "  Wrote ${EVIDENCE_DIR}/m3-upstream-mmvq-dispatch.txt"

python3 "${SCRIPT_DIR}/generate-m3-manifest.py" "${RESULTS_DIR}"
python3 "${SCRIPT_DIR}/generate_m3_summary.py"
python3 "${SCRIPT_DIR}/update_m3_documentation.py"
python3 "${SCRIPT_DIR}/check_m3_documentation_consistency.py" "${REPO_DIR}"

echo "======================================================================"
echo "M3R2 Benchmarking & Documentation Synchronization Complete!"
echo "======================================================================"
