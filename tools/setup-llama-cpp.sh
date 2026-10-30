#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="${REPO_ROOT}/.deps"
LLAMA_DIR="${DEPS_DIR}/llama.cpp"
BUILD_DEFAULT="${LLAMA_DIR}/build-default"
BUILD_CUBLAS="${LLAMA_DIR}/build-cublas"

PINNED_REPO="https://github.com/ggml-org/llama.cpp.git"
PINNED_COMMIT="169e4a7ff201e666c2325bcd35afb64573e39d09"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
NVCC="${CUDA_ROOT}/bin/nvcc"

echo "============================================================"
echo "Setting up Pinned llama.cpp Runtime"
echo "Repository: ${PINNED_REPO}"
echo "Commit:     ${PINNED_COMMIT}"
echo "============================================================"

"${REPO_ROOT}/tools/setup-cuda-compat.sh"
COMPAT_INC="${REPO_ROOT}/.cuda_compat/include"

mkdir -p "${DEPS_DIR}"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    echo "Cloning ${PINNED_REPO} into ${LLAMA_DIR}..."
    git clone "${PINNED_REPO}" "${LLAMA_DIR}"
fi

cd "${LLAMA_DIR}"
echo "Fetching and checking out pinned commit ${PINNED_COMMIT}..."
git fetch origin
git checkout -f "${PINNED_COMMIT}"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
    echo "ERROR: Checked out commit (${ACTUAL_COMMIT}) does not match pinned commit (${PINNED_COMMIT})!" >&2
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    echo "ERROR: Tracked source files in ${LLAMA_DIR} are modified!" >&2
    git status --short
    exit 1
fi

echo "Verified clean llama.cpp at pinned commit: ${ACTUAL_COMMIT}"

echo
echo "--- Building Default llama.cpp (MMQ / Default Dispatch) ---"
mkdir -p "${BUILD_DEFAULT}"
CUDAFLAGS="-I${COMPAT_INC} -Wno-deprecated-gpu-targets" \
cmake -S "${LLAMA_DIR}" -B "${BUILD_DEFAULT}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61;75" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF

cmake --build "${BUILD_DEFAULT}" --config Release --target llama-bench llama-cli --parallel "$(nproc)"
echo "Default build complete: ${BUILD_DEFAULT}/bin/llama-bench"

echo
echo "--- Building Forced-cuBLAS Diagnostic Build ---"
mkdir -p "${BUILD_CUBLAS}"
CUDAFLAGS="-I${COMPAT_INC} -Wno-deprecated-gpu-targets" \
cmake -S "${LLAMA_DIR}" -B "${BUILD_CUBLAS}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61;75" \
    -DCMAKE_CUDA_COMPILER="${NVCC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA_FORCE_CUBLAS=ON

cmake --build "${BUILD_CUBLAS}" --config Release --target llama-bench llama-cli --parallel "$(nproc)"
echo "Forced-cuBLAS build complete: ${BUILD_CUBLAS}/bin/llama-bench"

echo
echo "============================================================"
echo "llama.cpp Setup and Dual Build Complete"
echo "Default Binaries: ${BUILD_DEFAULT}/bin/"
echo "cuBLAS Binaries:  ${BUILD_CUBLAS}/bin/"
echo "============================================================"
