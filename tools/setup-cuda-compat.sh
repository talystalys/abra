#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
CUDA_TARGET_INC="${CUDA_ROOT}/targets/x86_64-linux/include"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPAT_DIR="${REPO_ROOT}/.cuda_compat"
OVERLAY_INC="${COMPAT_DIR}/include"
PATCH_FILE="${REPO_ROOT}/patches/cuda-12.9-glibc-2.41-math-functions.patch"

if [[ ! -d "${CUDA_TARGET_INC}" ]]; then
    echo "ERROR: CUDA include directory not found at ${CUDA_TARGET_INC}" >&2
    exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "ERROR: Patch file not found at ${PATCH_FILE}" >&2
    exit 1
fi

rm -rf "${COMPAT_DIR}"
mkdir -p "${OVERLAY_INC}/crt"

cp "${CUDA_TARGET_INC}/crt/math_functions.h" "${OVERLAY_INC}/crt/math_functions.h"
cp "${CUDA_TARGET_INC}/crt/math_functions.hpp" "${OVERLAY_INC}/crt/math_functions.hpp"

echo "Applying canonical patch: ${PATCH_FILE}"
if ! patch -p1 -d "${OVERLAY_INC}" < "${PATCH_FILE}"; then
    echo "ERROR: Failed to cleanly apply ${PATCH_FILE}" >&2
    exit 1
fi

for item in "${CUDA_TARGET_INC}/crt/"*; do
    fname="$(basename "${item}")"
    if [[ ! -f "${OVERLAY_INC}/crt/${fname}" ]]; then
        ln -sf "${item}" "${OVERLAY_INC}/crt/${fname}"
    fi
done

for item in "${CUDA_TARGET_INC}/"*; do
    fname="$(basename "${item}")"
    if [[ "${fname}" != "crt" && ! -e "${OVERLAY_INC}/${fname}" ]]; then
        ln -sf "${item}" "${OVERLAY_INC}/${fname}"
    fi
done

echo "CUDA compatibility overlay successfully established at: ${OVERLAY_INC}"

export PATH="${CUDA_ROOT}/bin:${PATH}"
export CUDACXX="${CUDA_ROOT}/bin/nvcc"
export CUDAFLAGS="-I${OVERLAY_INC} -Wno-deprecated-gpu-targets"
