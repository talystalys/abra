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

mkdir -p "${OVERLAY_INC}/crt"

# Copy and patch math_functions.h and math_functions.hpp
cp "${CUDA_TARGET_INC}/crt/math_functions.h" "${OVERLAY_INC}/crt/math_functions.h"
cp "${CUDA_TARGET_INC}/crt/math_functions.hpp" "${OVERLAY_INC}/crt/math_functions.hpp"

# Apply deterministic patch
sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) throw();/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) throw();/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) throw();/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) throw();/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/__func__(double sinpi(double a));/__func__(double sinpi(double a) throw());/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/__func__(double cospi(double a));/__func__(double cospi(double a) throw());/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/__func__(float sinpif(float a));/__func__(float sinpif(float a) throw());/' "${OVERLAY_INC}/crt/math_functions.h"
sed -i 's/__func__(float cospif(float a));/__func__(float cospif(float a) throw());/' "${OVERLAY_INC}/crt/math_functions.h"

sed -i 's/__func__(double sinpi(double a))/__func__(double sinpi(double a) throw())/' "${OVERLAY_INC}/crt/math_functions.hpp"
sed -i 's/__func__(double cospi(double a))/__func__(double cospi(double a) throw())/' "${OVERLAY_INC}/crt/math_functions.hpp"
sed -i 's/__func__(float sinpif(const float a))/__func__(float sinpif(const float a) throw())/' "${OVERLAY_INC}/crt/math_functions.hpp"
sed -i 's/__func__(float cospif(const float a))/__func__(float cospif(const float a) throw())/' "${OVERLAY_INC}/crt/math_functions.hpp"

# Symlink remainder of crt directory
for item in "${CUDA_TARGET_INC}/crt/"*; do
    fname="$(basename "${item}")"
    if [[ ! -f "${OVERLAY_INC}/crt/${fname}" ]]; then
        ln -sf "${item}" "${OVERLAY_INC}/crt/${fname}"
    fi
done

# Symlink top-level include items
for item in "${CUDA_TARGET_INC}/"*; do
    fname="$(basename "${item}")"
    if [[ "${fname}" != "crt" && ! -e "${OVERLAY_INC}/${fname}" ]]; then
        ln -sf "${item}" "${OVERLAY_INC}/${fname}"
    fi
done

echo "CUDA compatibility overlay established at: ${OVERLAY_INC}"
