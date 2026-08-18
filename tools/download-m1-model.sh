#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
MODEL_NAME="Qwen3-4B-Q4_K_M.gguf"
TARGET_FILE="${MODELS_DIR}/${MODEL_NAME}"

MODEL_URL="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/${MODEL_NAME}"
EXPECTED_SHA256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"

echo "============================================================"
echo "M1 Pinned Model Verification & Downloader"
echo "Model:    ${MODEL_NAME}"
echo "Expected: ${EXPECTED_SHA256}"
echo "============================================================"

mkdir -p "${MODELS_DIR}"

check_hash() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        return 1
    fi
    echo "Computing SHA-256 for ${file}..."
    local actual_hash
    actual_hash="$(sha256sum "${file}" | awk '{print $1}')"
    echo "Actual SHA-256:   ${actual_hash}"
    echo "Expected SHA-256: ${EXPECTED_SHA256}"
    if [[ "${actual_hash}" == "${EXPECTED_SHA256}" ]]; then
        return 0
    else
        return 1
    fi
}

if check_hash "${TARGET_FILE}"; then
    echo "SUCCESS: Model already exists and SHA-256 matches exactly."
    exit 0
fi

echo "Downloading ${MODEL_NAME} from ${MODEL_URL}..."
TMP_DOWNLOAD="${TARGET_FILE}.tmp"
rm -f "${TMP_DOWNLOAD}"

curl -L --fail --retry 5 --retry-delay 2 -C - -o "${TMP_DOWNLOAD}" "${MODEL_URL}"

if check_hash "${TMP_DOWNLOAD}"; then
    mv "${TMP_DOWNLOAD}" "${TARGET_FILE}"
    echo "SUCCESS: Downloaded and verified ${TARGET_FILE}"
else
    echo "ERROR: Downloaded file checksum mismatch!" >&2
    rm -f "${TMP_DOWNLOAD}"
    exit 1
fi
