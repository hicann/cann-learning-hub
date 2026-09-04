#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

ASCEND_GMRES_DIR="$(bash "${ROOT_DIR}/scripts/prepare_backend.sh")"
export ASCEND_GMRES_DIR

if [[ "${ASCEND_XYCE_FETCH_XYCE:-0}" == "1" ]]; then
    bash "${ROOT_DIR}/scripts/fetch_xyce.sh" || {
        echo "Warning: Xyce source clone failed; wrapper benchmark build will continue." >&2
        mkdir -p "${ROOT_DIR}/third_party/Xyce"
    }
fi

if [[ "${ASCEND_XYCE_BUILD_XYCE:-0}" == "1" ]]; then
    bash "${ROOT_DIR}/scripts/build_xyce.sh"
fi

command -v cmake >/dev/null 2>&1 || { echo "error: CMake is required for the Ascend Device build" >&2; exit 1; }
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DASCEND_GMRES_DIR="${ASCEND_GMRES_DIR}")
if [[ "${ASCEND_XYCE_HOST_ONLY:-0}" == "1" ]]; then CMAKE_ARGS+=(-DASCEND_XYCE_HOST_ONLY=ON); fi
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --config Release --parallel
