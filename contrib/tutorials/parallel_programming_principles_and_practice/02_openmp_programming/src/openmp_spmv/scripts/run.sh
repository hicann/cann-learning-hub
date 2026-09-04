#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [[ ! -x "${BUILD_DIR}/bin/spmv_benchmark" ]]; then
    bash "${ROOT_DIR}/scripts/build.sh"
fi

"${BUILD_DIR}/bin/spmv_benchmark" "$@"
