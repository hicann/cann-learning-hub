#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XYCE_SRC="${ROOT_DIR}/third_party/Xyce/source"
XYCE_BUILD="${ROOT_DIR}/build/xyce"

if [[ ! -f "${XYCE_SRC}/CMakeLists.txt" ]]; then
    bash "${ROOT_DIR}/scripts/fetch_xyce.sh"
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build upstream Xyce source" >&2
    exit 1
fi

cmake -S "${XYCE_SRC}" -B "${XYCE_BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${XYCE_BUILD}" --config Release --parallel
