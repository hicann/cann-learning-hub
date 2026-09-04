#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if ! command -v clang++ >/dev/null 2>&1; then
    echo "error: Kunpeng BiSheng Host compiler clang++ is not installed or not in PATH" >&2
    exit 1
fi
compiler_version="$(clang++ --version 2>&1)"
if ! grep -qi bisheng <<<"${compiler_version}"; then
    echo "error: clang++ is not the Kunpeng BiSheng Host compiler" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$(command -v clang++)"
cmake --build "${BUILD_DIR}" --config Release --parallel
