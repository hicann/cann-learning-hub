#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake is not installed or not in PATH" >&2
    exit 1
fi
if ! command -v mpicxx >/dev/null 2>&1; then
    echo "error: MPI C++ compiler wrapper (mpicxx) is not installed or not in PATH" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)"
cmake --build "${BUILD_DIR}" --config Release --parallel

echo "Built ${BUILD_DIR}/spmv_cpu and ${BUILD_DIR}/spmv_mpi"
