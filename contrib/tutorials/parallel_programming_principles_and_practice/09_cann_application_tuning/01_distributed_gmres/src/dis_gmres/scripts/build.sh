#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)

if [[ "${DIS_GMRES_STUB:-0}" == "1" ]]; then
  CMAKE_ARGS+=(-DDIS_GMRES_FORCE_STUB=ON)
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}" "$@"
if [[ "${DIS_GMRES_REQUIRE_REAL:-0}" == "1" ]] &&
   ! grep -q '^DIS_GMRES_HAS_CANN:INTERNAL=1$' "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null; then
  echo "error: real ACL+HCCL backend was required but CMake did not enable it" >&2
  exit 1
fi
cmake --build "${BUILD_DIR}" --config Release --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
