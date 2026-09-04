#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ASCEND_HOME="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}"
if [[ -n "${ASCEND_HOME}" && -f "${ASCEND_HOME}/set_env.sh" ]]; then
  source "${ASCEND_HOME}/set_env.sh"
fi
BUILD_DIR="${ROOT_DIR}/build"
if [[ "${HCCL_SPMV_STUB:-auto}" == "1" ]]; then STUB=ON; else STUB=OFF; fi
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DHCCL_SPMV_STUB="${STUB}"
if [[ "${HCCL_SPMV_REQUIRE_REAL:-0}" == "1" ]] &&
   ! grep -q '^HCCL_SPMV_HAS_CANN:INTERNAL=1$' "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null; then
  echo "error: real ACL+HCCL backend was required but CMake did not enable it" >&2
  exit 1
fi
cmake --build "${BUILD_DIR}" --parallel
