#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

ASCEND_HOME="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}"
if [[ -n "${ASCEND_HOME}" && -f "${ASCEND_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ASCEND_HOME}/set_env.sh"
fi

# A source CANN environment is required for a real build.  On a developer
# laptop without CANN, keep the project locally buildable with a diagnostic stub.
if [[ -z "${ASCEND_HOME}" || ! -f "${ASCEND_HOME}/include/acl/acl.h" ]]; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DACL_C_STUB=ON
else
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DASCEND_HOME="${ASCEND_HOME}" -DACL_C_STUB=OFF
fi
cmake --build "${BUILD_DIR}" --parallel
