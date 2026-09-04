#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

ASCEND_HOME="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}"
if [[ -n "${ASCEND_HOME}" && -f "${ASCEND_HOME}/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME}/set_env.sh"
fi

if [[ ! -x "${BUILD_DIR}/bin/xyce_benchmark" ]]; then
    bash "${ROOT_DIR}/scripts/build.sh"
fi

"${BUILD_DIR}/bin/xyce_benchmark" \
    --warmup "${ASCEND_XYCE_WARMUP:-0}" \
    --repeat "${ASCEND_XYCE_REPEAT:-10}" \
    --matrix-dir "${ROOT_DIR}/matrices" \
    --results-dir "${ROOT_DIR}/results" \
    --csv "${ROOT_DIR}/results/xyce_benchmark.csv" \
    "$@"
