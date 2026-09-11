#!/usr/bin/env bash
set -eo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
ASC_ARCH=${ASC_ARCH:-dav-2201}
ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR="${SCRIPT_DIR}/../../../source/06.04_super_kernel/custom_launch_sk"

PYTHON_EXEC=$(command -v "${PYTHON_BIN}" || true)
if [ -z "${PYTHON_EXEC}" ]; then
    echo "未找到 Python 解释器: ${PYTHON_BIN}" >&2
    exit 1
fi

cmake -S "${WORK_DIR}" -B "${WORK_DIR}/build"     -DCMAKE_ASC_ARCHITECTURES="${ASC_ARCH}"     -DPython3_EXECUTABLE="${PYTHON_EXEC}"
cmake --build "${WORK_DIR}/build" --parallel

(
    cd "${WORK_DIR}"
    export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID}"
    export NPU_DEVICE_ID="${NPU_DEVICE_ID:-${ASCEND_DEVICE_ID}}"
    export LD_LIBRARY_PATH="${PWD}/build:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${PWD}/build:${PWD}:${PYTHONPATH:-}"
    "${PYTHON_EXEC}" run_aclgraph_sk.py
)
