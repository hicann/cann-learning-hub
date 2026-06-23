#!/bin/bash
# 一键：生成数据 → 运行算子 → 校验结果
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/build_out"

if [[ $# -lt 3 ]]; then
    echo "Usage: bash run.sh m k n [transA transB]"
    exit 1
fi

M="$1" K="$2" N="$3"
TRANSA="${4:-0}"
TRANSB="${5:-1}"

if [[ ! -x "${BUILD_DIR}/quant_matmul_mxfp8_tutorial" ]]; then
    echo "ERROR: 请先执行 bash build.sh 编译工程"
    exit 1
fi

cd "${BUILD_DIR}"
python3 gen_data.py "${M}" "${K}" "${N}" "${TRANSA}" "${TRANSB}"
./quant_matmul_mxfp8_tutorial "${M}" "${K}" "${N}" "${TRANSA}" "${TRANSB}"
