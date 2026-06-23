#!/bin/bash
# 一键编译 MXFP8 SWAT 教程工程（仅支持 A5 / dav-3510）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 加载 CANN 环境
if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
elif [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

echo "NPU_ARCH=dav-3510"
echo "ops-tensor will be fetched automatically by cmake (FetchContent) if not cached locally."

cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build --parallel
cmake --install build --prefix ./build_out

echo ""
echo "Build OK. Executable: ${SCRIPT_DIR}/build_out/quant_matmul_mxfp8_tutorial"
echo "Run from build_out/: python3 gen_data.py <m> <k> <n> [transA transB]"
