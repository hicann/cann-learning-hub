#!/usr/bin/env bash
### 该脚本用于05_quantized_a8w8_matmul_operator_development.ipynb课程
### 一次性生成算子交付件框架并且编写完算子之后，使用该脚本一站式配置环境并测试注册到python之后的算子计算效果
### 效果见 当前目录下的res.txt 文件
###
###

set -euo pipefail



SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CANN_HOME=${CANN_HOME:-/home/developer/Ascend/cann-9.0.0}
CUSTOM_OPP_ROOT=${CUSTOM_OPP_ROOT:-/home/developer/Ascend/custom_opp}
TEST_MODE=${1:-min}

prepend_path_var() {
    local var_name=$1
    local new_path=$2
    local old_value="${!var_name:-}"
    local result="${new_path}"
    local item

    IFS=':' read -ra path_items <<< "${old_value}"
    for item in "${path_items[@]}"; do
        if [ -n "${item}" ] && [ "${item}" != "${new_path}" ]; then
            result="${result}:${item}"
        fi
    done
    export "${var_name}=${result}"
}

echo "[1/6] Load CANN env: ${CANN_HOME}"
source "${CANN_HOME}/set_env.sh"

echo "[2/6] Build QmmCustom OPP package"
cd "${SCRIPT_DIR}/qmm_custom"
bash build.sh




echo "[3/6] Install QmmCustom OPP package to: ${CUSTOM_OPP_ROOT}"
mkdir -p "${CUSTOM_OPP_ROOT}"

RUN_PKG=$(find build_out -maxdepth 1 -name '*.run' | head -n1)
if [ -z "${RUN_PKG}" ]; then
    echo "[ERROR] no .run package found under build_out"
    exit 1
fi

chmod +x "${RUN_PKG}"
if "${RUN_PKG}" --quiet -- --install-path="${CUSTOM_OPP_ROOT}"; then
    echo "[INFO] install-path mode succeeded"
else
    echo "[WARN] install-path mode failed, fallback to default install"
    "${RUN_PKG}" --quiet
fi

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

CUSTOM_VENDOR_PATH=""
for candidate in \
    "${CUSTOM_OPP_ROOT}/vendors/customize" \
    "${CANN_HOME}/opp/vendors/customize"
do
    if [ -f "${candidate}/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json" ]; then
        CUSTOM_VENDOR_PATH="${candidate}"
        break
    fi
done

if [ -n "${CUSTOM_VENDOR_PATH}" ]; then
    prepend_path_var ASCEND_CUSTOM_OPP_PATH "${CUSTOM_VENDOR_PATH}"
    if [ -d "${CUSTOM_VENDOR_PATH}/op_api/lib" ]; then
        prepend_path_var LD_LIBRARY_PATH "${CUSTOM_VENDOR_PATH}/op_api/lib"
    fi
else
    echo "[ERROR] installed QmmCustom OPP vendor path not found"
    find "${CUSTOM_OPP_ROOT}" "${CANN_HOME}/opp/vendors/customize" -maxdepth 4 -type f -name 'aic-ascend910b-ops-info.json' 2>/dev/null || true
    exit 1
fi

echo "[INFO] ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"


echo "[4/6] Build and install PyTorch extension"
python -m pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
cd "${SCRIPT_DIR}/torch_ops_extension"
rm -rf build dist custom_ops.egg-info
USE_NINJA=1 MAX_JOBS="${MAX_JOBS:-32}" python setup.py build_ext
USE_NINJA=1 MAX_JOBS="${MAX_JOBS:-32}" python setup.py bdist_wheel
python -m pip install dist/*.whl -I

echo "[5/6] Verify Python extension registration"
cd "${SCRIPT_DIR}"
python - <<'PY'
import os
import torch
import torch_npu
import custom_ops

print("ASCEND_CUSTOM_OPP_PATH =", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
print("torch.ops.custom.qmm_custom =", hasattr(torch.ops.custom, "qmm_custom"))
PY

echo "[6/6] Run QmmCustom function tests"
if [ "${TEST_MODE}" = "full" ]; then
    python test_cases.py
else
    python test_cases.py
    # python test_cases_min.py
fi
