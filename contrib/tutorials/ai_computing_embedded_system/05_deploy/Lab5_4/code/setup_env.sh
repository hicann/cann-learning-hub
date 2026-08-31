#!/bin/bash
# 香橙派开发板环境配置脚本
#
# 在执行ATC转换和OM推理前, 先运行此脚本加载CANN环境变量。
# 用法: source setup_env.sh

CANN_TOOLKIT_PATH="/usr/local/Ascend/ascend-toolkit"

if [ -f "${CANN_TOOLKIT_PATH}/set_env.sh" ]; then
    source "${CANN_TOOLKIT_PATH}/set_env.sh"
    echo "[OK] CANN环境已加载: ${CANN_TOOLKIT_PATH}"
else
    echo "[警告] 未找到CANN环境脚本: ${CANN_TOOLKIT_PATH}/set_env.sh"
    echo "       请确认CANN Toolkit已正确安装"
fi

export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=1
echo "[OK] 已设置单进程编译 (防止开发板内存耗尽)"

echo ""
echo "环境信息:"
echo "  Python: $(python3 --version 2>&1)"
echo "  ATC:    $(which atc 2>/dev/null || echo '未找到')"
echo "  NPU信息:"
npu-smi info 2>/dev/null || echo "  (无法获取NPU信息)"
