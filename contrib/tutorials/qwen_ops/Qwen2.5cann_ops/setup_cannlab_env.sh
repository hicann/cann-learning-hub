#!/usr/bin/env bash
# Source this file: source ./setup_cannlab_env.sh

QWEN_OPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QWEN_OPS_ROOT
export QWEN_OPS_CODE_ROOT="$QWEN_OPS_ROOT/Qwen2.5cann_ops"
export QWEN_OPS_MODEL_PATH="${QWEN_OPS_MODEL_PATH:-$QWEN_OPS_ROOT/Models/Qwen2.5-0.5B}"
export SOC_VERSION="${SOC_VERSION:-ascend910b3}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-8.5.2}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$ASCEND_HOME_PATH}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-$ASCEND_HOME_PATH}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-$ASCEND_HOME_PATH/toolkit}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

set +u
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash
if [[ -f "$ASCEND_HOME_PATH/set_env.sh" ]]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
elif [[ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]]; then
    source "$ASCEND_HOME_PATH/bin/setenv.bash"
fi
set -u

QWEN_OPS_LIBRARY_PATHS=()
for experiment_dir in "$QWEN_OPS_CODE_ROOT"/*Experiment; do
    [[ -d "$experiment_dir" ]] || continue
    QWEN_OPS_LIBRARY_PATHS+=("$experiment_dir/out/lib")
done
QWEN_OPS_LIBRARY_PATH=$(IFS=:; echo "${QWEN_OPS_LIBRARY_PATHS[*]}")
export LD_LIBRARY_PATH="$QWEN_OPS_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[qwen_ops] root=$QWEN_OPS_ROOT"
echo "[qwen_ops] CANN=$ASCEND_HOME_PATH, SoC=$SOC_VERSION, model=$QWEN_OPS_MODEL_PATH"
