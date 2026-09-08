#!/usr/bin/env bash
# May be sourced from any directory; paths are resolved from this file.

QWEN_OPS_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_OPS_ROOT="$QWEN_OPS_CODE_ROOT"
export QWEN_OPS_ROOT QWEN_OPS_CODE_ROOT
export QWEN_OPS_MODEL_PATH="${QWEN_OPS_MODEL_PATH:-$QWEN_OPS_ROOT/Models/Qwen2.5-0.5B}"
if [[ -z "${SOC_VERSION:-}" ]]; then
    QWEN_OPS_SOC_NAME=""
    if command -v npu-smi >/dev/null 2>&1; then
        QWEN_OPS_SOC_NAME=$(npu-smi info 2>/dev/null \
            | grep -oE '([Aa]scend)?(310P[0-9]|910B[0-9A-Za-z-]*|950[A-Za-z0-9_]+|kirin[0-9]+)' \
            | head -1 || true)
    fi
    if [[ -n "$QWEN_OPS_SOC_NAME" ]]; then
        QWEN_OPS_SOC_NAME=$(echo "$QWEN_OPS_SOC_NAME" \
            | sed -E 's/^[Aa][Ss][Cc][Ee][Nn][Dd]//; s/-[0-9]+$//' \
            | tr '[:upper:]' '[:lower:]')
        if [[ "$QWEN_OPS_SOC_NAME" =~ ^kirin ]]; then
            export SOC_VERSION="$QWEN_OPS_SOC_NAME"
        else
            export SOC_VERSION="ascend$QWEN_OPS_SOC_NAME"
        fi
    else
        export SOC_VERSION=ascend910b4
    fi
fi

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    QWEN_OPS_CANN_HOME="$ASCEND_HOME_PATH"
elif [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    QWEN_OPS_CANN_HOME="$ASCEND_TOOLKIT_HOME"
elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    QWEN_OPS_CANN_HOME=/usr/local/Ascend/ascend-toolkit/latest
else
    echo "[qwen_ops] CANN not found; set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$QWEN_OPS_CANN_HOME/set_env.sh" && ! -f "$QWEN_OPS_CANN_HOME/bin/setenv.bash" ]]; then
    echo "[qwen_ops] invalid CANN root: $QWEN_OPS_CANN_HOME" >&2
    return 1 2>/dev/null || exit 1
fi

export ASCEND_HOME_PATH="$QWEN_OPS_CANN_HOME"
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-$ASCEND_HOME_PATH}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-$ASCEND_HOME_PATH}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

case $- in
    *u*) QWEN_OPS_HAD_NOUNSET=1 ;;
    *) QWEN_OPS_HAD_NOUNSET=0 ;;
esac
set +u
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash
if [[ -f "$ASCEND_HOME_PATH/set_env.sh" ]]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
elif [[ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]]; then
    source "$ASCEND_HOME_PATH/bin/setenv.bash"
fi
if [[ "$QWEN_OPS_HAD_NOUNSET" == 1 ]]; then
    set -u
fi

QWEN_OPS_LIBRARY_PATHS=(
    "$QWEN_OPS_CODE_ROOT/01_rmsnorm_baseline/src/RmsNormBaselineExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/02_rope_baseline/src/RopeBaselineExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/03_swiglu_baseline/src/SwiGluBaselineExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/04_gemm_baseline/src/GemmBaselineExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/05_gqa_attention_baseline/src/GqaAttentionBaselineExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/06_rmsnorm_optimized/src/RmsNormOptimizedExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/07_rope_optimized/src/RopeOptimizedExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/08_swiglu_optimized/src/SwiGluOptimizedExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/09_gemm_optimized/src/GemmOptimizedExperiment/out/lib"
    "$QWEN_OPS_CODE_ROOT/10_gqa_attention_optimized/src/GqaAttentionOptimizedExperiment/out/lib"
)
QWEN_OPS_LIBRARY_PATH=$(IFS=:; echo "${QWEN_OPS_LIBRARY_PATHS[*]}")
export LD_LIBRARY_PATH="$QWEN_OPS_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[qwen_ops] root=$QWEN_OPS_ROOT"
echo "[qwen_ops] CANN=$ASCEND_HOME_PATH, SoC=$SOC_VERSION, model=$QWEN_OPS_MODEL_PATH"
unset QWEN_OPS_CANN_HOME QWEN_OPS_HAD_NOUNSET QWEN_OPS_LIBRARY_PATH QWEN_OPS_LIBRARY_PATHS QWEN_OPS_SOC_NAME
