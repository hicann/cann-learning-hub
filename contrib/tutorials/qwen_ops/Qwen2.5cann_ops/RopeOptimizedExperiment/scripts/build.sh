#!/usr/bin/env bash
# QwenRoPeCustomOpt build script
# SoC 自动检测: npu-smi info 解析 → 规范化 → ascend310p3 / ascend910b4 / ...
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOC_VERSION="${SOC_VERSION:-}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
KEEP_BUILD="${KEEP_BUILD:-0}"

# ── SoC 自动检测 + 规范化 ──────────────────────────────────────
# 从 npu-smi info 输出中匹配芯片标识符:
#   可能格式: 310P3 / Ascend310P3 / 910B4 / Ascend910B4 / 950DT_950x / kirin9030
# 规范化后:  ascend310p3 / ascend910b4 / ascend950dt_950x / kirin9030
detect_soc() {
    local raw=""

    if command -v npu-smi &>/dev/null; then
        # grep -oE 提取所有匹配芯片模式的 token
        raw=$(npu-smi info 2>/dev/null | grep -oE '([Aa]scend)?(310P[0-9]|910B[0-9A-Za-z\-]*|950[A-Za-z0-9_]+|kirin[0-9]+)' | head -1)
    fi

    if [[ -z "$raw" ]]; then
        echo ""
        return
    fi

    # 规范化: 去 Ascend 前缀 → 小写
    local normalized
    normalized=$(echo "$raw" | sed 's/^[Aa][Ss][Cc][Ee][Nn][Dd]//' | tr '[:upper:]' '[:lower:]')

    # 非 kirin 系列加 ascend 前缀
    if [[ "$normalized" =~ ^kirin ]]; then
        echo "$normalized"
    else
        echo "ascend$normalized"
    fi
}

if [[ -z "$SOC_VERSION" ]]; then
    SOC_VERSION=$(detect_soc)
    if [[ -z "$SOC_VERSION" ]]; then
        echo "[WARN] npu-smi not found or cannot parse SoC. Defaulting to ascend910b3."
        echo "        Set SOC_VERSION=xxx to override."
        SOC_VERSION="ascend910b3"
    else
        echo "[INFO] Auto-detected SoC: $SOC_VERSION"
    fi
fi

# ── CANN 路径 ─────────────────────────────────────────────────
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d "/usr/local/Ascend/ascend-toolkit/latest" ]]; then
    CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"
elif [[ -d "$HOME/Ascend/ascend-toolkit/latest" ]]; then
    CANN_PATH="$HOME/Ascend/ascend-toolkit/latest"
elif [[ -d "$HOME/Ascend/ascend-toolkit/cann-8.5.0" ]]; then
    CANN_PATH="$HOME/Ascend/ascend-toolkit/cann-8.5.0"
elif [[ -d "$HOME/ascend-toolkit/cann-8.5.0" ]]; then
    CANN_PATH="$HOME/ascend-toolkit/cann-8.5.0"
elif [[ -d "$HOME/CANN/cann-9.0.0" ]]; then
    CANN_PATH="$HOME/CANN/cann-9.0.0"
elif compgen -G "$HOME/CANN/cann-*" &>/dev/null; then
    CANN_PATH=$(ls -d "$HOME"/CANN/cann-* 2>/dev/null | sort -V | tail -1)
elif compgen -G "/usr/local/Ascend/ascend-toolkit/*" &>/dev/null; then
    CANN_PATH=$(ls -d /usr/local/Ascend/ascend-toolkit/* 2>/dev/null | grep -v latest | sort -V | tail -1)
else
    echo "[ERROR] Set ASCEND_HOME_PATH or install CANN."
    exit 1
fi
export ASCEND_HOME_PATH="$CANN_PATH"
export ASCEND_TOOLKIT_HOME="$CANN_PATH"

# ── 环境 ───────────────────────────────────────────────────────
set +e; set +u; set +o pipefail
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash
if [[ -f "$CANN_PATH/set_env.sh" ]]; then
    source "$CANN_PATH/set_env.sh"
elif [[ -f "$CANN_PATH/bin/setenv.bash" ]]; then
    source "$CANN_PATH/bin/setenv.bash"
else
    echo "[ERROR] No set_env.sh in $CANN_PATH"; exit 1
fi
set -e; set -u; set -o pipefail

# ── GCC 11 ─────────────────────────────────────────────────────
if [[ "$(uname -m)" == "aarch64" && -f /usr/include/c++/11/cstdint ]]; then
    export CC="${CC:-/usr/bin/gcc-11}"
    export CXX="${CXX:-/usr/bin/g++-11}"
    G="/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/aarch64-linux-gnu"
    export CPLUS_INCLUDE_PATH="$G${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="$G${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi

# ── 构建 ───────────────────────────────────────────────────────
cd "$ROOT_DIR"
[[ "$KEEP_BUILD" == "0" ]] && rm -rf build out
mkdir -p build out

echo "[BUILD] SOC_VERSION=$SOC_VERSION Compiling rope_baseline_kernel..."
cmake -S . -B build \
    -DSOC_VERSION="$SOC_VERSION" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/out" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"

cmake --build build -j
cmake --install build

# Post-install: some CANN versions emit a generic libascendc_kernels_npu.so;
# rename it (and its references) to an operator-specific name.
# If the build already emitted the correct name, skip retarget.
GENERIC_LIB="$ROOT_DIR/out/lib/libascendc_kernels_npu.so"
if [[ -f "$GENERIC_LIB" ]]; then
    python3 "$ROOT_DIR/../scripts/retarget_kernel_library.py" \
        --lib-dir "$ROOT_DIR/out/lib" \
        --register librope_torch_register.so \
        --kernel librope_kernels_npu.so
else
    echo "[BUILD] Kernel lib already correctly named, skipping retarget."
fi

echo "[BUILD] Done. Output: out/bin/rope_baseline_standalone"
echo "[BUILD]         out/lib/librope_kernels_npu.so"
echo "[BUILD]         out/lib/librope_torch_register.so"
