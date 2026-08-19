#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOC_VERSION="${SOC_VERSION:-}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
KEEP_BUILD="${KEEP_BUILD:-0}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

detect_soc() {
    local raw=""
    if command -v npu-smi >/dev/null 2>&1; then
        raw=$(npu-smi info 2>/dev/null | grep -oE '([Aa]scend)?(310P[0-9]|910B[0-9A-Za-z-]*|950[A-Za-z0-9_]+|kirin[0-9]+)' | head -1 || true)
    fi
    [[ -z "$raw" ]] && return
    local normalized
    normalized=$(echo "$raw" | sed 's/^[Aa][Ss][Cc][Ee][Nn][Dd]//' | tr '[:upper:]' '[:lower:]')
    if [[ "$normalized" =~ ^kirin ]]; then echo "$normalized"; else echo "ascend$normalized"; fi
}

if [[ -z "$SOC_VERSION" ]]; then
    SOC_VERSION=$(detect_soc || true)
    [[ -z "$SOC_VERSION" ]] && SOC_VERSION=ascend910b3
fi

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
else
    echo "[ERROR] Set ASCEND_HOME_PATH to CANN toolkit root."
    exit 1
fi
export ASCEND_HOME_PATH="$CANN_PATH"
export ASCEND_TOOLKIT_HOME="$CANN_PATH"

set +u
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash
[[ -f "$CANN_PATH/set_env.sh" ]] && source "$CANN_PATH/set_env.sh"
set -u

cd "$ROOT_DIR"
[[ "$KEEP_BUILD" == "0" ]] && rm -rf build out
mkdir -p build out

cmake -S . -B build \
    -DSOC_VERSION="$SOC_VERSION" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/out" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build -j"$BUILD_JOBS"
cmake --install build
python3 "$ROOT_DIR/../scripts/retarget_kernel_library.py" \
    --lib-dir "$ROOT_DIR/out/lib" \
    --register libswiglu_torch_register.so \
    --kernel libswiglu_kernels_npu.so

echo "[OK] build finished"
echo "[OK] out/bin/swiglu_optimized_standalone"
echo "[OK] out/lib/libswiglu_torch_register.so"
