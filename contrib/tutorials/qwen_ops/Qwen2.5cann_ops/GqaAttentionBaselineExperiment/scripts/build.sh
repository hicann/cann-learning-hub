#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOC_VERSION="${SOC_VERSION:-ascend910b3}"
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then CANN_PATH="$ASCEND_HOME_PATH"; elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest; elif [[ -d /home/user/Ascend/ascend-toolkit/cann-8.5.0 ]]; then CANN_PATH=/home/user/Ascend/ascend-toolkit/cann-8.5.0; else echo "[ERROR] Set ASCEND_HOME_PATH to CANN toolkit root."; exit 1; fi
export ASCEND_HOME_PATH="$CANN_PATH" ASCEND_TOOLKIT_HOME="$CANN_PATH"
set +u; [[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash; [[ -f "$CANN_PATH/set_env.sh" ]] && source "$CANN_PATH/set_env.sh"; set -u
rm -rf "$ROOT_DIR/build" "$ROOT_DIR/out"; cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -DSOC_VERSION="$SOC_VERSION" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/out" -DASCEND_CANN_PACKAGE_PATH="$CANN_PATH"; cmake --build "$ROOT_DIR/build" -j; cmake --install "$ROOT_DIR/build"
echo "[OK] out/bin/gqa_attention_baseline_standalone"
