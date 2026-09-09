#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ASCEND_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_HOME_PATH="$ASCEND_PATH"
LOCAL_OPP_ROOT="$PROJECT_DIR/custom_ops/generated/local_opp"
LOCAL_VENDOR_DIR="$LOCAL_OPP_ROOT/vendors/customize"
SET_ENV_SCRIPT="$LOCAL_VENDOR_DIR/bin/set_env.bash"

if [ ! -f "$SET_ENV_SCRIPT" ]; then
  echo "Local custom OPP is missing. Run bash scripts/build_ops.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

# set_env.bash 由安装包生成，内容引用 ${ASCEND_CUSTOM_OPP_PATH}/${LD_LIBRARY_PATH}，
# 在调用方启用 set -u（如 build_runner.sh）前需先预置为空，避免 unbound variable。
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$SET_ENV_SCRIPT"
export ASCEND_CUSTOM_OPP_PATH="$LOCAL_VENDOR_DIR"
if [ -d "$LOCAL_VENDOR_DIR/op_api/lib" ]; then
  export LD_LIBRARY_PATH="$LOCAL_VENDOR_DIR/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
