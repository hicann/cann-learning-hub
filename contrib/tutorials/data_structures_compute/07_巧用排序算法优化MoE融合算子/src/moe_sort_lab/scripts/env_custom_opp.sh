#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ASCEND_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_HOME_PATH="$ASCEND_PATH"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-${ASCEND_HOME_PATH}/opp/vendors/customize}"
if [ -d "${ASCEND_CUSTOM_OPP_PATH}/op_api/lib" ]; then
  export LD_LIBRARY_PATH="${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
