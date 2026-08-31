#!/bin/bash
# Set up environment for custom OPP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_OPP="$SCRIPT_DIR/../custom_ops/generated/StackExprOps/build_out/local_opp/vendors/customize"

if [ -d "$LOCAL_OPP/op_api" ]; then
    CUSTOM_OPP_PATH="$LOCAL_OPP"
elif [ -d "${ASCEND_HOME_PATH}/opp/vendors/vendors/customize/op_api" ]; then
    CUSTOM_OPP_PATH="${ASCEND_HOME_PATH}/opp/vendors/vendors/customize"
else
    CUSTOM_OPP_PATH="${ASCEND_HOME_PATH}/opp/vendors/customize"
fi

export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_PATH"

# Add custom op libraries to LD_LIBRARY_PATH
if [ -d "$CUSTOM_OPP_PATH/op_api/lib" ]; then
    export LD_LIBRARY_PATH="$CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH}"
fi
if [ -d "$CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" ]; then
    export LD_LIBRARY_PATH="$CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:${LD_LIBRARY_PATH}"
fi
if [ -d "$CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64" ]; then
    export LD_LIBRARY_PATH="$CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64:${LD_LIBRARY_PATH}"
fi

echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
