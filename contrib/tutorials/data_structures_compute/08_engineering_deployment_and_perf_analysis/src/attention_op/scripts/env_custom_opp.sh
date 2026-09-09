#!/bin/bash
# Set up environment for custom OPP (installed at ${HOME}/vendors/customize)

export ASCEND_CUSTOM_OPP_PATH="${HOME}/vendors/customize"

# Add custom op libraries to LD_LIBRARY_PATH
if [ -d "${HOME}/vendors/customize/op_api/lib" ]; then
    export LD_LIBRARY_PATH="${HOME}/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}"
fi
if [ -d "${HOME}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" ]; then
    export LD_LIBRARY_PATH="${HOME}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:${LD_LIBRARY_PATH}"
fi
if [ -d "${HOME}/vendors/customize/op_proto/lib/linux/aarch64" ]; then
    export LD_LIBRARY_PATH="${HOME}/vendors/customize/op_proto/lib/linux/aarch64:${LD_LIBRARY_PATH}"
fi

echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
