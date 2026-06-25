#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${SCRIPT_DIR}/build
# TODO: Set ASCEND_HOME_PATH in your environment, or replace the fallback path below
#       with your actual CANN installation directory, e.g. /usr/local/Ascend/ascend-toolkit/latest
CANN_PATH=${ASCEND_HOME_PATH:-/path/to/your/cann}

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}

cmake .. -DASCEND_CANN_PACKAGE_PATH=${CANN_PATH}
make -j$(nproc)

echo "Build success: ${BUILD_DIR}/minimal_demo"