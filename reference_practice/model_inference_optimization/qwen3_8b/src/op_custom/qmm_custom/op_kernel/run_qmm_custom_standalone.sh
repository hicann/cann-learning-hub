#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ASCEND_HOME=${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}
ASC_CMAKE_DIR=${ASC_CMAKE_DIR:-/mnt/workspace/gitCode/cann/asc-devkit/cmake/asc}
BUILD_DIR="${SCRIPT_DIR}/standalone_build"
SRC_DIR="${BUILD_DIR}/src"

source "${ASCEND_HOME}/set_env.sh"

mkdir -p "${BUILD_DIR}"
rm -rf "${SRC_DIR}"
mkdir -p "${SRC_DIR}"
ln -s "${SCRIPT_DIR}/qmm_custom_standalone.asc" "${SRC_DIR}/qmm_custom_standalone.asc"
ln -s "${SCRIPT_DIR}/qmm_custom_tiling.h" "${SRC_DIR}/qmm_custom_tiling.h"
cp "${SCRIPT_DIR}/CMakeLists.standalone.txt" "${SRC_DIR}/CMakeLists.txt"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -C /dev/null \
  -DCMAKE_PREFIX_PATH="${ASC_CMAKE_DIR}" \
  -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
  -DCMAKE_BUILD_TYPE=Debug \
  -G "Unix Makefiles"

cmake --build "${BUILD_DIR}" --target qmm_custom_standalone -j"$(nproc)"

"${BUILD_DIR}/qmm_custom_standalone" "$@"
