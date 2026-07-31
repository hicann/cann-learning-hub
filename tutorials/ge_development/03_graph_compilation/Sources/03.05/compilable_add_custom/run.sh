#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${PROJECT_DIR}/build"
OUTPUT_DIR="${PROJECT_DIR}/output"
CUSTOM_OP_DIR=""
CUSTOM_OP_LIBRARY_PATH=""
CUSTOM_OP_KERNEL_PATH=""
CUSTOM_OP_PROTO_HEADER_PATH=""

info() {
  echo "[INFO] $*"
}

error() {
  echo "[ERROR] $*" >&2
}

detect_opp_os_dir() {
  local os_name
  os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "${os_name}" in
    mingw*|msys*|cygwin*)
      echo "windows"
      ;;
    *)
      echo "linux"
      ;;
  esac
}

detect_opp_arch_dir() {
  local arch_name
  arch_name="$(uname -m | tr '[:upper:]' '[:lower:]')"
  case "${arch_name}" in
    aarch64|arm64)
      echo "aarch64"
      ;;
    x86_64|amd64)
      echo "x86_64"
      ;;
    *)
      echo "${arch_name}"
      ;;
  esac
}

get_custom_op_library_name() {
  if [[ "$(detect_opp_os_dir)" == "windows" ]]; then
    echo "cust_opapi.dll"
    return
  fi
  echo "libcust_opapi.so"
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  echo 8
}

usage() {
  cat <<'EOF'
Usage:
  bash run.sh

Options:
  -h, --help    显示帮助信息
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  error "ASCEND_HOME_PATH is empty. Please source CANN set_env.sh first."
  exit 1
fi

CUSTOM_OP_DIR="${PROJECT_DIR}/output/op_graph/lib/$(detect_opp_os_dir)/$(detect_opp_arch_dir)"
CUSTOM_OP_LIBRARY_PATH="${PROJECT_DIR}/output/op_graph/lib/$(detect_opp_os_dir)/$(detect_opp_arch_dir)/$(get_custom_op_library_name)"
CUSTOM_OP_KERNEL_PATH="${PROJECT_DIR}/output/op_graph/lib/$(detect_opp_os_dir)/$(detect_opp_arch_dir)/add_custom_kernel.npubin"
CUSTOM_OP_PROTO_HEADER_PATH="${PROJECT_DIR}/output/op_graph/include/add_custom.h"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}" "${CUSTOM_OP_DIR}"

JOBS="$(detect_jobs)"

# CMake records the source tree as an absolute path. If the cached source path
# does not match the current project directory, recreate only the stale build
# directory before configuring it again.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHED_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt")"
  if [[ -n "${CACHED_SOURCE_DIR}" && "${CACHED_SOURCE_DIR}" != "${PROJECT_DIR}" ]]; then
    cmake -E remove_directory "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
  fi
fi

info "Step 1/3: configure and build sample targets"
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${JOBS}"

if [[ ! -f "${CUSTOM_OP_LIBRARY_PATH}" ]]; then
  error "Custom op library was not generated: ${CUSTOM_OP_LIBRARY_PATH}"
  exit 1
fi
if [[ ! -f "${CUSTOM_OP_PROTO_HEADER_PATH}" ]]; then
  error "Custom op proto header was not generated: ${CUSTOM_OP_PROTO_HEADER_PATH}"
  exit 1
fi

# CANN 9.0 looks for libcust_opapi.so directly below each path entry. This
# direct-directory form is also supported by CANN 9.1.
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OP_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"

SOC_VERSION_VALUE="${SOC_VERSION:-Ascend910B4}"
info "Step 2/3: compile Ascend C kernel for ${SOC_VERSION_VALUE}"
rm -f "${CUSTOM_OP_KERNEL_PATH}"
"${BUILD_DIR}/compilable_add_kernel_compile" \
  "${PROJECT_DIR}/ge/add_custom_kernel.cpp" \
  "${CUSTOM_OP_KERNEL_PATH}" \
  "${SOC_VERSION_VALUE}"

if [[ ! -s "${CUSTOM_OP_KERNEL_PATH}" ]]; then
  error "Kernel binary was not generated: ${CUSTOM_OP_KERNEL_PATH}"
  exit 1
fi

info "Step 3/3: run GE online compilation and NPU execution"
(
  cd "${BUILD_DIR}"
  ./compilable_add_session_run
)

info "Sample pipeline finished."
