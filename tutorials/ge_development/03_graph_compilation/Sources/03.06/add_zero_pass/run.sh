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
BUILD_DIR="${SCRIPT_DIR}/build"
PASS_LIBRARY="${BUILD_DIR}/es_output/lib64/libnotebook_add_zero_pass.so"
MARKER_FILE="${SCRIPT_DIR}/notebook_add_zero_pass.executed"

info() {
  echo "[INFO] $*"
}

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "[ERROR] ASCEND_HOME_PATH is empty. Please source the CANN set_env.sh first." >&2
  exit 1
fi

if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
else
  JOBS=8
fi

info "Step 1/3: generate ES API and build the C++ Fusion Pass"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${JOBS}"

if [[ ! -s "${PASS_LIBRARY}" ]]; then
  echo "[ERROR] Fusion Pass library was not generated: ${PASS_LIBRARY}" >&2
  exit 1
fi

info "Step 2/3: load the Pass, compile the graph, and run it on NPU"
rm -f "${MARKER_FILE}"
GE_NOTEBOOK_PASS_SO="${PASS_LIBRARY}" \
GE_PASS_MARKER="${MARKER_FILE}" \
"${PYTHON_EXECUTABLE:-python3}" "${SCRIPT_DIR}/run_add_zero_pass.py"

info "Step 3/3: verify that Replacement really executed"
if [[ ! -f "${MARKER_FILE}" ]] || ! grep -qx "NotebookAddZeroPass executed" "${MARKER_FILE}"; then
  echo "[ERROR] GE did not execute NotebookAddZeroPass::Replacement" >&2
  exit 1
fi

echo "[OK] GE 已执行 C++ PatternFusionPass，Replacement 标记校验通过"
