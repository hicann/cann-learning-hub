#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

function main()
{
  echo "[INFO] The sample starts to run"
  cd "${ScriptPath}/../out" || { echo "[ERROR] out directory not found, please build first"; exit 1; }
  local args=()
  for arg in "$@"; do
    case "$arg" in
      --model=*) args+=("--model=$(realpath -m "${ScriptPath}/../${arg#--model=}")") ;;
      --input=*) args+=("--input=$(realpath -m "${ScriptPath}/../${arg#--input=}")") ;;
      *) args+=("$arg") ;;
    esac
  done
  ./yolov13_main "${args[@]}"
}

main
