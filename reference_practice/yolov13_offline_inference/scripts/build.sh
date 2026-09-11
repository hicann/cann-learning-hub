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

ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

function build()
{
  UserKernel=`arch`
  if [[ ${TargetKernel} = "x86" ]] || [[ ${TargetKernel} = "X86" ]];then
    TargetCompiler="g++"
    TargetKernel="x86"
  else
    if [[ ${UserKernel} == "x86_64" ]];then
      TargetCompiler="aarch64-linux-gnu-g++"
      TargetKernel="arm"
    else
      TargetCompiler="g++"
      TargetKernel="arm"
    fi
  fi
  if [ -d ${ScriptPath}/../build/intermediates/host ];then
    rm -rf ${ScriptPath}/../build/intermediates/host
  fi

  mkdir -p ${ScriptPath}/../build/intermediates/host
  cd ${ScriptPath}/../build/intermediates/host

  cmake ../../../src -DCMAKE_CXX_COMPILER=${TargetCompiler} -DCMAKE_SKIP_RPATH=TRUE
  if [ $? -ne 0 ];then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi
  make
  if [ $? -ne 0 ];then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi
  cd - > /dev/null
}

function target_kernel()
{
    local arch=""
    if command -v lscpu >/dev/null 2>&1; then
        arch_info=$(lscpu 2>/dev/null | grep -i "Architecture" | head -n1)
        if [[ -n "$arch_info" ]]; then
            arch=$(echo "$arch_info" | awk -F: '{print $2}' | tr -d ' \t\n\r' | tr '[:upper:]' '[:lower:]')
            if [[ "$arch" == "aarch64" || "$arch" == "armv8" || "$arch" == "armv7l" || "$arch" == "arm" ]]; then
                TargetKernel="arm"
                return 0
            elif [[ "$arch" == "x86_64" || "$arch" == "i686" || "$arch" == "i386" ]]; then
                TargetKernel="x86"
                return 0
            fi
        fi
    fi

    arch=$(uname -m 2>/dev/null | tr '[:upper:]' '[:lower:]')
    if [[ "$arch" == "aarch64" || "$arch" == "armv8" || "$arch" == "armv7l" || "$arch" == "arm" ]]; then
        TargetKernel="arm"
        return 0
    elif [[ "$arch" == "x86_64" || "$arch" == "i686" || "$arch" == "i386" ]]; then
        TargetKernel="x86"
        return 0
    fi

    declare -i CHOICE_TIMES=0
    TargetKernel=""
    while [[ "${TargetKernel}" == "" ]]; do
        ((CHOICE_TIMES++))
        if [[ ${CHOICE_TIMES} -gt 3 ]]; then
            echo "[ERROR] TargetKernel entered incorrectly three times."
            return 1
        fi
        read -p "Please input TargetKernel? [arm/x86]: " input_kernel
        input_kernel=$(echo "$input_kernel" | tr '[:upper:]' '[:lower:]')
        if [[ "$input_kernel" == "arm" || "$input_kernel" == "x86" ]]; then
            TargetKernel="$input_kernel"
        else
            echo "[WARNING] The ${CHOICE_TIMES}th parameter input error!"
        fi
    done
    return 0
}

function main()
{
  echo "[INFO] Sample preparation"
  target_kernel || return 1
  build || return 1
  echo "[INFO] Sample preparation is complete"
}
main
