#!/bin/bash
set -e
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${BASE_DIR}" || exit 1

# remove historical compilation results
rm -rf build dist

# use ninja to build system and parallel compilation
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required to build torch ops extensions. Install it first, for example: pip3 install ninja" >&2
    exit 1
fi
export USE_NINJA=1

# set parallel jobs for compilation
if [ -z "$MAX_JOBS" ]; then
    export MAX_JOBS=`nproc`
    echo "Using $MAX_JOBS parallel jobs for compilation"
fi

# compile wheel package using incremental compilation
python3 setup.py build_ext
python3 setup.py bdist_wheel

# install wheel package
cd "${BASE_DIR}/dist" || exit 1
pip3 install *.whl -I
cd - >/dev/null
