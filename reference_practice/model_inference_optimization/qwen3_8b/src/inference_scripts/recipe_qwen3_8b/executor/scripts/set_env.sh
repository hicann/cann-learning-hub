# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash

rm -rf /root/atc_data/

CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
RECIPES_PATH=$(dirname "$(dirname "$CURRENT_PATH")")
export PYTHONPATH="${RECIPES_PATH}:${PYTHONPATH}"

if [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
    export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"
elif [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/bin/setenv.bash" ]; then
    source "${ASCEND_TOOLKIT_HOME}/bin/setenv.bash"
    export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"
elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit"
elif [ -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]; then
    source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
else
    echo "Cannot locate Ascend toolkit set_env.sh. Please source CANN environment first or set ASCEND_TOOLKIT_HOME."
    exit 1
fi

export TASK_QUEUE_ENABLE=2
