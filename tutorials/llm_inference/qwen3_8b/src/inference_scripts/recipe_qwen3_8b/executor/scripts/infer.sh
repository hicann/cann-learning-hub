# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/set_env.sh"
FUNCTION_ABS_PATH="${SCRIPT_PATH}/function.sh"
VALIDATE_ABS_PATH="${SCRIPT_PATH}/validate_infer_args.py"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
FUNCTION_ABS_PATH=$(realpath "${FUNCTION_ABS_PATH}")
VALIDATE_ABS_PATH=$(realpath "${VALIDATE_ABS_PATH}")

source ${SET_ENV_ABS_PATH}
source ${FUNCTION_ABS_PATH}

# Set defaults here or override via command-line args.
MODEL="qwen"
MODE="offline"
YAML_FILE="qwen3_8b_1tp.yaml"

# Parse --key value arguments (overrides defaults above).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";     shift 2 ;;
        --mode)    MODE="$2";      shift 2 ;;
        --yaml)         YAML_FILE="$2";      shift 2 ;;
        -h|--help)
            echo "用法: $0 --model qwen --mode offline --yaml <yaml文件名或绝对路径>"
            echo ""
            echo "  离线推理: $0 --model qwen --yaml qwen3_8b_1tp.yaml"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ "$MODE" != "offline" ]; then
    echo "本教程脚本仅包含 Qwen3-8B 离线推理。"
    exit 1
fi

export MODEL_DIR="$MODEL"
export YAML_PARENT_PATH="${SCRIPT_PATH}/../../models/${MODEL}/config"

if [[ "${YAML_FILE}" = /* ]]; then
    export YAML="${YAML_FILE}"
else
    export YAML="${YAML_PARENT_PATH}/${YAML_FILE}"
fi

python3 "${VALIDATE_ABS_PATH}" \
    --models-root "${SCRIPT_PATH}/../../models" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --yaml "${YAML}" || exit 1

echo "====================> 启动离线推理 (model=${MODEL}, yaml=${YAML_FILE})"

launch "$MODE"
