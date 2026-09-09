#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"

python src/scripts/train_yolo_single_npu_amp.py \
  --config src/configs/yolo_ascend.yaml
