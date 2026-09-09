#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NPU_PER_NODE="${NPU_PER_NODE:-8}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPU_PER_NODE}" \
  src/scripts/train_yolo_ddp_amp.py \
  --config src/configs/yolo_ascend.yaml
