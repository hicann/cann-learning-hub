#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

: "${MASTER_ADDR:?set MASTER_ADDR to the IP of rank 0 node}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export NPU_PER_NODE="${NPU_PER_NODE:-8}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPU_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/scripts/train_yolo_ddp_amp.py \
  --config src/configs/yolo_ascend.yaml
