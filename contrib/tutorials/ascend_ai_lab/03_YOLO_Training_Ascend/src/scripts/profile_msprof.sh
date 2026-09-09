#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PROFILE_OUTPUT="${PROFILE_OUTPUT:-profiles/yolo_single_npu}"

mkdir -p "${PROFILE_OUTPUT}"

TRAIN_CMD="python src/scripts/train_yolo_single_npu_amp.py --config src/configs/yolo_ascend.yaml --epochs 2 --profile"

msprof \
  --application="${TRAIN_CMD}" \
  --output="${PROFILE_OUTPUT}" \
  --runtime-api=on \
  --task-time=on \
  --aicpu=on \
  --sys-hardware-mem=on \
  --sys-cpu-profiling=on \
  --sys-profiling=on
