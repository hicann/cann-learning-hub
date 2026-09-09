#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

OUTPUT_DIR="${OUTPUT_DIR:-profiles/yolo_edge}"
IMAGE_PATH="${IMAGE_PATH:-src/data/images/bus.jpg}"
mkdir -p "${OUTPUT_DIR}"

APP_CMD="python src/scripts/pyacl_yolo_infer.py --config src/configs/yolo_edge.yaml --image ${IMAGE_PATH} --repeat 50"

msprof \
  --application="${APP_CMD}" \
  --output="${OUTPUT_DIR}" \
  --runtime-api=on \
  --task-time=on \
  --aicpu=on \
  --sys-cpu-profiling=on \
  --sys-hardware-mem=on

echo "Profiling data saved to ${OUTPUT_DIR}. Open it in MindStudio Profiler for timeline analysis."
