#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

ONNX_PATH="${ONNX_PATH:-models/yolov5s.onnx}"
OM_PATH="${OM_PATH:-models/yolov5s_310b4}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
INPUT_SHAPE="${INPUT_SHAPE:-images:1,3,640,640}"
PRECISION_MODE="${PRECISION_MODE:-allow_mix_precision}"

mkdir -p "$(dirname "${OM_PATH}")"

atc \
  --framework=5 \
  --model="${ONNX_PATH}" \
  --output="${OM_PATH}" \
  --input_shape="${INPUT_SHAPE}" \
  --input_format=NCHW \
  --soc_version="${SOC_VERSION}" \
  --precision_mode="${PRECISION_MODE}" \
  --log=info

echo "OM saved to ${OM_PATH}.om"
