#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RMS_DIR="$ROOT_DIR/RmsNormBaselineExperiment"
GQA_DIR="$ROOT_DIR/GqaAttentionBaselineExperiment"
export LD_LIBRARY_PATH="$RMS_DIR/out/lib:$GQA_DIR/out/lib:${LD_LIBRARY_PATH:-}"
python3 "$ROOT_DIR/qwen2_5_forward_benchmark.py" --variant baseline "$@"
