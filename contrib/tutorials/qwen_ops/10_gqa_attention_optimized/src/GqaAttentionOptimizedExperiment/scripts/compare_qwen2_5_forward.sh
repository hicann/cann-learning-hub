#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COURSE_ROOT=$(cd "$ROOT_DIR/../../.." && pwd)
RMS_DIR="$COURSE_ROOT/06_rmsnorm_optimized/src/RmsNormOptimizedExperiment"
GQA_DIR="$ROOT_DIR"
export LD_LIBRARY_PATH="$RMS_DIR/out/lib:$GQA_DIR/out/lib:${LD_LIBRARY_PATH:-}"
REPEAT=1
if [[ "${1:-}" == "--repeat" ]]; then
  REPEAT="$2"
  shift 2
fi
for ((run = 1; run <= REPEAT; run++)); do
  echo "[RUN] optimized GQA correctness $run/$REPEAT"
  python3 "$ROOT_DIR/tests/test_qwen_forward.py" "$@"
done
