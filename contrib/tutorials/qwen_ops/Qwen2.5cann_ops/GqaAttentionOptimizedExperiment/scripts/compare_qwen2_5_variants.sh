#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPEAT="${REPEAT:-3}"
echo "[COMPARE] Qwen2.5 baseline custom operators"
bash "$ROOT_DIR/GqaAttentionBaselineExperiment/scripts/compare_qwen2_5_forward.sh" --repeat "$REPEAT" "$@"
echo "[COMPARE] Qwen2.5 optimized custom operators"
bash "$ROOT_DIR/GqaAttentionOptimizedExperiment/scripts/compare_qwen2_5_forward.sh" --repeat "$REPEAT" "$@"
