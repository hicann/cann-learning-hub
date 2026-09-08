#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COURSE_ROOT=$(cd "$ROOT_DIR/../../.." && pwd)
REPEAT="${REPEAT:-3}"
echo "[COMPARE] Qwen2.5 baseline custom operators"
bash "$COURSE_ROOT/05_gqa_attention_baseline/src/GqaAttentionBaselineExperiment/scripts/compare_qwen2_5_forward.sh" --repeat "$REPEAT" "$@"
echo "[COMPARE] Qwen2.5 optimized custom operators"
bash "$ROOT_DIR/scripts/compare_qwen2_5_forward.sh" --repeat "$REPEAT" "$@"
