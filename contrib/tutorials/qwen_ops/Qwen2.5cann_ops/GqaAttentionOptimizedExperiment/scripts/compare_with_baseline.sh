#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASELINE_DIR="${BASELINE_DIR:-$(cd "$ROOT_DIR/../GqaAttentionBaselineExperiment" && pwd)}"
for script in "$BASELINE_DIR/scripts/profile.sh" "$ROOT_DIR/scripts/profile.sh"; do [[ -x "$script" ]] || { echo "[ERROR] missing executable: $script"; exit 1; }; done
export BATCH="${BATCH:-1}" Q_HEADS="${Q_HEADS:-8}" KV_HEADS="${KV_HEADS:-2}" Q_LEN="${Q_LEN:-32}" KV_LEN="${KV_LEN:-32}" HEAD_DIM="${HEAD_DIM:-64}" BLOCK_DIM="${BLOCK_DIM:-8}" WARMUP="${WARMUP:-10}" REPEAT="${REPEAT:-50}" ROUNDS="${ROUNDS:-5}" CAUSAL="${CAUSAL:-1}"
echo "[COMPARE] baseline"; bash "$BASELINE_DIR/scripts/profile.sh"
echo "[COMPARE] optimized"; bash "$ROOT_DIR/scripts/profile.sh"
