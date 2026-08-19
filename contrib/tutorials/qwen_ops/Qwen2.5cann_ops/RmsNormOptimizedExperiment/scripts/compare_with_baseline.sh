#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASELINE_DIR="${BASELINE_DIR:-$(cd "$ROOT_DIR/../RmsNormBaselineExperiment" && pwd)}"

ROWS="${ROWS:-128}"
HIDDEN="${HIDDEN:-1024}"
BLOCK_DIM="${BLOCK_DIM:-8}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
EPS="${EPS:-1e-6}"

[[ -x "$BASELINE_DIR/scripts/profile.sh" ]] || { echo "[ERROR] baseline profile.sh not found: $BASELINE_DIR"; exit 1; }
[[ -x "$ROOT_DIR/scripts/profile.sh" ]] || { echo "[ERROR] optimized profile.sh not found."; exit 1; }

echo "[COMPARE] rows=$ROWS hidden=$HIDDEN blockDim=$BLOCK_DIM warmup=$WARMUP repeat=$REPEAT eps=$EPS"

echo "[COMPARE] Running baseline..."
(
    cd "$BASELINE_DIR"
    ROWS="$ROWS" HIDDEN="$HIDDEN" BLOCK_DIM="$BLOCK_DIM" WARMUP="$WARMUP" REPEAT="$REPEAT" EPS="$EPS" \
        bash scripts/profile.sh
)

echo "[COMPARE] Running optimized..."
(
    cd "$ROOT_DIR"
    ROWS="$ROWS" HIDDEN="$HIDDEN" BLOCK_DIM="$BLOCK_DIM" WARMUP="$WARMUP" REPEAT="$REPEAT" EPS="$EPS" \
        bash scripts/profile.sh
)
