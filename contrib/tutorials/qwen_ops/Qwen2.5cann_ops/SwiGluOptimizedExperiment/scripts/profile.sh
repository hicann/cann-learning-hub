#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROWS="${ROWS:-128}"
HIDDEN="${HIDDEN:-1024}"
BLOCK_DIM="${BLOCK_DIM:-8}"
TILE_LENGTH="${TILE_LENGTH:-1024}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
ROUNDS="${ROUNDS:-5}"
BINARY="$ROOT_DIR/out/bin/swiglu_optimized_standalone"

[[ -x "$BINARY" ]] || { echo "[ERROR] Run scripts/build.sh first."; exit 1; }

export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:${LD_LIBRARY_PATH:-}"
"$BINARY" --rows "$ROWS" --hidden "$HIDDEN" --block-dim "$BLOCK_DIM" --tile-length "$TILE_LENGTH" --warmup "$WARMUP" --repeat "$REPEAT" --rounds "$ROUNDS"

if command -v msprof >/dev/null 2>&1; then
    mkdir -p "$ROOT_DIR/output/profile"
    msprof --application="$BINARY --rows $ROWS --hidden $HIDDEN --block-dim $BLOCK_DIM --tile-length $TILE_LENGTH --warmup 5 --repeat 10 --rounds 1" \
        --output="$ROOT_DIR/output/profile"
    echo "[OK] msprof output: output/profile"
fi