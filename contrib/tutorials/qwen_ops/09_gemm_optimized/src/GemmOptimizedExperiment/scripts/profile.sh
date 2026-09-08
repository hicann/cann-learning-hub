#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
M="${M:-128}"
K="${K:-1024}"
N="${N:-512}"
BLOCK_DIM="${BLOCK_DIM:-16}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"
ROUNDS="${ROUNDS:-5}"
BINARY="$ROOT_DIR/out/bin/gemm_optimized_standalone"

[[ -x "$BINARY" ]] || { echo "[ERROR] Run scripts/build.sh first."; exit 1; }

export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:${LD_LIBRARY_PATH:-}"
"$BINARY" --m "$M" --k "$K" --n "$N" --block-dim "$BLOCK_DIM" --warmup "$WARMUP" --repeat "$REPEAT" --rounds "$ROUNDS"

if command -v msprof >/dev/null 2>&1; then
    mkdir -p "$ROOT_DIR/output/profile"
    msprof --application="$BINARY --m $M --k $K --n $N --block-dim $BLOCK_DIM --warmup 2 --repeat 5 --rounds 1" \
        --output="$ROOT_DIR/output/profile"
    echo "[OK] msprof output: output/profile"
fi