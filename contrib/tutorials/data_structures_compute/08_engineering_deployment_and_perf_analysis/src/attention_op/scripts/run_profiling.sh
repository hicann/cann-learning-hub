#!/bin/bash
# 对多个 seq_len 执行 msProf 上板性能采集
# 用法：bash scripts/run_profiling.sh [seq_len ...] [--output DIR]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$LAB_DIR/aclnn_runner/build/main_attention_benchmark"
DATA_DIR="$LAB_DIR/data"

SEQ_LENS=()
OUTPUT_DIR="$LAB_DIR/prof"
while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) SEQ_LENS+=("$1"); shift ;;
    esac
done
if [ ${#SEQ_LENS[@]} -eq 0 ]; then SEQ_LENS=(512 1024 2048 4096); fi

if [ ! -f "$RUNNER" ]; then
    echo "runner 未编译，请先执行 bash scripts/build_runner.sh"
    exit 1
fi

for seq_len in "${SEQ_LENS[@]}"; do
    echo "=== seq_len=$seq_len msprof 采集 ==="
    out="$OUTPUT_DIR/prof_$seq_len"
    rm -rf "$out"; mkdir -p "$out"
    source "$SCRIPT_DIR/env_custom_opp.sh"
    msprof op --output="$out" "$RUNNER" "$DATA_DIR" "$seq_len" 2>&1 | tail -8
done
echo "采集完成，报告位于 $OUTPUT_DIR"
