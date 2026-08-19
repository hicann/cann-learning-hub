#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROWS="${ROWS:-128}"
HIDDEN="${HIDDEN:-1024}"
BLOCK_DIM="${BLOCK_DIM:-8}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
ROUNDS="${ROUNDS:-5}"
EPS="${EPS:-1e-6}"
BINARY="$ROOT_DIR/out/bin/rmsnorm_baseline_standalone"

[[ -x "$BINARY" ]] || { echo "[ERROR] Run scripts/build.sh first."; exit 1; }

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
elif [[ -d /home/user/ascend-toolkit/cann-8.5.0 ]]; then
    CANN_PATH=/home/user/ascend-toolkit/cann-8.5.0
else
    echo "[ERROR] Set ASCEND_HOME_PATH to CANN toolkit root."
    exit 1
fi

set +u
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash
[[ -f "$CANN_PATH/set_env.sh" ]] && source "$CANN_PATH/set_env.sh"
set -u

export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:${LD_LIBRARY_PATH:-}"

"$BINARY" --rows "$ROWS" --hidden "$HIDDEN" --block-dim "$BLOCK_DIM" --warmup "$WARMUP" --repeat "$REPEAT" --rounds "$ROUNDS" --eps "$EPS"

if command -v msprof >/dev/null 2>&1; then
    mkdir -p "$ROOT_DIR/output/profile"
    chmod 755 "$ROOT_DIR/output" "$ROOT_DIR/output/profile"
    chmod 755 "$ROOT_DIR/output" "$ROOT_DIR/output/profile"
    msprof --application="$BINARY --rows $ROWS --hidden $HIDDEN --block-dim $BLOCK_DIM --warmup 5 --repeat 10 --rounds 1 --eps $EPS" \
        --output="$ROOT_DIR/output/profile" \
        --ai-core=on \
        --aic-mode=task-based \
        --aic-metrics=PipeUtilization
    echo "[OK] msprof output: output/profile"
fi
