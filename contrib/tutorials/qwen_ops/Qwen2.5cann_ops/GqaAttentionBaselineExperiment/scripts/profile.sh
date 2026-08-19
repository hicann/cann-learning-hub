#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BINARY="$ROOT_DIR/out/bin/gqa_attention_baseline_standalone"
[[ -x "$BINARY" ]] || { echo "[ERROR] Run scripts/build.sh first."; exit 1; }
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then CANN_PATH="$ASCEND_HOME_PATH"; elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest; elif [[ -d /home/user/Ascend/ascend-toolkit/cann-8.5.0 ]]; then CANN_PATH=/home/user/Ascend/ascend-toolkit/cann-8.5.0; else echo "[ERROR] Set ASCEND_HOME_PATH"; exit 1; fi
set +u; [[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash; [[ -f "$CANN_PATH/set_env.sh" ]] && source "$CANN_PATH/set_env.sh"; set -u
export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:${LD_LIBRARY_PATH:-}"
ARGS="--batch ${BATCH:-1} --q-heads ${Q_HEADS:-8} --kv-heads ${KV_HEADS:-2} --q-len ${Q_LEN:-32} --kv-len ${KV_LEN:-32} --head-dim ${HEAD_DIM:-64} --block-dim ${BLOCK_DIM:-8} --warmup ${WARMUP:-10} --repeat ${REPEAT:-50} --rounds ${ROUNDS:-5} --causal ${CAUSAL:-1}"
"$BINARY" $ARGS
if command -v msprof >/dev/null 2>&1; then mkdir -p "$ROOT_DIR/output/profile"; msprof --application="$BINARY $ARGS" --output="$ROOT_DIR/output/profile" --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization; fi
