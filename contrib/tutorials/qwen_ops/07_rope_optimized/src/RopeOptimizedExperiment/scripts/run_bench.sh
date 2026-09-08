#!/usr/bin/env bash
set -eo pipefail

QWEN_OPS_COURSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$QWEN_OPS_COURSE_ROOT/scripts/setup_cannlab_env.sh"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
else
    echo "[ERROR] Set ASCEND_HOME_PATH to CANN toolkit root." >&2
    exit 1
fi
[[ -f "$CANN_PATH/set_env.sh" ]] && source "$CANN_PATH/set_env.sh"
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && { source /usr/local/Ascend/driver/bin/setenv.bash 2>/dev/null || true; }
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0]+'/lib')" 2>/dev/null || echo "")
[[ -n "$TORCH_LIB" ]] && export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/out/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT_DIR"
exec python3 "$@"
