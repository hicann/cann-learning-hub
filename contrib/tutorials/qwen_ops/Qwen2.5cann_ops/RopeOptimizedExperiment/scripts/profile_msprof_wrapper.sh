#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CANN_PATH="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-8.5.2}"
if [[ -f "$CANN_PATH/set_env.sh" ]]; then
    source "$CANN_PATH/set_env.sh"
elif [[ -f "$CANN_PATH/bin/setenv.bash" ]]; then
    source "$CANN_PATH/bin/setenv.bash"
else
    echo "[ERROR] CANN environment script not found under $CANN_PATH" >&2
    exit 1
fi
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash

TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0]+'/lib')" 2>/dev/null || echo "")
[[ -n "$TORCH_LIB" ]] && export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:$LD_LIBRARY_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

echo "[WRAPPER] CANN sourced"
echo "[WRAPPER] LD_LIBRARY_PATH entries: $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | wc -l)"

cd "$ROOT_DIR"
exec python3 scripts/profile_target.py
