#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# CANN environment (CannLearningLab: /home/developer/Ascend/cann-8.5.2)
CANN_PATH="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-8.5.2}"
[[ -f "$CANN_PATH/set_env.sh" ]] && { source "$CANN_PATH/set_env.sh" 2>/dev/null || true; }
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && { source /usr/local/Ascend/driver/bin/setenv.bash 2>/dev/null || true; }
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0]+'/lib')" 2>/dev/null || echo "")
[[ -n "$TORCH_LIB" ]] && export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:$LD_LIBRARY_PATH"

cd "$ROOT_DIR"
exec python3 "$@"
