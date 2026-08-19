#!/usr/bin/env bash
# QwenRoPeCustom — 测试运行脚本
# 自动检测 CANN 环境并设置 LD_LIBRARY_PATH, 然后运行测试
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ── CANN 环境 ──────────────────────────────────────────────
CANN_PATH=""
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d "/usr/local/Ascend/ascend-toolkit/latest" ]]; then
    CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"
elif [[ -d "$HOME/CANN/cann-9.0.0" ]]; then
    CANN_PATH="$HOME/CANN/cann-9.0.0"
elif compgen -G "$HOME/CANN/cann-*" &>/dev/null; then
    CANN_PATH=$(ls -d "$HOME"/CANN/cann-* 2>/dev/null | sort -V | tail -1)
elif compgen -G "/usr/local/Ascend/cann-*" &>/dev/null; then
    CANN_PATH=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -1)
fi

if [[ -n "$CANN_PATH" ]]; then
    if [[ -f "$CANN_PATH/set_env.sh" ]]; then
        source "$CANN_PATH/set_env.sh"
    elif [[ -f "$CANN_PATH/bin/setenv.bash" ]]; then
        source "$CANN_PATH/bin/setenv.bash"
    fi
fi
[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && source /usr/local/Ascend/driver/bin/setenv.bash

# ── Torch lib ──────────────────────────────────────────────
TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0]+'/lib')" 2>/dev/null || echo "")
if [[ -n "$TORCH_LIB" ]]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

# ── 项目 lib ───────────────────────────────────────────────
export LD_LIBRARY_PATH="$ROOT_DIR/out/lib:$LD_LIBRARY_PATH"

# ── 运行测试 ───────────────────────────────────────────────
cd "$ROOT_DIR"
exec python3 "$@"
