#!/bin/bash
# moe_router_fused 一键回归（M4）：环境检查 + 8 用例正确性 + 可选确定性复查
# 用法:
#   bash tools/run_all.sh                # 全部用例各跑 1 次
#   bash tools/run_all.sh --repeats 3    # 每用例重复 3 次（确定性检查）
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 环境：ASCEND_HOME_PATH（允许外部已 source set_env.sh）
if [ -z "$ASCEND_HOME_PATH" ]; then
    for cand in /usr/local/Ascend/ascend-toolkit/latest "$HOME/Ascend/cann-9.0.0" /usr/local/Ascend/cann-9.0.0; do
        if [ -f "$cand/set_env.sh" ]; then
            # shellcheck disable=SC1091
            source "$cand/set_env.sh" >/dev/null 2>&1
            break
        fi
    done
fi
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[ERROR] 未找到 CANN 环境，请先 source \$ASCEND_HOME_PATH/set_env.sh" >&2
    exit 1
fi
echo "[env] ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

# 算子包：run.sh 依赖 build_out/_CPack_Packages/... 已生成
if [ ! -d "$ROOT/src/custom_op/build_out/_CPack_Packages" ]; then
    echo "[build] 算子包未构建，先执行 src/custom_op/build.sh ..."
    bash "$ROOT/src/custom_op/build.sh" >/dev/null
fi

python3 "$ROOT/tools/test_moe_router.py" "$@"
