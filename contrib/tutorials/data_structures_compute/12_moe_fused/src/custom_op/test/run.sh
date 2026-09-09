#!/bin/bash
# 编译并运行 MoeRouterFused 正确性测试
# 用法: bash run.sh <case_dir> [--device <id>] [--dump <prefix>] [--bench <iters>]
# case_dir 支持相对路径（按调用方当前目录解析，脚本内部会先转为绝对路径）
set -e

# 先把用例目录参数转为绝对路径（后面会 cd 到 test/ 目录）
if [ -n "${1:-}" ]; then
    CASE_ARG="$(realpath -m "$1")"
    set -- "$CASE_ARG" "${@:2}"
fi

CUSTOM_OP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_OUT="$CUSTOM_OP_ROOT/build_out"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set" >&2
    exit 1
fi

# ---- 部署算子包到 build_out/opp_pkg，作为 ASCEND_CUSTOM_OPP_PATH ----
PKG_SRC="$BUILD_OUT/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/customize"
DEPLOY="$BUILD_OUT/opp_pkg"
if [ ! -d "$PKG_SRC/op_api" ]; then
    echo "[ERROR] package not found at $PKG_SRC, run build.sh first" >&2
    exit 1
fi
rm -rf "$DEPLOY"
mkdir -p "$DEPLOY"
cp -r "$PKG_SRC/." "$DEPLOY/"

export ASCEND_CUSTOM_OPP_PATH="$DEPLOY"
export LD_LIBRARY_PATH="$DEPLOY/op_api/lib:$DEPLOY/op_impl/ai_core/tbe/op_tiling:$DEPLOY/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$DEPLOY/op_proto/lib/linux/aarch64:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH}"

# ---- 编译测试程序 ----
mkdir -p "$TEST_DIR/build"
cd "$TEST_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j"$(nproc)" > /dev/null

# ---- 运行 ----
cd "$TEST_DIR"
./build/execute_moe_router_fused "$@"
