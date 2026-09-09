#!/bin/bash
set -euo pipefail

# 目标平台：TARGET=ascend910b（默认）或 TARGET=ascend310b
TARGET="${TARGET:-ascend910b}"
case "$TARGET" in
  ascend910b) ;;
  ascend310b) ;;
  *) echo "Unsupported TARGET=$TARGET; use ascend910b or ascend310b" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/custom_ops/src"
JSON_FILE="$PROJECT_DIR/custom_ops/json/tree_queue_lite_ops.json"
GEN_DIR="$PROJECT_DIR/custom_ops/generated"
OUT_DIR="$GEN_DIR/TreeQueueLiteOps"
LOCAL_OPP_ROOT="$GEN_DIR/local_opp"
VENDOR_NAME="customize"
ASCEND_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

find_msopgen() {
  local candidate
  for candidate in \
    "$ASCEND_PATH/python/site-packages/bin/msopgen" \
    "$ASCEND_PATH/bin/msopgen"; do
    if [ -x "$candidate" ]; then echo "$candidate"; return 0; fi
  done
  command -v msopgen 2>/dev/null || true
}

MSOPGEN="$(find_msopgen)"
if [ -z "$MSOPGEN" ]; then
  echo "msopgen not found. Source CANN first, for example:" >&2
  echo "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh" >&2
  exit 1
fi

resolve_project_dir() {
  local d="$1"
  if [ -d "$d/op_host" ] && [ -d "$d/op_kernel" ]; then echo "$d"; return 0; fi
  if [ -d "$d/TreeQueueLiteOps/op_host" ] && [ -d "$d/TreeQueueLiteOps/op_kernel" ]; then
    echo "$d/TreeQueueLiteOps"; return 0
  fi
  echo "$d"
}

echo "=== Building TreeQueuePipelineLite for target: $TARGET ==="
mkdir -p "$GEN_DIR"
rm -rf "$OUT_DIR"
"$MSOPGEN" gen -i "$JSON_FILE" -op TreeQueuePipelineLite \
  -c "ai_core-$TARGET" -lan cpp -out "$OUT_DIR"
OUT_DIR="$(resolve_project_dir "$OUT_DIR")"

if [ ! -d "$OUT_DIR/op_host" ] || [ ! -d "$OUT_DIR/op_kernel" ]; then
  echo "Generated project is incomplete: $OUT_DIR" >&2
  exit 1
fi

cp "$SRC_DIR/TreeQueuePipelineLite/op_host/tree_queue_pipeline_lite.cpp" "$OUT_DIR/op_host/tree_queue_pipeline_lite.cpp"
cp "$SRC_DIR/TreeQueuePipelineLite/op_host/tree_queue_pipeline_lite_tiling.h" "$OUT_DIR/op_host/tree_queue_pipeline_lite_tiling.h"
cp "$SRC_DIR/TreeQueuePipelineLite/op_kernel/tree_queue_pipeline_lite.cpp" "$OUT_DIR/op_kernel/tree_queue_pipeline_lite.cpp"
cp "$SRC_DIR/TreeQueuePipelineLite/op_kernel/tree_queue_pipeline_lite_tiling.h" "$OUT_DIR/op_kernel/tree_queue_pipeline_lite_tiling.h"

if [ -f "$OUT_DIR/CMakePresets.json" ]; then
  python3 "$SCRIPT_DIR/patch_cmakepresets.py" "$OUT_DIR/CMakePresets.json" "$TARGET"
fi

cd "$OUT_DIR"
rm -rf build_out
mkdir -p build_out
source "$ASCEND_PATH/set_env.sh" 2>/dev/null || true
if [ -f CMakePresets.json ]; then
  cmake -S . -B build_out --preset=default
else
  cmake -S . -B build_out -DASCEND_COMPUTE_UNIT="$TARGET"
fi
cmake --build build_out --target binary -j1
cmake --build build_out --target package -j1

RUN_PKG="$(find build_out -maxdepth 1 -name 'custom_opp_*.run' | head -1 || true)"
if [ -z "$RUN_PKG" ]; then
  echo "No custom_opp_*.run package found under $OUT_DIR/build_out" >&2
  exit 1
fi

# 把自定义算子安装到本实验目录下的 local_opp，不写系统 CANN 的 opp/vendors，
# 普通 Notebook 用户无需 sudo / chmod / chown 即可完成安装。
# 安装器会把已部署文件设为只读；先恢复写权限再删除，否则重复安装时 rm -rf 失败。
if [ -e "$LOCAL_OPP_ROOT" ]; then
  chmod -R u+w "$LOCAL_OPP_ROOT" 2>/dev/null || true
  rm -rf "$LOCAL_OPP_ROOT"
fi
mkdir -p "$LOCAL_OPP_ROOT"
bash "$RUN_PKG" --install-path="$LOCAL_OPP_ROOT"

LOCAL_VENDOR_DIR="$LOCAL_OPP_ROOT/vendors/$VENDOR_NAME"
if [ ! -f "$LOCAL_VENDOR_DIR/bin/set_env.bash" ]; then
  echo "Custom OPP installation is incomplete: $LOCAL_VENDOR_DIR/bin/set_env.bash is missing" >&2
  exit 1
fi

echo "=== Generated project: $OUT_DIR ==="
echo "Local custom OPP: $LOCAL_VENDOR_DIR"
echo "Recommended runtime environment:"
echo "  source ${PROJECT_DIR}/scripts/env_custom_opp.sh"
