#!/bin/bash
set -euo pipefail

# TARGET=ascend910b（默认）或 TARGET=ascend310b
TARGET="${TARGET:-ascend910b}"
case "$TARGET" in
  ascend910b) BLOCK_DIM=16 ;;
  ascend310b) BLOCK_DIM=8 ;;
  *) echo "Unsupported TARGET=$TARGET; use ascend910b or ascend310b" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/custom_ops/src"
JSON_FILE="$PROJECT_DIR/custom_ops/json/moe_lite_ops.json"
GEN_DIR="$PROJECT_DIR/custom_ops/generated"
OUT_DIR="$GEN_DIR/MoELiteOps"
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
  if [ -d "$d/op_host" ] && [ -d "$d/op_kernel" ]; then echo "$d"; return; fi
  if [ -d "$d/MoELiteOps/op_host" ] && [ -d "$d/MoELiteOps/op_kernel" ]; then echo "$d/MoELiteOps"; return; fi
  echo "$d"
}

copy_impls() {
  local dst="$1"
  local -a ops=(
    "MoeTopKLite:moe_top_k_lite"
    "MoeSortQuickSortLite:moe_sort_quick_sort_lite"
    "MoeSortHeapSortLite:moe_sort_heap_sort_lite"
    "MoeTokenPermuteLite:moe_token_permute_lite"
    "MoeTokenUnpermuteLite:moe_token_unpermute_lite"
  )
  for item in "${ops[@]}"; do
    op="${item%%:*}"
    file="${item##*:}"
    cp "$SRC_DIR/$op/op_host/$file.cpp" "$dst/op_host/$file.cpp"
    cp "$SRC_DIR/$op/op_host/${file}_tiling.h" "$dst/op_host/${file}_tiling.h"
    cp "$SRC_DIR/$op/op_kernel/$file.cpp" "$dst/op_kernel/$file.cpp"
    cp "$SRC_DIR/$op/op_kernel/${file}_tiling.h" "$dst/op_kernel/${file}_tiling.h"
  done
}

echo "=== Building MoE sorting lab for $TARGET (BLOCK_DIM=$BLOCK_DIM) ==="
mkdir -p "$GEN_DIR"
rm -rf "$OUT_DIR"

ops=(MoeTopKLite MoeSortQuickSortLite MoeSortHeapSortLite MoeTokenPermuteLite MoeTokenUnpermuteLite)
for i in "${!ops[@]}"; do
  op="${ops[$i]}"
  if [ "$i" -eq 0 ]; then
    "$MSOPGEN" gen -i "$JSON_FILE" -op "$op" -c "ai_core-$TARGET" -lan cpp -out "$OUT_DIR"
  else
    OUT_DIR="$(resolve_project_dir "$OUT_DIR")"
    "$MSOPGEN" gen -i "$JSON_FILE" -op "$op" -c "ai_core-$TARGET" -lan cpp -m 1 -out "$OUT_DIR"
  fi
  OUT_DIR="$(resolve_project_dir "$OUT_DIR")"
done

if [ ! -d "$OUT_DIR/op_host" ] || [ ! -d "$OUT_DIR/op_kernel" ]; then
  echo "Generated project is incomplete: $OUT_DIR" >&2
  exit 1
fi

copy_impls "$OUT_DIR"
sed -i -E "s/static constexpr uint32_t BLOCK_DIM = [0-9]+/static constexpr uint32_t BLOCK_DIM = ${BLOCK_DIM}/g" "$OUT_DIR/op_host"/*.cpp

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

echo "=== Generated project: $OUT_DIR ==="
echo "Install the produced custom_opp_*.run package before running the benchmark."
