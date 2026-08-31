#!/bin/bash
# ============================================================
# export_om.sh - 昇腾香橙派上 YOLOv8 模型 ONNX -> OM 转换脚本
# ============================================================
# 用法:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash export_om.sh
#
# 说明:
#   1. 将 YOLOv8n 的 ONNX 模型转换为昇腾 310B4 的 OM 模型
#   2. 分别生成纯 OM (无AIPP) 和 AIPP-OM (含AIPP预处理)
#   3. 香橙派 AIPro 的 SoC 版本为 Ascend310B4
# ============================================================

set -e

# === 配置 ===
ONNX_MODEL=${1:-"yolov8n.onnx"}
OUTPUT_DIR=${2:-"../output"}
AIPP_CFG="aipp_yolo.cfg"
SOC_VERSION="Ascend310B4"
INPUT_SHAPE="images:1,3,640,640"

mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "  昇腾香橙派 YOLOv8 OM 模型转换"
echo "  SoC: $SOC_VERSION"
echo "  ONNX: $ONNX_MODEL"
echo "======================================================"

# === 检查 ATC 是否可用 ===
if ! command -v atc &> /dev/null; then
    echo "[ERROR] atc 命令未找到，请先加载 CANN 环境:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

# === 1. 转换纯 OM (无 AIPP，输入为 float32) ===
echo ""
echo "[1/2] 转换纯 OM (无 AIPP)..."
atc \
    --model="$ONNX_MODEL" \
    --framework=5 \
    --output="$OUTPUT_DIR/yolov8n_pure" \
    --soc_version="$SOC_VERSION" \
    --input_format=NCHW \
    --input_shape="$INPUT_SHAPE" \
    --output_type=FP32 \
    --log=error

echo "[OK] 纯 OM: $OUTPUT_DIR/yolov8n_pure.om"

# === 2. 转换 AIPP-OM (含 AIPP，输入为 uint8) ===
echo ""
echo "[2/2] 转换 AIPP-OM (含 AIPP 预处理)..."
atc \
    --model="$ONNX_MODEL" \
    --framework=5 \
    --output="$OUTPUT_DIR/yolov8n_aipp" \
    --soc_version="$SOC_VERSION" \
    --input_format=NCHW \
    --input_shape="$INPUT_SHAPE" \
    --insert_op_conf="$AIPP_CFG" \
    --output_type=FP32 \
    --log=error

echo "[OK] AIPP-OM: $OUTPUT_DIR/yolov8n_aipp.om"

echo ""
echo "======================================================"
echo "  转换完成！"
echo "  纯 OM:     $OUTPUT_DIR/yolov8n_pure.om  (float32 输入, 4.69 MB)"
echo "  AIPP-OM:   $OUTPUT_DIR/yolov8n_aipp.om  (uint8 输入, 1.17 MB)"
echo "======================================================"
