#!/bin/bash
# ATC模型转换脚本: ONNX -> OM
#
# 在香橙派开发板上执行, 将models目录下的ONNX模型转换为output目录下的OM模型。
# 目标芯片: Ascend310B4 (香橙派)
#
# 用法:
#   bash atc_convert.sh              # 转换所有ONNX模型
#   bash atc_convert.sh fp32         # 仅转换FP32模型
#   bash atc_convert.sh pruned       # 仅转换剪枝模型

set -e

SOC_VERSION="Ascend310B4"
MODEL_DIR="../models"
OUTPUT_DIR="../output"

mkdir -p "$OUTPUT_DIR"

convert_model() {
    local name=$1
    local onnx_file="$MODEL_DIR/${name}.onnx"
    local output_prefix="$OUTPUT_DIR/${name}"

    echo "============================================================"
    echo "ATC转换: ${name}.onnx -> ${name}.om"
    echo "============================================================"

    if [ ! -f "$onnx_file" ]; then
        echo "[错误] ONNX文件不存在: $onnx_file"
        return 1
    fi

    atc \
        --framework=5 \
        --model="$onnx_file" \
        --output="$output_prefix" \
        --soc_version="$SOC_VERSION" \
        --input_format=NCHW \
        --input_shape="input:1,1,28,28" \
        --log=error

    local om_file="${output_prefix}.om"
    if [ -f "$om_file" ]; then
        local size=$(du -h "$om_file" | cut -f1)
        echo "[成功] OM模型已生成: $om_file (${size})"
    else
        echo "[失败] OM模型未生成"
        return 1
    fi
    echo ""
}

echo "============================================================"
echo "ATC模型转换 (目标: ${SOC_VERSION})"
echo "============================================================"
echo "ONNX模型目录: $MODEL_DIR"
echo "OM输出目录:   $OUTPUT_DIR"
echo ""

TARGET=${1:-all}

if [ "$TARGET" = "all" ] || [ "$TARGET" = "fp32" ]; then
    convert_model "simplecnn_mnist_fp32"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "pruned" ]; then
    convert_model "simplecnn_mnist_pruned"
fi

echo "============================================================"
echo "ATC转换全部完成"
echo "============================================================"
ls -lh "$OUTPUT_DIR"/*.om 2>/dev/null || echo "无OM文件"
