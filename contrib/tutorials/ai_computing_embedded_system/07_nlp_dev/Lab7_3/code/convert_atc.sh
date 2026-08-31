#!/bin/bash
# ============================================================
# Qwen1.5-0.5B-Chat ONNX -> OM 模型转换脚本 (昇腾香橙派 310B)
# ============================================================
# 使用说明：
#   1. 确保已安装 CANN 9.0 完整工具包 (非仅runtime)
#   2. 确保已设置环境变量: source ~/.bashrc (含CANN环境)
#   3. 在项目根目录执行: bash convert_atc.sh
# ============================================================

# 设置CANN环境变量 (根据实际安装路径调整)
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "============================================"
echo "  ATC模型转换: ONNX -> OM (Ascend310B)"
echo "============================================"

# 创建输出目录
mkdir -p ./om_export

# 查看ONNX模型信息
echo "[1/2] ONNX模型信息:"
echo "  路径: ./onnx_export/qwen_merged.onnx"
ls -lh ./onnx_export/qwen_merged.onnx

# ============================================================
# 关键修正说明:
#   原命令使用 --soc_version=Ascend910B3 (错误!)
#   香橙派使用的是 Ascend 310B 芯片
#   应改为 --soc_version=Ascend310B4
#
#   若模型导出时包含 position_ids 输入,
#   需在 --input_shape 中补充 position_ids:1,32
# ============================================================

# 执行ATC转换
echo ""
echo "[2/2] 开始ATC转换..."
atc --framework=5 \
    --model='./onnx_export/qwen_merged.onnx' \
    --output='./om_export/qwen_merged' \
    --soc_version=Ascend310B4 \
    --input_shape='input_ids:1,32;attention_mask:1,32' \
    --log=error

# 检查转换结果
if [ -f "./om_export/qwen_merged.om" ]; then
    echo ""
    echo "============================================"
    echo "  转换成功!"
    echo "============================================"
    ls -lh ./om_export/qwen_merged.om
    echo ""
    echo "现在可以运行: python3 qwen1.5-0.5b-chat.py"
else
    echo ""
    echo "============================================"
    echo "  转换失败，请检查错误信息"
    echo "============================================"
    echo "常见问题:"
    echo "  1. soc_version不匹配 -> 确认芯片型号(Ascend310B3或Ascend310B4)"
    echo "  2. input_shape缺少position_ids -> 检查ONNX模型输入节点"
    echo "  3. 内存不足 -> 增大swap或减小seq_len"
    echo "  4. CANN环境未设置 -> source set_env.sh"
fi
