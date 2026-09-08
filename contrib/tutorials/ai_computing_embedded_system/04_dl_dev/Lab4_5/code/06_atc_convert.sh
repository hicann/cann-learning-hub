#!/bin/bash
# 06_atc_convert.sh - ATC 模型转换脚本
# 实验4.5 昇腾香橙派部署深度学习网络实验
#
# 使用昇腾 ATC (Ascend Tensor Compiler) 工具
# 将 ONNX 模型转换为昇腾 OM (Offline Model) 离线模型
#
# OM 模型是昇腾 NPU 的专用模型格式，部署到香橙派上运行
#
# 运行方式 (在安装了 CANN 工具链的环境上):
#   bash 06_atc_convert.sh
#
# 输出:
#   models/student_model.om  - 昇腾离线模型

echo "======================================================"
echo "步骤6: ATC 模型转换 (ONNX -> OM)"
echo "======================================================"

# 环境变量设置 (根据实际 CANN 安装路径调整)
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 输入输出路径
ONNX_MODEL="./models/student_model.onnx"
OM_MODEL="./models/student_model.om"

# 检查 ONNX 模型是否存在
if [ ! -f "$ONNX_MODEL" ]; then
    echo "错误: ONNX 模型 $ONNX_MODEL 不存在"
    echo "请先运行: python 05_export_onnx.py"
    exit 1
fi

echo "输入模型: $ONNX_MODEL"
echo "输出模型: $OM_MODEL"

# ATC 转换命令
# --framework=5: ONNX 框架
# --soc_version: 芯片型号
#   - Ascend310B: 香橙派 AIPro (Ascend 310B)
#   - Ascend910B: 昇腾 910B 服务器
# --input_shape: 输入张量形状 (batch_size, channels, height, width)
atc --framework=5 \
    --model="$ONNX_MODEL" \
    --output="${OM_MODEL%.om}" \
    --soc_version=Ascend310B3 \
    --input_shape="input:1,3,224,224" \
    --input_format=NCHW \
    --log=info

# 检查转换结果
if [ -f "$OM_MODEL" ]; then
    echo ""
    echo "转换成功!"
    echo "OM 模型: $OM_MODEL"
    ls -lh "$OM_MODEL"
    echo ""
    echo "下一步: 将 OM 模型部署到香橙派进行推理"
    echo "  运行: python 07_acl_inference.py"
else
    echo "转换失败，请检查 ATC 日志"
    exit 1
fi
