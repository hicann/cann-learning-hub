#!/usr/bin/env bash
# 从 cann-recipes-infer 仓库根目录，通过 models/qwen 工作目录启动统一执行器
#
# 用法：
#   bash run_cann_infer.sh <yaml文件名> [cann-recipes-infer仓库路径]
#
# 示例：
#   bash run_cann_infer.sh qwen3_custom_1tp.yaml
#   bash run_cann_infer.sh qwen3_custom_1tp.yaml /mnt/workspace/gitCode/cann/cann-recipes-infer

set -euo pipefail

YAML_FILE="${1:?请提供 YAML 文件名作为第一个参数，如 qwen3_custom_1tp.yaml}"
REPO_PATH="${2:-cann-recipes-infer}"

if [ ! -d "$REPO_PATH" ]; then
  echo "错误：找不到 cann-recipes-infer 仓库路径 $REPO_PATH"
  exit 1
fi

cd "$REPO_PATH"

# 统一执行器 executor/scripts/infer.sh 运行时会将工作目录切换到
# 对应模型子目录（如 models/qwen）读取配置与资源
bash executor/scripts/infer.sh --model qwen --yaml "$YAML_FILE"
