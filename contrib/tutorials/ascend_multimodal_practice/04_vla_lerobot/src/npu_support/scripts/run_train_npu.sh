#!/bin/bash
# ============================================================================
# ACT SO-101 真机数据 NPU 训练启动脚本（基于 CANN 官方样例裁剪）
#
# 本脚本在昇腾 NPU 上对 SO-101 真机方块抓取数据训练 ACT 策略。
# 改编自 CANN 官方样例 cann-recipes-embodied-intelligence/manipulation/act/train
# 主要区别：使用真机数据(local/so101_block)、单卡训练、禁用仿真评测。
#
# 前置条件：
#   1. 已运行 setup_lerobot_npu.sh（clone lerobot + 打 NPU 补丁）
#   2. 已激活含 torch_npu 的 CANN 环境
#   3. 已缓存 resnet18 权重（无外网时，见 SETUP_NPU.md）
#
# 用法：
#   bash src/npu_support/scripts/run_train_npu.sh act_so101_smoke   # 20步快速验证
#   bash src/npu_support/scripts/run_train_npu.sh act_so101          # 完整训练10万步
#   bash src/npu_support/scripts/run_train_npu.sh act_so101 --resume # 续训
#   bash src/npu_support/scripts/run_train_npu.sh act_so101 --nproc 8 # 多卡(若有多卡)
# ============================================================================
set -euo pipefail

# 昇腾训练性能相关环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000
export HOST_CACHE_CAPACITY=20

# ===== 路径解析（适配本课程目录结构）=====
# 脚本位置: src/npu_support/scripts/run_train_npu.sh
# 课程根:   04_vla_lerobot/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAPTER_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"          # 04_vla_lerobot/
SRC_ROOT="$CHAPTER_ROOT/src"
CONFIG_DIR="$SCRIPT_DIR/../configs"

# lerobot 源码目录：默认放在 src/npu_support/lerobot（由 setup 脚本 clone）
LEROBOT_ROOT="${LEROBOT_ROOT:-$SRC_ROOT/npu_support/lerobot}"

# 缓存目录（避免污染 HOME，CANNLab 容器磁盘有限时便于清理）
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SRC_ROOT/npu_support/.cache}"
export HF_HOME="${HF_HOME:-$SRC_ROOT/npu_support/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SRC_ROOT/npu_support/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SRC_ROOT/npu_support/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SRC_ROOT/npu_support/.cache/huggingface/transformers}"
export TOKENIZERS_PARALLELISM=false

# ===== 参数解析 =====
NPROC=1                          # 默认单卡（CANNLab 通常 1*NPU）
MASTER_PORT=29500
MODEL_TYPE=""
USE_RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc)
            NPROC="$2"; shift 2 ;;
        --port)
            MASTER_PORT="$2"; shift 2 ;;
        --resume)
            USE_RESUME=true; shift ;;
        -h|--help)
            echo "用法: $0 <config_name> [--nproc N] [--port P] [--resume]"
            echo "示例:"
            echo "  $0 act_so101_smoke              # 20步smoke验证"
            echo "  $0 act_so101                     # 完整训练"
            echo "  $0 act_so101 --resume            # 从checkpoint续训"
            echo "可用配置: $(ls "$CONFIG_DIR" | sed 's/.yaml//' | tr '\n' ' ')"
            exit 0 ;;
        *)
            if [[ -z "$MODEL_TYPE" ]]; then MODEL_TYPE="$1"; shift
            else echo "未知参数: $1"; exit 1; fi ;;
    esac
done

if [[ -z "$MODEL_TYPE" ]]; then
    echo "错误：必须指定配置名（如 act_so101_smoke 或 act_so101）"
    echo "可用配置: $(ls "$CONFIG_DIR" | sed 's/.yaml//' | tr '\n' ' ')"
    exit 1
fi

CONFIG_PATH="$CONFIG_DIR/${MODEL_TYPE}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# ===== 前置检查 =====
# 检查 lerobot 源码是否存在（用 pyproject.toml 判断，不依赖 .git）
if [[ ! -f "$LEROBOT_ROOT/pyproject.toml" ]]; then
    echo "lerobot 源码未找到: $LEROBOT_ROOT"
    echo "请先运行: bash src/npu_support/scripts/setup_lerobot_npu.sh"
    exit 1
fi

if ! python -c "import torch, torch_npu" 2>/dev/null; then
    echo "torch/torch_npu 不可用，请确认已激活 CANN 环境且安装了 torch_npu"
    exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate 未安装，请在 lerobot 目录执行: pip install -e . --no-deps"
    exit 1
fi

# ===== 输出目录（带时间戳，避免覆盖）=====
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$SRC_ROOT/npu_support/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${MODEL_TYPE}_${TIMESTAMP}.log"

RAW_OUTPUT_DIR=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
RAW_JOB_NAME=$(awk '/^[[:space:]]*job_name:/{gsub(/^[[:space:]]*job_name:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
OUTPUT_DIR_FINAL="${RAW_OUTPUT_DIR}_${TIMESTAMP}"
JOB_NAME_FINAL="${RAW_JOB_NAME}_${TIMESTAMP}"

# ===== 构造训练参数 =====
# 注意：dataset.root 在 yaml 里是相对路径 ../data_final，相对于 lerobot 根目录解析
# 因此需要把真机数据软链或拷贝到 lerobot 同级，这里用绝对路径覆盖更稳妥
TRAIN_ARGS=(--config_path="$CONFIG_PATH")
TRAIN_ARGS+=(--dataset.root="$SRC_ROOT/data_final")   # 用绝对路径指向真机数据
if [[ "$USE_RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume=true)
fi

# ===== accelerate 启动参数 =====
ACCELERATE_ARGS=(--main_process_port "$MASTER_PORT" --num_processes "$NPROC")
if (( NPROC > 1 )); then
    ACCELERATE_ARGS+=(--multi_gpu)
fi

# ===== 启动训练 =====
cd "$LEROBOT_ROOT"
echo "============================================="
echo "ACT 训练启动（昇腾 NPU，真机数据）"
echo "配置: $CONFIG_PATH"
echo "数据: $SRC_ROOT/data_final (local/so101_block)"
echo "输出: $OUTPUT_DIR_FINAL"
echo "日志: $LOG_FILE"
echo "NPU 进程数: $NPROC"
echo "续训: $USE_RESUME"
echo "============================================="

nohup accelerate launch "${ACCELERATE_ARGS[@]}" \
    "$(command -v lerobot-train)" \
    "${TRAIN_ARGS[@]}" \
    --output_dir="$OUTPUT_DIR_FINAL" \
    --job_name="$JOB_NAME_FINAL" \
    > "$LOG_FILE" 2>&1 &
PID=$!

sleep 3
if ! ps -p "$PID" >/dev/null 2>&1; then
    echo "❌ 训练启动失败，查看日志: $LOG_FILE"
    tail -n 40 "$LOG_FILE" || true
    exit 1
fi

echo "✅ 训练已启动，PID: $PID"
echo "💡 实时查看日志: tail -f $LOG_FILE"
echo "💡 训练产物保存在: $OUTPUT_DIR_FINAL"
