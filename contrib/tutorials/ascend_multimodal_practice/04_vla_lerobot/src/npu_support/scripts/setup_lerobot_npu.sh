#!/bin/bash
# ============================================================================
# LeRobot 昇腾 NPU 环境准备脚本（实验4 真机数据版）
#
# 本脚本基于 CANN 官方样例 cann-recipes-embodied-intelligence 的 setup.sh 改编：
#   - clone lerobot 到 src/npu_support/lerobot（而非三级工作区）
#   - checkout 到官方验证过的固定 commit
#   - 应用 NPU 训练补丁（让 LeRobot 支持 ascend 设备）
#   - 安装 ACT 训练依赖（去掉了仿真用的 gym-aloha，真机不需要）
#   - 复用当前环境的 torch / torch_npu（不硬编码下载链接）
#
# 前置条件：
#   1. 已激活 CANNLab 的 CANN 环境（含 mindspore 或 torch_npu）
#   2. 已 source CANN 环境变量：source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
# 用法：
#   bash src/npu_support/scripts/setup_lerobot_npu.sh
#   bash src/npu_support/scripts/setup_lerobot_npu.sh --skip-torch-check   # 跳过torch检查
# ============================================================================
set -euo pipefail

# ===== 路径 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                # 04_vla_lerobot/src
LEROBOT_ROOT="$SRC_ROOT/npu_support/lerobot"
PATCH_PATH="$SCRIPT_DIR/../patches/lerobot_ascend_train_common.patch"

# 官方验证过的 lerobot commit（补丁基于此 commit 编写）
LEROBOT_COMMIT="58f70b6bd370864139a3795ac3497a9eae8c42d5"

SKIP_TORCH_CHECK=false

# ===== ACT 训练依赖（去掉 gym-aloha 仿真包，真机不需要）=====
BASE_DEPS=(
    "datasets>=4.0.0,<4.2.0"
    "diffusers>=0.27.2,<0.36.0"
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0"
    "accelerate>=1.10.0,<2.0.0"
    "setuptools>=71.0.0,<81.0.0"
    "cmake>=3.29.0.1,<4.2.0"
    "einops>=0.8.0,<0.9.0"
    "opencv-python-headless>=4.9.0,<4.13.0"
    "av>=15.0.0,<16.0.0"
    "jsonlines>=4.0.0,<5.0.0"
    "packaging>=24.2,<26.0"
    "pynput>=1.7.7,<1.9.0"
    "pyserial>=3.5,<4.0"
    "wandb>=0.20.0,<0.22.0"
    "draccus==0.10.0"
    "gymnasium>=1.1.1,<2.0.0"
    "rerun-sdk>=0.24.0,<0.27.0"
    "deepdiff>=7.0.1,<9.0.0"
    "imageio[ffmpeg]>=2.34.0,<3.0.0"
    "termcolor>=2.4.0,<4.0.0"
    "torchcodec>=0.1.0"          # 视频解码（真机mp4也需要）
)

# ===== 参数解析 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-torch-check) SKIP_TORCH_CHECK=true; shift ;;
        -h|--help)
            echo "用法: $0 [--skip-torch-check]"
            echo "  --skip-torch-check  跳过末尾的 torch/torch_npu 导入校验"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

command -v git >/dev/null 2>&1 || { echo "错误：需要 git"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "错误：需要 python"; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "错误：需要 pip"; exit 1; }

mkdir -p "$SRC_ROOT/npu_support"

# ===== 1. 获取 lerobot 源码（固定 commit 58f70b6b）=====
# GitHub 在 CANNLab 不稳定，改用本课程 GitCode 仓库的 lerobot-source 分支
# （lerobot 源码已预先上传到 gcw_rz2GAHA1/cann-learning-hub-zy 的 lerobot-source 分支）
GITCODE_LEROBOT_URL="https://gitcode.com/gcw_rz2GAHA1/cann-learning-hub-zy.git"
LEROBOT_BRANCH="lerobot-source"
# lerobot 源码在仓库中的路径（sparse checkout 用）
LEROBOT_REPO_PATH="contrib/tutorials/ascend_multimodal_practice/04_vla_lerobot/src/npu_support/lerobot"

cd "$SRC_ROOT/npu_support"

if [[ -d "$LEROBOT_ROOT/src/lerobot" ]]; then
    echo "[INFO] lerobot 源码已存在: $LEROBOT_ROOT"
else
    echo "[INFO] 从 GitCode 下载 lerobot 源码（commit $LEROBOT_COMMIT）"
    echo "[INFO] 用 sparse checkout 只拉 lerobot 目录（约 3MB，GitCode 网络稳定）"
    # 临时目录做 sparse checkout
    TMP_CLONE=$(mktemp -d)
    cd "$TMP_CLONE"
    git init -q
    git remote add origin "$GITCODE_LEROBOT_URL"
    git config core.sparseCheckout true
    echo "$LEROBOT_REPO_PATH" > .git/info/sparse-checkout
    if ! timeout 180 git fetch origin "$LEROBOT_BRANCH" --depth=1 2>&1; then
        echo "[ERROR] 从 GitCode 拉取失败，请检查网络"
        rm -rf "$TMP_CLONE"
        exit 1
    fi
    git checkout -q "$LEROBOT_BRANCH"
    # 复制到目标位置
    mkdir -p "$LEROBOT_ROOT"
    cp -R "$LEROBOT_REPO_PATH"/* "$LEROBOT_ROOT"/ 2>/dev/null
    cp -R "$LEROBOT_REPO_PATH"/.[!.]* "$LEROBOT_ROOT"/ 2>/dev/null || true
    cd "$SRC_ROOT/npu_support"
    rm -rf "$TMP_CLONE"
    echo "[INFO] ✅ lerobot 源码已下载到 $LEROBOT_ROOT"
fi

# ===== 2. 应用 NPU 补丁 =====
# 进入 lerobot 目录应用补丁（补丁路径是相对 lerobot 根目录的）
cd "$LEROBOT_ROOT"
if [[ ! -f "$PATCH_PATH" ]]; then
    echo "[ERROR] 补丁文件不存在: $PATCH_PATH"
    exit 1
fi
# 用 patch 命令（不依赖 .git，比 git apply 更通用）
# 先检查是否已应用过（patch --dry-run 反向测试）
if patch --dry-run -R -p1 < "$PATCH_PATH" >/dev/null 2>&1; then
    echo "[INFO] 补丁已应用过，跳过: $(basename "$PATCH_PATH")"
elif patch --dry-run -p1 < "$PATCH_PATH" >/dev/null 2>&1; then
    echo "[INFO] 应用 NPU 补丁: $(basename "$PATCH_PATH")"
    patch -p1 < "$PATCH_PATH"
else
    echo "[ERROR] 补丁无法干净应用。请确认 lerobot 源码完整且未被修改。"
    exit 1
fi

# ===== 3. 安装训练依赖（不改动已有的 torch/torch_npu）=====
cd "$LEROBOT_ROOT"
echo "[INFO] 安装 ACT 训练依赖（保留当前 torch 栈不变）"
pip install "${BASE_DEPS[@]}"
pip install -e . --no-deps

# ===== 4. 校验 torch / torch_npu =====
if [[ "$SKIP_TORCH_CHECK" == true ]]; then
    echo "[WARN] 按要求跳过 torch/torch_npu 校验"
elif python -c "import torch, torch_npu" >/dev/null 2>&1; then
    echo "[INFO] ✅ torch / torch_npu 可用"
else
    cat >&2 <<MSG
[ERROR] torch / torch_npu 仍不可用。

本脚本不硬编码 torch_npu 下载链接，因为有效 wheel 组合依赖宿主机架构、CANN 版本和 Ascend 软件栈。

两种解决方式：
  1. 先激活一个已验证可用的 Ascend 训练环境（CANNLab 的 cann_py311 内核通常已预装），再重跑本脚本。
  2. 手动安装匹配的 torch / torch_npu wheel 后重跑：
     pip install /path/to/torch.whl /path/to/torch_npu.whl

确认无误后可加 --skip-torch-check 跳过本检查。
MSG
    exit 1
fi

cat <<MSG

[SUCCESS] ✅ LeRobot 昇腾 NPU 训练环境已就绪。

lerobot 路径: $LEROBOT_ROOT
固定 commit:  $LEROBOT_COMMIT
补丁文件:     $PATCH_PATH
Python:       $(python -V 2>&1)

下一步：
  1. 若机器无外网，提前缓存 resnet18 权重（见 SETUP_NPU.md）
  2. 运行 smoke 测试验证链路：
     bash src/npu_support/scripts/run_train_npu.sh act_so101_smoke
  3. smoke 通过后运行完整训练：
     bash src/npu_support/scripts/run_train_npu.sh act_so101
MSG
