# 实验4 昇腾 NPU 训练集成说明

> 本方案基于 CANN 官方样例 [cann-recipes-embodied-intelligence/manipulation/act/train](https://gitcode.com/cann/cann-recipes-embodied-intelligence/tree/master/manipulation/act/train) 裁剪适配，让实验4的 **SO-101 真机数据**能在 **CANNLab 昇腾 NPU** 上训练。

## 一、方案定位（先看这个）

官方样例是「仿真数据 + 仿真评测 + MuJoCo 渲染」的完整闭环，本方案**只取其中的训练部分**，适配真机场景：

| 环节 | 官方样例 | 本方案（实验4真机） |
|------|----------|---------------------|
| 数据 | aloha_sim 仿真（14维双臂） | **so101_block 真机（6维单臂）** |
| 训练设备 | 昇腾 NPU 8卡 | 昇腾 NPU（默认单卡） |
| 评测 | MuJoCo 在线仿真（68%成功率） | **禁用**（真机无仿真环境） |
| 渲染 | osmesa 渲染视频 | **不用**（真机数据已是mp4） |

**为什么这样裁剪**：真机数据没有对应的 MuJoCo 仿真场景，无法做"在线成功率"评测和渲染可视化。真机的评测方式是训练后在 04.04 notebook 做离线动作对比，或部署到真机评估。

## 二、目录结构

```
04_vla_lerobot/src/npu_support/
├── SETUP_NPU.md                          ← 本文件
├── patches/
│   └── lerobot_ascend_train_common.patch ← 官方 NPU 补丁（来自 CANN 仓库）
├── configs/
│   ├── act_so101_smoke.yaml              ← smoke 测试配置（20步）
│   └── act_so101.yaml                    ← 完整训练配置（5000步）
├── scripts/
│   ├── setup_lerobot_npu.sh              ← 环境准备（下载lerobot+补丁+装依赖）
│   └── run_train_npu.sh                  ← 训练启动脚本
├── lerobot/                              ← setup 后下载的 lerobot 源码（运行时生成）
├── .cache/                               ← 缓存（含 ResNet18 权重，运行时生成）
├── logs/                                 ← 训练日志（运行时生成）
└── ../ckpt/                              ← 训练产物 checkpoint（运行时生成）
```

## 三、运行步骤（完整流程，已验证）

### 前置条件

1. CANNLab 云开发环境，模板 `cann_8.5.2-py3.11-A2-arm`，规格 `1*NPU 910B3 16vCPUs 32GiB`
2. 已激活 **`Python 3.11.4 (CANN)`** 内核（该内核通常已预装 torch / torch_npu）
3. 已 source CANN 环境（内核通常自动完成）：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

> 📌 **关于 Python 版本**：CANN 官方样例文档写的是 "Python 3.10"，但那只是官方**验证时**的版本。补丁代码使用的是 3.10+ 语法，**Python 3.11.4 完全兼容**。CANNLab 的 `Python 3.11.4 (CANN)` 内核就是 3.11.4，无需降级。

### 步骤 1：准备 NPU 训练环境（首次，约 5-10 分钟）

在 `04_vla_lerobot/` 目录下执行：

```bash
cd contrib/tutorials/ascend_multimodal_practice/04_vla_lerobot
bash src/npu_support/scripts/setup_lerobot_npu.sh
```

脚本会（**已针对 CANNLab 优化**）：
1. 从 **GitCode 仓库的 `lerobot-source` 分支**下载 lerobot 源码（commit `58f70b6b`，约 3MB）
   > ⚠️ 不直接 clone GitHub（CANNLab 访问 GitHub 极不稳定，经常卡死）。源码已预先上传到本课程 GitCode 仓库。
2. 应用 NPU 补丁（让 LeRobot 识别 ascend 设备 + 放宽视频时间戳容差）
3. 安装 ACT 训练依赖（**保留当前环境的 torch/torch_npu 不变**）
4. 校验 `torch` / `torch_npu` 可用

成功标志：最后显示 `[SUCCESS] ✅ LeRobot 昇腾 NPU 训练环境已就绪`

> ⚠️ 若报 `torch/torch_npu 不可用`：确认选了 CANN 模板并激活了 Python 3.11.4 (CANN) 内核。确认无误后可加 `--skip-torch-check`。

### 步骤 2：缓存 ResNet18 权重（重要，否则训练会卡在下载）

ACT 用 ResNet18 做视觉主干，首次会尝试从 `download.pytorch.org` 下载权重（44.7MB）。**CANNLab 访问 pytorch.org 很慢（10-50 kB/s，可能卡几十分钟）**，必须提前缓存：

```bash
cd contrib/tutorials/ascend_multimodal_practice/04_vla_lerobot
mkdir -p src/npu_support/.cache/torch/hub/checkpoints/
curl -L -o src/npu_support/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  "https://download.pytorch.org/models/resnet18-f37072fd.pth"
ls -lh src/npu_support/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

> 💡 用 `curl` 直接下载会比 Python 的 torch.hub 快（curl 走单连接，约 1-2 分钟下完 44.7MB）。
> 如果 pytorch.org 也慢，可用镜像：
> ```bash
> curl -L -o src/npu_support/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
>   "https://hub.gitmirror.com/https://download.pytorch.org/models/resnet18-f37072fd.pth"
> ```

### 步骤 3：运行 smoke 测试（20步，约 2-3 分钟）

```bash
bash src/npu_support/scripts/run_train_npu.sh act_so101_smoke
```

**成功标志**（看日志 `tail -f src/npu_support/logs/train_act_so101_smoke_*.log`）：

```
INFO ... Start offline training on a fixed dataset
Training:   5%|▌  | 1/20 ... loss:89.456       ← 第1步 loss 出来
Training:  50%|█████  | 10/20 ... loss:xx       ← loss 在下降
Training: 100%|██████████| 20/20 ... loss:10.2  ← 🎉 全部完成
INFO ... Checkpoint policy after step 20
INFO ... End of training
```

**验证通过的 loss 趋势参考**（实测）：
- step 1：loss ≈ 89.5
- step 17：loss ≈ 10.3
- step 20：loss ≈ 10.2（收敛，降了 88%）

训练产物：`src/ckpt/act_so101_smoke_<时间戳>/`（含 checkpoints）

> ✅ **smoke 通过 = NPU 训练链路完全打通**：数据加载 → 视频解码(pyav) → ResNet18 → ACT模型 → NPU前向反向 → checkpoint 保存，全链路 OK。

### 步骤 4：完整训练（5000步，约 2-4 小时）

```bash
bash src/npu_support/scripts/run_train_npu.sh act_so101
```

训练日志：`src/npu_support/logs/train_act_so101_*.log`（用 `tail -f` 实时查看）
训练产物：`src/ckpt/act_so101_<时间戳>/`（含 checkpoints / final / config.json）

> 💡 完整训练耗时较长，可放后台跑（脚本已用 `nohup` 启动，关掉终端不影响）。用 `tail -f` 随时看进度。

### 步骤 5：用训练结果做离线评测

训练完成后，回到 notebook `04.04_eval_practice.ipynb`，把模型路径指向新训练的 checkpoint：

```python
model_paths = [
    "src/ckpt/act_so101_<时间戳>/checkpoints/last",   # ← 自己训练的
    "src/pretrained_model",                              # 课程提供的预训练模型
]
```

## 四、关键配置说明（已在 configs/*.yaml 中预设）

本方案针对 CANNLab 环境做了以下**关键调整**（已写死在配置里，学习者无需手动改）：

| 配置项 | 官方默认 | 本方案 | 原因 |
|--------|----------|--------|------|
| `video_backend` | `torchcodec` | **`pyav`** | torchcodec 和 NPU 版 PyTorch 不兼容（加载不了 libtorchcodec），pyav 可用 |
| `num_workers` | `4` | **`0`** | CANNLab `/dev/shm`（共享内存）太小，多 worker 会 Bus error 崩溃 |
| `policy.device` | `cuda` | **`npu`** | NPU 补丁注册了 ascend 设备 |
| `policy.use_amp` | `true` | **`false`** | NPU 上 AMP 支持不完善，关闭更稳 |

## 五、数据说明

本方案**直接使用课程提供的真机数据**（04.02 自动下载）：

| 项 | 值 |
|----|-----|
| 数据集 | `local/so101_block` |
| 路径 | `src/data_final/`（100 episodes，449MB，从 ModelScope 自动下载） |
| 格式 | LeRobot v3（parquet + mp4） |
| 动作维度 | 6（SO-101 单臂） |
| 摄像头 | front + wrist（640×360@30fps，av1 编码） |

## 六、与 notebook 训练方式的关系

实验4 提供两条训练路径，按你的环境二选一：

| 路径 | 适用环境 | 怎么用 |
|------|----------|--------|
| **A. notebook 训练**（04.03 Cell[1]） | 有 NVIDIA GPU，或仅做 CPU 流程验证 | 直接在 notebook 里跑 `lerobot-train`（pip 安装版） |
| **B. NPU 脚本训练**（本方案） | **CANNLab 昇腾 NPU** | 用 `npu_support/scripts/` 下的脚本（需下载源码打补丁） |

> LeRobot 官方至今**不支持昇腾 NPU**（`is_torch_device_available` 硬编码只认 cuda/mps/xpu/cpu），必须用补丁。所以 **NPU 训练走脚本（路径B）**，不能直接在 notebook 里 `pip install lerobot` 了事。04.03 的 Cell[1] 会自动检测设备并引导你走对应路径。

## 七、常见问题（FAQ）

| 问题 | 原因 | 解决 |
|------|------|------|
| `import torch_npu` 失败 | 未激活 CANN 环境 | 选 CANN 模板 + 激活 Python 3.11.4 (CANN) 内核 |
| `git fetch` 卡死（GitHub） | CANNLab 访问 GitHub 不稳定 | 本方案已改用 GitCode 下载（步骤1），不依赖 GitHub |
| ResNet18 下载卡住 | pytorch.org 慢 | 按步骤2用 curl 预先缓存权重 |
| `Could not load libtorchcodec` | torchcodec 和 NPU 不兼容 | 已在配置改用 `pyav`（步骤1的补丁+配置已处理） |
| `DataLoader worker Bus error` | `/dev/shm` 共享内存不足 | 已在配置改 `num_workers: 0`（配置已处理） |
| `KeyError: 'observation.images.front'` | stats.json 缺图片特征统计 | 数据集已更新 stats.json（ModelScope 上的版本已修复） |
| `query timestamps violate the tolerance` | 真机数据时间戳微秒级偏差 | 已在补丁放宽 torchvision 容差（步骤1的补丁已处理） |
| 视频解码报错（av1） | av1 编码兼容性 | pyav 已能解 av1；若仍报错可转 h264（见下方） |
| wandb 警告 | wandb 版本不匹配 | 配置已 `enable: false`，无需登录，忽略警告 |

**视频转 h264（备用，仅当 pyav 解 av1 报错时）**：
```bash
for f in src/data_final/videos/*/chunk-*/*.mp4; do
  ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 23 "$f.tmp.mp4" && mv "$f.tmp.mp4" "$f"
done
```

## 八、参考来源

- CANN 官方样例：https://gitcode.com/cann/cann-recipes-embodied-intelligence/tree/master/manipulation/act/train
- 官方验证结果：ACT on aloha_sim，8卡，100k步（官方），68% 成功率（仿真评测，本方案未采用）
- 补丁原理：在 lerobot 的 `utils.py` 注册 `npu` 设备，在 `lerobot_train.py` 注入 `transfer_to_npu`，在 `video_utils.py` 放宽视频时间戳容差
