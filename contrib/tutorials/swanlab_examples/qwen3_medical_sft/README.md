# Qwen3 医学场景全参微调与 NPU 推理接入实战

在昇腾 NPU 上，参考 [SwanLab Qwen3 医学模型微调教程](https://docs.swanlab.cn/course/llm_train_course/03-sft/4.qwen3-medical-finetune/) 完成 Qwen3-1.7B 的全参微调，训练出一个具备医学问答能力的模型；再把训练产出的权重接入昇腾官方推理框架 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)，在 NPU 上跑起来验证效果。

完整代码与分步说明见 `qwen3_medical_sft.ipynb`；本 README 仅覆盖环境相关信息与文件导览。

## 体验配置

| 项目 | 数值 |
| --- | --- |
| NPU 卡数 | 单卡（本文实测环境为昇腾 Atlas A3，910C 芯片，64GB HBM） |
| 基座模型 | Qwen3-1.7B |
| 训练精度 | BF16 |
| 训练样本数 | 2166 条（训练集）/ 241 条（验证集），共 2407 条，按 9:1 切分 |
| max_length | 2048 |
| 训练步数控制 | `max_steps=680`（约合 5 个 epoch 上限），配合 `EarlyStoppingCallback` 提前停止，实测收敛于约 650 步附近 |
| batch size / 梯度累积 | `per_device_train_batch_size=2`，`gradient_accumulation_steps=8` |
| 推理耗时（`cann-recipes-infer` 实测，两次独立复现结果一致） | Prefill 27.95 ms，Decode 平均 5.18 ms |

以上配置为本文实际验证环境，未测试更低配置下的可行性；若显存较小，可参考 Notebook 训练部分的 `per_device_train_batch_size`、`gradient_accumulation_steps` 参数，酌情调小 batch size 或开启更激进的显存优化策略。

## 前置条件

| 项目 | 要求 |
| --- | --- |
| 硬件 | 昇腾 Atlas A2/A3 系列产品（`cann-recipes-infer` 官方支持的产品型号） |
| CANN | 9.0.0 |
| PyTorch / torch_npu | PyTorch 2.7.1 / torch_npu 2.7.1.post4 |
| Python | 3.11 |
| 仓库支持权重 | `cann-recipes-infer` 官方验证过 Qwen3-8B、Qwen2.5-7B-Instruct；本文验证的 Qwen3-1.7B 通过复用 Qwen3-8B 配置模板接入，详见「统一执行器配置模版」相关说明 |

### 安装教程层依赖

在以上底层环境（CANN、PyTorch、torch_npu）已就绪的基础上，还需要安装以下 Python 库：

```bash
pip install modelscope transformers accelerate swanlab --break-system-packages
```

### 配置 SwanLab API Key

首次使用需要在 [SwanLab 官网](https://swanlab.cn/) 注册账号，在终端环境中完成登录：

```bash
swanlab login
```

执行后按提示粘贴账号 API Key（在 SwanLab 网页端「用户设置 → API Key」页面获取）。

### 打开 Notebook

```bash
jupyter notebook qwen3_medical_sft.ipynb
```

## 目录说明

| 文件 | 作用 |
| --- | --- |
| `README.md` | 本文档，环境信息与文件导览 |
| `qwen3_medical_sft.ipynb` | 完整代码与分步说明：下载数据 → 加载模型并构建监督标签 → BF16 全参数 SFT → Transformer 原生推理 → cann-recipes-infer 统一执行器部署 |
| `qwen3_medical_sft.yaml` | `cann-recipes-infer` 官方 Qwen3-8B 单卡推理配置模板，接入自定义权重的起点 |
| `prepare_config.py` | 将训练产出的 checkpoint 路径写入运行时 YAML 的 `model_path` 字段 |
| `run_cann_infer.sh` | 从 `models/qwen` 工作目录启动统一执行器，执行推理 |
| `validation.md` | 本次验证的硬件/软件范围、Notebook 完整执行的实测数据、实测遇到并解决的阻塞问题；SwanLab 训练记录与 cann-recipes-infer 推理性能数据见 `qwen3_medical_sft.ipynb` |

以下文件不在本目录内，属于 `cann-recipes-infer` 仓库本身，会在部署过程中被读取或修改：

| 路径（相对 cann-recipes-infer 仓库根目录） | 作用 |
| --- | --- |
| `executor/scripts/set_env.sh` | 环境变量配置脚本，需将 `cann_path` 替换为真实 CANN 安装路径 |
| `executor/scripts/infer.sh` | 统一执行器入口脚本，负责加载模型、执行推理 |
| `models/qwen/config/qwen3_8b_1tp.yaml` | 官方提供的 Qwen3-8B 单卡推理配置模板 |
| `dataset/default_prompt.json` | 默认离线推理测试用的 prompt 文件，验证效果时临时替换为领域问题 |

