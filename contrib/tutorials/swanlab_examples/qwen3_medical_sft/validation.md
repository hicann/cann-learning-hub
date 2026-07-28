# Validation

## 1. 环境以及其实测值

| 项目 | 实测值 |
| --- | --- |
| 硬件 | 昇腾 Atlas A3（910C），单卡，64GB HBM |
| CANN | 9.0.0 |
| torch_npu | 2.7.1.post4 |
| PyTorch | 2.7.1 |
| transformers | 5.14.1 |
| Python | 3.11 |
| modelscope | 1.22.0（实测环境安装版本，未强制锁定） |
| swanlab | 0.9.0 |

## 2. 完整 Notebook 实测以及结果数据

依据 `qwen3_medical_sft.ipynb` 完整执行，实测数据如下：

| 阶段 | 指标 | 实测值 |
| --- | --- | --- |
| 数据 | 总样本数 | 2407 条（训练集 2166 / 验证集 241，9:1 切分） |
| 训练 | 训练步数 | ≤680 步（`max_steps=680` 上限，约合 5 个 epoch，配合 `EarlyStoppingCallback` 提前停止，实测收敛于约 650 步附近） |
| 训练 | train/loss（首个 logging step，约） | 1.93 |
| 训练 | train/loss（训练结束时，约） | 1.07 |
| 训练 | eval/loss（训练结束时，约） | 1.16 |
| Transformer 原生推理 | 输出格式 | 正确包含 `<think>...</think>` 思考过程及最终回答 |

> 上表训练相关数值取自 SwanLab 训练曲线读数（见 `qwen3_medical_sft.ipynb` 中「SwanLab 云端记录」小节的 `eval/loss`、`train/loss` 截图），为近似值。

## 3. 实测修正

| 问题 | 现象 | 修正方式 |
| --- | --- | --- |
| modelscope 下载数据集接口用错 | `snapshot_download` 下载数据集报 404 | 改用 `git clone` 方式获取数据集 |
| `set_env.sh` 中 `cann_path` 为占位符 | 执行统一执行器报错找不到 `setenv.bash` | 手动替换为真实 CANN 安装路径 |
| checkpoint 缺失 `lm_head.weight` | 接入推理后输出为重复乱码字符 | 手动补充 `lm_head.weight`（复用 `embed_tokens` 权重，见 `qwen3_medical_sft.ipynb` 第 5.3 节） |
| `model_name` 与实际模型规模不一致 | `qwen3_medical_sft.yaml` 中 `model_name` 为 `qwen3_8b`，实际权重为 1.7B | 保留原值，未修改：`cann-recipes-infer` 官方仅为 `qwen3_8b`、`qwen25_7b_instruct` 提供固定的 `model_name` 取值，未见支持自定义规模标签的依据；实测该字段未影响本次加载与推理结果 |
| `trust_remote_code=True` 触发隐藏联网校验，卡死不报错 | 加载 Qwen3-1.7B 模型时（notebook 第 2 部分）进程长时间无响应，`npu-smi info` 显示 HBM 占用长期不变，`top` 显示进程 CPU 占用接近 0%（sleeping 状态），怀疑是联网请求超时挂起，而非真实计算 | 在加载模型代码前加入环境变量强制离线模式，跳过网络校验：`os.environ["HF_HUB_OFFLINE"] = "1"`、`os.environ["TRANSFORMERS_OFFLINE"] = "1"`。该问题是否出现取决于运行环境能否访问 huggingface.co：网络受限（如仅放行 modelscope.cn 等国内域名）的环境下会必现，网络开放的环境下可能感知不到 |
| `cann-recipes-infer/models/qwen/requirements.txt` 锁定 `torch==2.8.0` | 执行 `pip install -r requirements.txt` 后覆盖了预装的 `torch 2.7.1+cpu`，导致 `import torch_npu` 报 `undefined symbol` ABI 不兼容错误 | 执行完依赖安装后需验证 `python3 -c "import torch, torch_npu"` 是否仍能正常导入且版本为 `2.7.1+cpu` / `2.7.1.post4`；若被覆盖，执行 `pip uninstall torch -y` 卸载多装的版本，回退到镜像预装的 `torch 2.7.1+cpu` |
| `torch_npu._C._get_cann_version` 内部读取到非 UTF-8 编码内容 | `source set_env.sh` 设置 `ASCEND_HOME_PATH` 后，`import torch_npu` 触发版本校验逻辑，报 `UnicodeDecodeError: 'utf-8' codec can't decode byte ...`，导致 `RuntimeError: Failed to load the backend extension: torch_npu` | 该函数仅用于版本兼容性提示，非核心推理逻辑必需。对 `torch_npu/npu/utils.py` 中 `get_cann_version` 函数打补丁，用 `try/except UnicodeDecodeError` 包裹调用并返回空字符串兜底（该文件路径通常需要 `sudo` 权限修改） |
| `set_env.sh` 中 `cann_path` 变量身兼两职冲突 | 该变量既用于 `source $cann_path/bin/setenv.bash`（需要带 `aarch64-linux` 层级的路径，因为部分 CANN 安装包中 `cann-9.0.0/bin/setenv.bash` 是指向不存在文件的断链接），又被强制赋值给 `ASCEND_HOME_PATH`（该变量语义上应指向不带 `aarch64-linux` 的上一级目录，`runtime`/`compiler` 等子目录实际存在于该层）。两种用途路径层级要求不一致，导致 `torch_npu` 的 `_cann_package_check` 报 `ASCEND_RUNTIME_PATH` 目录不存在 | 删除 `set_env.sh` 中 `source $cann_path/bin/setenv.bash` 之后那行 `export ASCEND_HOME_PATH=$cann_path` 的强制覆盖，让 `setenv.bash` 脚本内部基于自身路径正确推导出的 `ASCEND_HOME_PATH` 值生效，不再被覆盖 |

> 以上后三条问题均出现在同一次环境搭建过程中，且具有一定的环境相关性（与所用 CANN 安装包的目录结构、镜像预装的 Python/torch 版本组合有关）；其他复现者若使用的 CANN 安装方式与本文实测环境（云端 Ascend Atlas A3 NPU 开发环境，镜像标识 `cann_9.0.0-py3.11-A3-arm`）不同，可能不会触发全部问题，但排查思路可参考本节。
