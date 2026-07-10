# Qwen3-8B 推理优化实践
本教程以 `Qwen3-8B` 为例，展示如何在昇腾 NPU 上使用 cann-recipes-infer 离线推理框架完成 Baseline 推理、Profiling 分析，并验证 Dense RMSNorm NPU 融合路径的性能收益。课程主流程采用 recipes 的典型用法：查看并修改 `models/qwen/config/` 下的 YAML 配置，通过 `executor/scripts/infer.sh` 拉起离线推理，再从 `res/` 与 `prof/` 目录读取日志和性能产物。

教程包含以下内容：
- Notebooks：包含环境准备、YAML 修改、`infer.sh` 启动、Profiling 分析和 Dense RMSNorm NPU 融合路径 A/B 验证步骤，可在 GitCode 提供的轻量级 Notebook 上运行，也可在本地 Jupyter 环境中执行。
- SRC：包含教程中使用的 recipes 推理入口、Qwen3-8B 模型适配代码、NPU 适配辅助函数、运行后处理工具与样例 prompt。

## 关于 cann-recipes-infer

[cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 是 CANN 推理优化样例仓，面向 LLM 与多模态模型推理业务中的典型模型和加速算法，提供基于 CANN 平台的优化样例，帮助开发者快速完成 NPU 推理部署与性能验证。仓库覆盖 Qwen、DeepSeek、Kimi、LongCat、Hunyuan、SANA-Video、Wan、GPT-OSS、HSTU 等多类模型样例，既包含文本大模型，也包含图像、视频和推荐类推理实践。

仓库以 `executor/` 提供统一推理执行框架，包含离线推理入口、在线推理服务、模型加载、配置解析、调度执行和通用工具模块；以 `models/` 提供不同模型的脚本与 YAML 配置；以 `module/` 和 `ops/` 提供基础层、量化模块以及 AscendC、PyPTO、TileLang 等算子样例。不同模型样例围绕昇腾 NPU 推理常见优化点展开，包括融合算子、图模式编译、Packed Sequence、Page Attention、TP/EP/DP/CP 等并行策略、多流并行、KV Cache 管理、W8A8/W8A8C8/混合量化，以及部分模型中的 MTP 投机推理能力。

本课程采用该仓库的 recipes 工作流与 Qwen3-8B 单卡 BF16 推理所需代码子集，保留 YAML 配置、`executor/scripts/infer.sh` 启动方式、离线推理、Profiling 和 Dense RMSNorm NPU 融合验证链路。学习本课程后，可以继续到完整 `cann-recipes-infer` 仓库中查看更多模型和更复杂的部署优化实践。

>- **注意：**
>- 在线体验请直接在 GitCode Notebook 环境中执行。Notebook 环境需包含 CANN、`torch`、`torch_npu`、`transformers`、`modelscope`、`datasets` 和 recipes 推理所需依赖；本地运行前请先执行 `/usr/local/Ascend/ascend-toolkit/set_env.sh` 或等价 CANN 环境脚本。
>- `transformers` 版本需支持 Qwen3 模型结构，建议使用 4.51.0 或更高版本。
>- `Qwen3-8B` BF16 权重约 16GB，短上下文单卡验证建议使用 64GB HBM NPU，磁盘空间建议至少 40GB。教程默认关闭 thinking 模式并使用短输出，降低在线环境资源压力。
>- 模型权重使用 `Qwen/Qwen3-8B`。首次运行会通过 ModelScope 下载并缓存权重；如环境中已准备本地权重，可设置 `QWEN3_8B_MODEL_PATH=/path/to/Qwen3-8B`。

## 模型权重

- 模型标识：`Qwen/Qwen3-8B`。
- 下载通道：Notebook 默认使用 ModelScope。
- 本地权重：已准备模型目录时，设置 `QWEN3_8B_MODEL_PATH=/path/to/Qwen3-8B` 后会直接加载本地权重。
- 下载时机：首次执行 `02_baseline_inference.ipynb` 的 Baseline 推理 cell 时开始下载权重；已存在缓存或本地权重时会直接加载。
- 异常排查：如果下载阶段网络不稳定，可重新执行当前 cell；已完成的缓存会被复用。

## 端到端复现

Notebook 环境中按顺序打开并 Run All：

1. `01_chapter_intro.ipynb`
2. `02_baseline_inference.ipynb`
3. `03_profiling_analysis.ipynb`
4. `04_npu_optimization.ipynb`

首次执行 `02_baseline_inference.ipynb` 的 Baseline 推理 cell 会开始下载并缓存 `Qwen/Qwen3-8B` 权重；如果已设置 `QWEN3_8B_MODEL_PATH`，则直接使用本地权重目录。

`02/03/04` 的环境准备单元会定位仓库目录、创建 `Sources/model_inference_optimization/qwen3_8b` 运行目录、导入 CANN 环境，并打印 recipes 目录、YAML 目录、推理入口和 NPU 可见卡号。手工操作时，进入 `src/inference_scripts/recipe_qwen3_8b/models/qwen/config/` 修改 YAML；Notebook 中会把对应 YAML 复制到 `Sources/.../recipe_yaml/`，只填入当前环境可用的模型路径。启动推理时会显式展示并执行：

```bash
cd src/inference_scripts/recipe_qwen3_8b
bash executor/scripts/infer.sh --model qwen --mode offline --yaml <yaml>
```

recipes 日志保存在 `src/inference_scripts/recipe_qwen3_8b/models/qwen/res/<日期>/<case_name>/`，课程指标会同步写入 `Sources/model_inference_optimization/qwen3_8b/run_outputs/`。

第 3 章会在 YAML 中打开 `model_config.enable_profiler=true`。运行后进入 recipes 结果目录下的 `prof/`，查看 `kernel_details`、trace、`op_statistic` 或 `op_summary` 等性能产物，再基于真实算子耗时选择第 4 章的优化点。

本地终端运行前先准备 CANN 和可见 NPU。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
```

如需使用本地权重：

```bash
export QWEN3_8B_MODEL_PATH=/path/to/Qwen3-8B
```

## 代码与权重说明

- 本实践的 Notebook、推理脚本和 Qwen3 NPU 适配辅助代码位于本目录 `src/`。
- recipes 推理框架子集位于 `src/inference_scripts/recipe_qwen3_8b/`，保留 Qwen3-8B 单卡离线推理所需的 executor、model loader、Qwen 模型文件与公共线性层代码。
- recipes YAML 位于 `src/inference_scripts/recipe_qwen3_8b/models/qwen/config/`：`qwen3_8b_1tp.yaml` 用于 Baseline，`qwen3_8b_1tp_profile.yaml` 用于 Profiling，`qwen3_8b_1tp_add_rmsnorm.yaml` 用于 Dense RMSNorm NPU 融合验证。
- Qwen dense 模型实现位于 `src/inference_scripts/recipe_qwen3_8b/models/qwen/models/modeling_qwen.py`，基于 Qwen2/Qwen3 dense 结构适配 NPU 推理。
- Qwen3 NPU norm 精度检查位于 `src/qwen3_npu_adaptation.py`；第 4 章 A/B 只切换 recipes Qwen3-8B 路径中是否启用 `torch_npu.npu_rms_norm` 与 `torch_npu.npu_add_rms_norm` 融合。
- 模型权重使用 `Qwen/Qwen3-8B`；首次运行会通过默认下载通道获取权重，已准备本地权重时可通过 `QWEN3_8B_MODEL_PATH` 指定。

## Notebooks

| Notebook | Link | 状态 |
|--|--|--|
| 1. 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/01_chapter_intro.ipynb) | ✅ 已发布 |
| 2. Baseline 跑通 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/02_baseline_inference.ipynb) | ✅ 已发布 |
| 3. Profiling 分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/03_profiling_analysis.ipynb) | ✅ 已发布 |
| 4. Dense RMSNorm NPU 融合路径优化验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/04_npu_optimization.ipynb) | ✅ 已发布 |
| 5. 量化Qwen3-8B模型 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/05_quantization_qwen3_8b.ipynb) | ✅ 已发布 |
| 6. 自定义量化 A8W8 matmul 算子开发并接入 Qwen3-8B | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/06_custom_matmul_operator_development_and_integration_with_qwen3_8b.ipynb) | ✅ 已发布 |
