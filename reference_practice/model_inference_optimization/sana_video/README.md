# Sana-Video 推理优化实践
本教程以 `Sana-Video` 为例，展示如何在昇腾 NPU 上完成模型跑通、Profiling 分析，并在整网中接入 `RMSNorm` 融合算子验证性能收益。

教程包含以下内容：
- Notebooks：包含环境准备、Baseline 跑通、Profiling 分析和整网优化验证步骤，可在 gitcode 提供的轻量级 notebook 上运行，也可在本地 Jupyter 环境中执行。
- SRC：包含教程中使用的最小兼容适配代码、推理入口与样例数据。

>- **注意：**
>- 本教程依赖上游 `Sana` 仓库代码，Notebook 会在运行目录中拉取指定 commit 并覆盖最小 NPU 适配文件。
>- 在线体验请直接在 GitCode Notebook 环境中执行；Notebook 默认复用环境中已预装的 `torch`、`torch_npu`、`torchvision` 与 `torchaudio`。本地运行前请先准备兼容版本的上述依赖，并配置 CANN 与 `torch_npu` 环境。
>- 运行本教程时，主机内存需至少 16GB。Notebook 计算类型建议选择 NPU 910B、CPU 32GB，容器镜像建议选择 ubuntu22-cann8.5-py3.11-jupyter-notebook。
## Notebooks

| Notebook | Link | 状态 |
|--|--|--|
| 1. 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=reference_practice/model_inference_optimization/sana_video&scanFilePath=reference_practice/model_inference_optimization/sana_video/01_chapter_intro.ipynb) | ✅ 已发布 |
| 2. Baseline 跑通 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=reference_practice/model_inference_optimization/sana_video&scanFilePath=reference_practice/model_inference_optimization/sana_video/02_baseline_run.ipynb) | ✅ 已发布 |
| 3. Profiling 分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=reference_practice/model_inference_optimization/sana_video&scanFilePath=reference_practice/model_inference_optimization/sana_video/03_profiling_analysis.ipynb) | ✅ 已发布 |
| 4. RMSNorm 融合接入与收益验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=reference_practice/model_inference_optimization/sana_video&scanFilePath=reference_practice/model_inference_optimization/sana_video/04_rmsnorm_fusion_optimization.ipynb) | ✅ 已发布 |
