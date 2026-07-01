# Torch-RecHub 推荐系统实战教程

本目录位于 `contrib/tutorials/torch-rechub`，收录基于 Torch-RecHub 推荐系统教程整理的 NPU 实战 notebook，覆盖 CTR 精排、序列兴趣建模、召回、多任务学习、实验跟踪、模型导出与推理验证等常见推荐系统链路。

源教程地址：[Torch-RecHub tutorials](https://github.com/datawhalechina/torch-rechub/tree/main/tutorials)

## 迁移说明

适配的源仓库为 [datawhalechina/torch-rechub](https://github.com/datawhalechina/torch-rechub)。本目录中的 notebook 保持源教程的代码组织和运行逻辑，仅做以下适配：

- 将教程纳入 `cann-learning-hub` 的外部贡献教程目录：`contrib/tutorials/torch-rechub`。
- 在 notebook 开头补充迁移说明，标明源教程地址和当前目录位置。
- 将原 notebook 中的 attachment 图片改为仓库内相对路径图片，图片文件统一放在 `images/` 目录，便于在代码托管平台和本地 Markdown 预览中显示。
- 新增本 README，汇总教程列表、运行依赖和迁移说明。

## 目录说明

```text
contrib/tutorials/torch-rechub
├── images/
│   ├── deepFM.png
│   ├── DIN.png
│   ├── DSSM.png
│   └── MMoE.png
├── 00_QuickStart_CTR_DeepFM.ipynb
├── 01_Ranking_DIN.ipynb
├── 02_Matching_DSSM.ipynb
├── 03_MultiTask_MMOE.ipynb
├── 04_Experiment_Tracking_Light.ipynb
├── 05_Model_Export_and_Serving.ipynb
└── README.md
```

## Notebook 列表

| 序号 | Notebook | 内容简介 | Link | 状态 |
|---|---|---|---|---|
| 00 | [QuickStart：CTR 预测（DeepFM）](./00_QuickStart_CTR_DeepFM.ipynb) | 跑通 `DataFrame -> Feature -> DataGenerator -> DeepFM -> CTRTrainer -> AUC` 的完整 CTR 训练链路 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb) |已迁移 |
| 01 | [序列兴趣建模：DIN](./01_Ranking_DIN.ipynb) | 使用 Amazon-Electronics 样例数据理解历史行为序列、`SequenceFeature` 与 DIN attention | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb) |已迁移 |
| 02 | [匹配/召回：DSSM + Annoy](./02_Matching_DSSM.ipynb) | 基于 MovieLens-1M 样例数据跑通双塔召回与向量 Top-K 检索 |[在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb) | 已迁移 |
| 03 | [多任务学习：MMOE](./03_MultiTask_MMOE.ipynb) | 使用 Ali-CCP 样例数据演示多目标建模、expert、gate 与 tower | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/03_MultiTask_MMOE.ipynb) |已迁移 |
| 04 | [实验跟踪：model_logger](./04_Experiment_Tracking_Light.ipynb) | 演示 WandB / SwanLab / TensorBoardX 等轻量实验跟踪接入方式 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/04_Experiment_Tracking_Light.ipynb) |已迁移 |
| 05 | [模型导出与推理验证：ONNX](./05_Model_Export_and_Serving.ipynb) | 演示 DeepFM 与 DSSM 的 ONNX 导出、ONNXRuntime 推理验证和量化入口 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/05_Model_Export_and_Serving.ipynb) |已迁移 |

## 环境要求

- Python 3.10+
- PyTorch 与 `torch_npu`
- `torch-rechub`
- notebook 中涉及的可选依赖，例如 `onnxruntime`、`wandb`、`swanlab`、`tensorboardX`、`annoy`

运行前请参考源仓库的安装和数据准备说明：[Torch-RecHub](https://github.com/datawhalechina/torch-rechub)。
