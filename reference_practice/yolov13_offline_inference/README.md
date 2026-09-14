# YOLOv13 离线推理应用案例

本案例以 YOLOv13 目标检测模型为例，展示在昇腾 AI 处理器上完成 ONNX 模型导出、ATC 离线编译、ACL 推理和结果后处理的完整流程，并通过 `aclmdlExecute` 时延统计完成多流参数调优。

教程包含以下内容：
- Notebooks：包含环境准备、模型转换、离线推理、结果可视化和性能调优步骤，可在 GitCode Notebook 或本地 Jupyter 环境中学习。
- `src`：包含 ACL C++ 推理程序及其构建依赖。
- `scripts`：包含图片预处理、结果可视化、构建和运行脚本。
- `answers`：包含章节练习参考答案。

> **注意：**
> - 本案例只提供代码和课程资料，不包含模型权重、OM 模型和测试图片。请按 Notebook 或下方步骤自行下载。
> - 本案例需要已安装 CANN Toolkit、OPP 算子包和可用的昇腾设备。请先设置 `ASCEND_HOME_PATH`，再执行 `source "$ASCEND_HOME_PATH/set_env.sh"`；如果 CANNLab 使用非默认仓库目录，还可以设置 `GITCODE_REPO_ROOT`。
> - 示例验证以静态 Shape `1x3x640x640` 为例，`soc_version` 必须替换为实际设备型号。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 系列产品（已验证） |
| CANN 版本 | 第 1、2、3、5 章和第 4 章结果可视化：9.0.0 及以上；第 4 章多流候选 OM 编译与性能调优：9.2.0 及以上 |
| Python | 3.11 |
| 运行环境 | Linux，已安装 CANN Toolkit 和 OPP 算子包；第 2～4 章需要可用的昇腾 NPU |

第 4 章的图片后处理可以在 CANN 9.0.0 及以上版本运行。`--multi_stream_parallel_mode` 属于 CANN 9.2.0 及以上版本支持的能力；低于该版本时，Notebook 会打印不支持提示并跳过多流编译和性能测试。

## 在线体验环境

本案例支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | 平台预置环境 | Python 3.11 | 可直接打开 Notebook 在线学习和运行；第 4 章多流部分需要环境中的 CANN 版本不低于 9.2.0 |
| CANNLab 云开发环境 | `cann_9.0.0 py3.11-A2-arm` 或更高版本 | Python 3.11 | 参考 [CANNLab 环境体验指南](../../docs/CANNLab_env_experience_guide.md) 创建环境；运行第 4 章多流部分时，请使用 CANN 9.2.0 及以上版本 |

## Notebooks

| Notebook | 内容 |
|--|--|
| [1. 章节介绍](./01_chapter_intro.ipynb) | 学习目标、流程和目录说明 |
| [2. 环境准备与模型转换](./02_model_prepare.ipynb) | 准备 YOLOv13、导出 ONNX 并使用 ATC 生成 OM |
| [3. ACL 离线推理](./03_acl_offline_inference.ipynb) | 编译并运行 ACL C++ 推理程序 |
| [4. 结果处理与性能调优](./04_result_and_tuning.ipynb) | 后处理、可视化和多流参数调优 |
| [5. 章节练习](./05_chapter_practice.ipynb) | 巩固模型转换、推理和调优知识 |

## 目录结构

```text
├── 01_chapter_intro.ipynb
├── 02_model_prepare.ipynb
├── 03_acl_offline_inference.ipynb
├── 04_result_and_tuning.ipynb
├── 05_chapter_practice.ipynb
├── answers/05_answer.txt
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   ├── preprocess.py
│   └── draw_boxes.py
└── src/
    ├── CMakeLists.txt
    ├── acl.json
    ├── sample_yolov13.cpp
    └── common/
```

## 快速运行

```bash
cd scripts
bash build.sh
bash run.sh --model=../model/yolov13.om --input=../data/bus.bin
```

准备图片后，可执行 `python3 scripts/preprocess.py` 生成输入 BIN；推理完成后执行 `python3 scripts/draw_boxes.py` 生成带检测框的图片。

Notebook 会通过 `npu-smi info -t board` 自动读取 `Chip Name` 和 `NPU Name`，拼接出 `soc_version`。如果环境中有多个 NPU，可在执行前设置 `NPU_ID` 和 `CHIP_ID`；也可以设置 `GITCODE_REPO_ROOT` 指向自定义的仓库目录。ATC 参数说明见[官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/latest/devaids/atctool/atlasatcparam_16_0036.html)。
