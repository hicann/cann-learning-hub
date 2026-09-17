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
reference_practice/yolov13_offline_inference/
├── 01_chapter_intro.ipynb
├── 02_model_prepare.ipynb
├── 03_acl_offline_inference.ipynb
├── 04_result_and_tuning.ipynb
├── 05_chapter_practice.ipynb
├── answers/
│   └── 05_answer.txt
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   ├── preprocess.py
│   └── draw_boxes.py
├── src/
│   ├── CMakeLists.txt
│   ├── acl.json
│   ├── sample_yolov13.cpp
│   └── common/
│       ├── sampleDevice.h
│       ├── sampleModel.h
│       ├── sampleModel.cpp
│       ├── utils.h
│       └── utils.cpp
├── CMakeLists.txt
├── model/          // 模型目录，需手动创建，存放 yolov13.om
└── data/           // 数据目录，需手动创建，存放图片、BIN 与结果文件
```

## 快速运行

```bash
# 1. 构建推理程序（在案例根目录下执行）
bash scripts/build.sh
# 产物：out/yolov13_main

# 2. 运行推理（--model/--input 相对于案例根目录）
bash scripts/run.sh --model=model/yolov13.om --input=data/bus.bin --warmup-runs=5 --runs=10
```

准备图片后，可在图片所在目录执行 `python3 <案例根>/scripts/preprocess.py` 生成输入 BIN；推理完成后在结果文件所在目录执行 `python3 <案例根>/scripts/draw_boxes.py` 生成带检测框的图片。

Notebook 会通过 `npu-smi info -t board` 自动读取 `Chip Name` 和 `NPU Name`，拼接出 `soc_version`。如果环境中有多个 NPU，可在执行前设置 `NPU_ID` 和 `CHIP_ID`；也可以设置 `GITCODE_REPO_ROOT` 指向自定义的仓库目录。ATC 参数说明见[官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/atlasatcparam_16_0036.html)。

## 实施步骤

### 1. 准备 YOLOv13 模型

1. 下载 YOLOv13 权重。

   从 [YOLOv13 Release 页面](https://github.com/iMoonLab/yolov13/releases/tag/yolov13) 下载 `yolov13n.pt`：

   ```
   wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
   ```

   > **说明**：GitHub 下载请使用稳定代理。若下载不稳定，可通过浏览器手动下载。

2. 安装 YOLOv13 依赖。

   YOLOv13 使用自定义模块，标准 `ultralytics` 包无法直接加载，需要从 YOLOv13 源码仓库安装其 fork 版本：

   ```
   curl -L -o yolov13.zip "https://codeload.github.com/iMoonLab/yolov13/zip/refs/heads/main"
   unzip yolov13.zip

   pip3 uninstall ultralytics -y 2>/dev/null
   pip3 install --no-deps yolov13-main/

   pip3 install huggingface_hub seaborn py-cpuinfo
   ```

3. 导出 ONNX 模型。

   ```python
   import torch
   from ultralytics import YOLO

   model = YOLO("yolov13n.pt")
   model.export(format="onnx", opset=11, imgsz=640)
   ```

   生成 `yolov13n.onnx`。

4. 使用 ATC 将 ONNX 模型转换为昇腾离线模型（\*.om）。

   ```
   atc --model=yolov13n.onnx --framework=5 --output=yolov13 \
       --soc_version=Ascend910A \
       --input_shape="images:1,3,640,640" \
       --output_type=FP32
   ```

   将生成的 `yolov13.om` 放入案例的 `model/` 目录。

   **`--soc_version` 参数**：指定推理所用昇腾 AI 处理器型号，取值必须与实际部署硬件匹配。常用查询方式：

   - **Atlas A2 训练/推理系列、Atlas 训练/推理系列**：执行 `npu-smi info`，将 **Name** 字段前加 `Ascend`（例如 Name `910B4` → soc_version `Ascend910B4`）
   - **Atlas A3 训练/推理系列**：执行 `npu-smi info -t board -i <id> -c <chip_id>`，将 **Chip Name** 和 **NPU Name** 拼接为 `Chip Name_NPU Name`（例如 Chip Name `Ascend910B4`、NPU Name `1234` → `Ascend910B4_1234`）。`id` 为设备 ID（通过 `npu-smi info -l` 查询），`chip_id` 为芯片 ID（通过 `npu-smi info -m` 查询）
   - **Ascend 950 系列**：执行 `npu-smi info -t board -i <id>`，将 **Chip Name** 和 **NPU Name** 拼接。`id` 为设备 ID（通过 `npu-smi info -l` 查询）

   ATC 参数详情见 [ATC 参数说明官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/atlasatcparam_16_0036.html)。

### 2. 准备测试图片与预处理

1. 从 YOLOv13 官方仓库下载测试图片并放入 `data/` 目录：

   ```
   cd data/
   wget https://raw.githubusercontent.com/iMoonLab/yolov13/main/ultralytics/assets/bus.jpg
   ```

2. 在图片所在目录执行预处理脚本，将 jpg 转换为 BIN：

   ```
   python3 ../scripts/preprocess.py
   ```

   脚本会扫描当前目录下的 `*.jpg`、`*.jpeg`、`*.png`、`*.bmp`（跳过 `*_out.jpg`），对每张图片执行 letterbox 缩放至 640x640（填充色 114）、BGR→RGB、归一化到 [0,1]、HWC→CHW 并扩展 batch 维，输出 FP32 BIN 文件（shape `[1, 3, 640, 640]`）及 `_info.txt`（记录 `scale`、`pad_left`、`pad_top`、`orig_h`、`orig_w`，供后处理反算坐标）。

   > **注意**：若提示 `ModuleNotFoundError: No module named 'cv2'`，请先安装 OpenCV：
   > - aarch64 平台（如 Atlas 推理卡）推荐安装 `opencv-python-headless`（无 GUI 依赖，避免 libGLdispatch 的 TLS 内存分配冲突）：
   >   ```
   >   pip3 install opencv-python-headless
   >   ```
   > - x86_64 平台或无 TLS 冲突时，也可使用标准 `opencv-python`。

## 编译与运行

### 1. 编译

```
bash scripts/build.sh
```

脚本会检测目标 CPU 架构（x86 主机交叉编译 ARM 目标时使用 `aarch64-linux-gnu-g++`，其余情况使用本地 `g++`），执行 CMake 构建，产物为 `out/yolov13_main`。

### 2. 运行

```
bash scripts/run.sh
```

`run.sh` 将 `--model=`、`--input=` 参数转换为绝对路径后在 `out/` 目录下运行 `yolov13_main`（程序以 `out/` 为工作目录，其内相对路径均相对于 `out/`）。支持以下参数：

| 参数 | 说明 |
|:---|:---|
| `--model=<path>` | OM 模型路径，默认 `../model/yolov13.om` |
| `--input=<path>` | 输入文件路径，默认 `../data/bus.bin` |
| `--warmup-runs=<N>` | 预热推理次数，不计入统计，默认 `0`，可为 0 |
| `--runs=<N>` | 正式推理次数，默认 `1`，必须 ≥ 1 |

例如：

```bash
bash scripts/run.sh --model=model/yolov13.om --input=data/bus.bin --warmup-runs=5 --runs=10
```

### 3. 期望输出

```
[INFO] acl init success
[INFO] set device success
[INFO] create context success
[INFO] create stream success
[INFO] load model ../model/yolov13.om success.
[INFO] start to process file: ../data/bus.bin
[INFO] BENCHMARK aclmdlExecute run=0 latency_us=1526
[INFO] Average aclmdlExecute latency: 1.526 ms
[INFO] detected 5 objects:
[INFO]   [0] bus (0.94)  bbox: [92,136,559,435]
[INFO]   [1] person (0.92)  bbox: [110,236,222,535]
[INFO]   [2] person (0.87)  bbox: [211,241,284,510]
[INFO]   [3] person (0.86)  bbox: [475,232,560,519]
[INFO]   [4] person (0.55)  bbox: [80,330,123,515]
[INFO] recognized persons: 4, vehicles: 1
[INFO] result saved to ../data/bus_result.txt
[INFO] YOLOv13 SAMPLE PASSED.
```

### 4. （可选）可视化检测结果

在结果文件所在目录执行：

```
cd ../data/
python3 ../scripts/draw_boxes.py
```

脚本读取 `<图片名>_result.txt`（每行 `<标签> <置信度> <x1> <y1> <x2> <y2>`），根据 `_info.txt` 中的缩放比例和 padding 反算坐标，在原图上绘制检测框并生成 `<图片名>_out.jpg`。

## 多流参数调优任务

### 功能说明

CANN 社区版 9.2.0 起，GE 提供多流增强调优特性，可自动在图内分配执行流，挖掘算子间的并行执行机会。本案例在静态 Shape 离线编译场景下，通过 ATC 参数 `--multi_stream_parallel_mode` 配置多流并行模式，并为每个候选参数生成独立的 OM。

> **注意：** 多流增强特性需要 CANN 社区版 9.2.0。请前往 [CANN 软件下载页面](https://www.hiascend.com/cann/download?versionId=770&ids=d806%2Ch0501%2Ch0601%2Ch0703)，按以下条件选择安装包：
> - 版本类型：`Weekly`
> - 产品系列：A3 系列
> - CPU 架构：AArch64
> - 操作系统：openEuler
> - 安装方式：离线安装
> 下载后按页面上的安装说明部署环境。

| 取值 | 说明 |
|:---|:---|
| `cv` | 开启 Cube 算子与 Vector 算子的并行执行 |
| `LoadBalance:N` | 负载均衡算法，将所有算子均匀分布在 N 条流上执行 |
| `MainStream:N` | 主流算法，串行算子分布在主流上执行，其他可并行算子分布在其他流上执行 |
| 不配置 | 不启用自动多流并行优化，作为默认性能基线 |

其中 `N` 为正整数，取值范围为 `[1, 64]`。配置的流数量超过实际可用计算资源时，性能可能下降。详细说明请参见[`--multi_stream_parallel_mode` 参数说明官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/docs/zh/user_guides/atc_tools/CLI_options/--multi_stream_parallel_mode.md)。

### 调优任务

在 CANN 社区版 9.2.0 及以上环境中，按第 4 章 Notebook 的操作编译各候选参数对应的 OM（例如 `atc --model=yolov13n.onnx --framework=5 --output=./model/yolov13_cv --soc_version=<实际型号> --input_shape="images:1,3,640,640" --output_type=FP32 --multi_stream_parallel_mode=cv`），并使用 `--warmup-runs=10 --runs=90` 运行 100 次推理（前 10 次预热不计入统计），以 90 次成功正式推理的平均 `aclmdlExecute` 时延作为评价指标，平均时延最小者即为当前测试环境下的推荐参数。确定最优参数后，以如下格式提交 PR：

```text
--multi_stream_parallel_mode=LoadBalance:55
```

调优过程中需保持模型结构、输入 shape、`soc_version`、预热次数和正式推理次数一致，`--multi_stream_parallel_mode` 为唯一变量。

## 附注

- 检测结果使用 COCO 数据集 80 类标签（代码内置 `kCocoLabels`）。
- 需要处理更多图片时，可在 `src/sample_yolov13.cpp` 的 `main()` 函数中扩展 `inputFiles` 列表。
- 置信度阈值（`kConfThresh`，默认 0.25）和 NMS IoU 阈值（`kIouThresh`，默认 0.45）可在 `src/sample_yolov13.cpp` 中调整。
