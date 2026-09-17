# YOLOv13 Offline Inference Sample

This sample demonstrates the complete workflow of YOLOv13 object detection on an Ascend AI processor: ONNX model export, ATC offline compilation, ACL inference, and result post-processing. It also covers multi-stream parameter tuning based on `aclmdlExecute` latency statistics.

The tutorial contains the following:
- Notebooks: covering environment preparation, model conversion, offline inference, result visualization, and performance tuning. They can be run in the GitCode Notebook or a local Jupyter environment.
- `src`: the ACL C++ inference program and its build dependencies.
- `scripts`: image preprocessing, result visualization, build, and run scripts.
- `answers`: reference answers for the chapter exercises.

> **Notes:**
> - This sample only provides code and course materials. It does not include model weights, OM models, or test images. Please download them yourself following the Notebook or the steps below.
> - CANN Toolkit, the OPP operator package, and a working Ascend device are required. Set `ASCEND_HOME_PATH` first, then run `source "$ASCEND_HOME_PATH/set_env.sh"`. Set `GITCODE_REPO_ROOT` if the repository is mounted at a non-default path in CANNLab.
> - The sample is validated with static shape `1x3x640x640`; replace `soc_version` with your actual device model.

## Software and Hardware Compatibility

| Item | Requirement |
|--|--|
| Supported hardware | Atlas A2 series products (validated) |
| CANN version | Chapters 1, 2, 3, 5 and result visualization in Chapter 4: 9.0.0 or later; multi-stream OM compilation and performance tuning in Chapter 4: 9.2.0 or later |
| Python | 3.11 |
| Runtime | Linux with CANN Toolkit and OPP installed; Chapters 2 to 4 require an Ascend NPU |

Result visualization in Chapter 4 works with CANN 9.0.0 or later. The `--multi_stream_parallel_mode` option requires CANN 9.2.0 or later. On older versions, the Notebook prints an unsupported message and skips multi-stream compilation and performance measurement.

## Online Experience Environments

This sample supports both of the following environments:

| Environment | Image / version | Python kernel | Notes |
|--|--|--|--|
| cann-learning-hub online Notebook | Platform-provided environment | Python 3.11 | Open the corresponding Notebook in the repository to read and run it. Chapter 4 multi-stream cells require CANN 9.2.0 or later. |
| CANNLab cloud development environment | `cann_9.0.0 py3.11-A2-arm` or later | Python 3.11 | Follow the [CANNLab environment guide](../../docs/CANNLab_env_experience_guide.md). Use CANN 9.2.0 or later for Chapter 4 multi-stream cells. |

## Notebooks

| Notebook | Content |
|--|--|
| [1. Chapter introduction](./01_chapter_intro.ipynb) | Learning objectives and workflow |
| [2. Model preparation and conversion](./02_model_prepare.ipynb) | Prepare YOLOv13, export ONNX, and generate OM with ATC |
| [3. ACL offline inference](./03_acl_offline_inference.ipynb) | Build and run the ACL C++ inference program |
| [4. Result processing and performance tuning](./04_result_and_tuning.ipynb) | Post-processing, visualization, and multi-stream tuning |
| [5. Chapter practice](./05_chapter_practice.ipynb) | Exercises for model conversion, inference, and tuning |

## Directory Structure

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
├── model/          // Model directory, create manually, stores yolov13.om
└── data/           // Data directory, create manually, stores images, BIN files, and results
```

## Quick Start

```bash
# 1. Build the inference program (run from the sample root directory)
bash scripts/build.sh
# Artifact: out/yolov13_main

# 2. Run inference (--model/--input are relative to the sample root directory)
bash scripts/run.sh --model=model/yolov13.om --input=data/bus.bin --warmup-runs=5 --runs=10
```

After preparing images, run `python3 <sample_root>/scripts/preprocess.py` in the image directory to generate input BIN files; after inference, run `python3 <sample_root>/scripts/draw_boxes.py` in the directory containing the result files to generate images with detection boxes.

The Notebook automatically reads `Chip Name` and `NPU Name` via `npu-smi info -t board` and combines them into `soc_version`. On multi-NPU environments, set `NPU_ID` and `CHIP_ID` before execution; set `GITCODE_REPO_ROOT` to point to a custom repository directory. See the [ATC parameter documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/atlasatcparam_16_0036.html) for details.

## Implementation Steps

### 1. Prepare the YOLOv13 model

1. Download the YOLOv13 weights.

   Download `yolov13n.pt` from the [YOLOv13 Release page](https://github.com/iMoonLab/yolov13/releases/tag/yolov13):

   ```
   wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
   ```

   > **Note**: Please use a stable proxy for GitHub downloads. If the download is unstable, you can manually download via browser from the link above.

2. Install YOLOv13 dependencies.

   YOLOv13 uses custom modules that the standard `ultralytics` package cannot load directly. You need to install the forked version of ultralytics from the YOLOv13 source repository:

   ```
   # Download the YOLOv13 source repository (includes forked ultralytics)
   curl -L -o yolov13.zip "https://codeload.github.com/iMoonLab/yolov13/zip/refs/heads/main"
   unzip yolov13.zip

   # Install forked ultralytics (uninstall standard version first if present)
   pip3 uninstall ultralytics -y 2>/dev/null
   pip3 install --no-deps yolov13-main/

   # Install missing dependencies
   pip3 install huggingface_hub seaborn py-cpuinfo
   ```

3. Export to ONNX.

   ```python
   import torch
   from ultralytics import YOLO

   model = YOLO("yolov13n.pt")
   model.export(format="onnx", opset=11, imgsz=640)
   ```

   This generates `yolov13n.onnx`.

4. Convert the ONNX model to an Ascend AI processor offline model (\*.om) with ATC.

   ```
   atc --model=yolov13n.onnx --framework=5 --output=yolov13 \
       --soc_version=Ascend910A \
       --input_shape="images:1,3,640,640" \
       --output_type=FP32
   ```

   Place the generated `yolov13.om` file in the `model/` directory.

   **`--soc_version` parameter**: Specifies the Ascend AI processor model used for inference. The value must match the deployment hardware. Common ways to check:

   - **Atlas A2 training/inference series, Atlas training/inference series**: Run `npu-smi info`, prepend `Ascend` to the **Name** field (e.g., Name `910B4` → soc_version `Ascend910B4`)
   - **Atlas A3 training/inference series**: Run `npu-smi info -t board -i <id> -c <chip_id>`, combine **Chip Name** and **NPU Name** as `Chip Name_NPU Name` (e.g., Chip Name `Ascend910B4`, NPU Name `1234` → `Ascend910B4_1234`). `id` is the device ID (query via `npu-smi info -l`), `chip_id` is the chip ID (query via `npu-smi info -m`)
   - **Ascend 950 series**: Run `npu-smi info -t board -i <id>`, combine **Chip Name** and **NPU Name**. `id` is the device ID (query via `npu-smi info -l`)

   See the [ATC parameter documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/atlasatcparam_16_0036.html) for details.

### 2. Prepare the test image and preprocess

1. Download the test image from the YOLOv13 official repository and place it in the `data/` directory:

   ```
   cd data/
   wget https://raw.githubusercontent.com/iMoonLab/yolov13/main/ultralytics/assets/bus.jpg
   ```

2. Run the preprocessing script in the image directory to convert jpg to bin:

   ```
   python3 ../scripts/preprocess.py
   ```

   The script scans `*.jpg`, `*.jpeg`, `*.png`, `*.bmp` files in the current directory (skipping `*_out.jpg`). For each image it performs letterbox resizing to 640x640 (pad color 114), BGR→RGB conversion, normalization to [0,1], HWC→CHW transposition, and batch-dimension expansion, then outputs an FP32 BIN file (shape `[1, 3, 640, 640]`) and an `_info.txt` file recording `scale`, `pad_left`, `pad_top`, `orig_h`, and `orig_w` for coordinate back-projection in post-processing.

   > **Note**: If you encounter `ModuleNotFoundError: No module named 'cv2'`, install OpenCV first:
   > - On aarch64 platforms (e.g., Atlas inference cards), install `opencv-python-headless` (no GUI dependency, avoids libGLdispatch TLS memory allocation conflicts):
   >   ```
   >   pip3 install opencv-python-headless
   >   ```
   > - On x86_64 platforms or when no TLS conflict exists, standard `opencv-python` can also be used.

## Build and Run

### 1. Build

```
bash scripts/build.sh
```

The script detects the target CPU architecture (uses `aarch64-linux-gnu-g++` for cross-compiling to ARM from an x86 host, otherwise the local `g++`), runs the CMake build, and produces `out/yolov13_main`.

### 2. Run

```
bash scripts/run.sh
```

`run.sh` converts `--model=` and `--input=` to absolute paths and runs `yolov13_main` in the `out/` directory (the program uses `out/` as its working directory, and its internal relative paths are relative to `out/`). Supported arguments:

| Argument | Description |
|:---|:---|
| `--model=<path>` | OM model path. The default is `../model/yolov13.om` |
| `--input=<path>` | Input file path. The default is `../data/bus.bin` |
| `--warmup-runs=<N>` | Number of warmup runs, not counted in statistics. The default is `0`; zero is allowed |
| `--runs=<N>` | Number of measured runs. The default is `1`; must be ≥ 1 |

For example:

```bash
bash scripts/run.sh --model=model/yolov13.om --input=data/bus.bin --warmup-runs=5 --runs=10
```

### 3. Expected output

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

### 4. (Optional) Visualize detection results

Run in the directory containing the result files:

```
cd ../data/
python3 ../scripts/draw_boxes.py
```

The script reads `<image>_result.txt` (each line: `<label> <confidence> <x1> <y1> <x2> <y2>`), back-projects the coordinates using the scale and padding from `_info.txt`, draws detection boxes on the original image, and generates `<image>_out.jpg`.

## Multi-stream Parameter Tuning Task

### Feature Description

In CANN Community Edition 9.2.0, the GE advanced tuning feature Multi-Stream Enhancement can automatically allocate execution streams within a graph to expose opportunities for parallel execution among operators. In the Static Shape offline compilation scenario, this sample configures the multi-stream parallel mode through the ATC option `--multi_stream_parallel_mode` and generates a separate OM for each candidate value.

> [!NOTE]
> Multi-Stream Enhancement requires CANN Community Edition 9.2.0. Go to the [CANN Software Download page](https://www.hiascend.com/cann/download?versionId=770&ids=d806%2Ch0501%2Ch0601%2Ch0703) and select the installation package using the following criteria:
>
> - Version type: `Weekly`
> - Product series: A3 series
> - CPU architecture: AArch64
> - Operating system: openEuler
> - Installation method: Offline installation
>
> After downloading the package, follow the installation instructions on the page to deploy the environment.

| Value | Description |
|:---|:---|
| `cv` | Enables parallel execution of Cube and Vector operators |
| `LoadBalance:N` | Uses the load-balancing algorithm to distribute operators across at most `N` streams |
| `MainStream:N` | Uses the main-stream algorithm, where serial operators run on the main stream and other parallelizable operators are distributed across other streams |
| Not configured | Disables automatic multi-stream parallelism and serves as the default performance baseline |

`N` is a positive integer in the range `[1, 64]`. Performance may degrade if the configured stream count exceeds the available compute resources. For details, see the [`--multi_stream_parallel_mode` parameter description](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/docs/zh/user_guides/atc_tools/CLI_options/--multi_stream_parallel_mode.md).

### Tuning Task

In CANN Community Edition 9.2.0 or later, follow Chapter 4 of the Notebook to compile an OM for each candidate value (for example, `atc --model=yolov13n.onnx --framework=5 --output=./model/yolov13_cv --soc_version=<detected SOC_VERSION> --input_shape="images:1,3,640,640" --output_type=FP32 --multi_stream_parallel_mode=cv`). Then run 100 inferences per candidate using `--warmup-runs=10 --runs=90` (the first 10 warmup runs are excluded from statistics), and use the average `aclmdlExecute` latency of the 90 successful measured runs as the evaluation metric. The candidate with the lowest average latency is the recommended value for the current test environment. After determining the optimal value, submit a PR with the following result:

```text
--multi_stream_parallel_mode=LoadBalance:55
```

Keep the model structure, input shape, `soc_version`, warmup count, and measured-run count consistent during tuning; `--multi_stream_parallel_mode` is the only variable.

## Notes

- Detection results use the COCO dataset 80-class labels (built into the code as `kCocoLabels`).
- To process more images, extend the `inputFiles` list in the `main()` function of `src/sample_yolov13.cpp`.
- The confidence threshold (`kConfThresh`, default 0.25) and NMS IoU threshold (`kIouThresh`, default 0.45) can be adjusted in `src/sample_yolov13.cpp`.
