# code目录说明

本目录存放香橙派开发板上运行的Python程序和脚本, 用于OM模型的转换与手写数字识别推理。

## 文件清单

| 文件 | 语言 | 功能说明 |
|---|---|---|
| `setup_env.sh` | Shell | CANN环境加载脚本, 配置环境变量并限制编译并行度 |
| `atc_convert.sh` | Shell | ATC模型转换脚本, 将ONNX转换为OM (Ascend310B4) |
| `preprocess.py` | Python | 图像预处理模块, 加载图片并归一化为模型输入格式 |
| `acl_classifier.py` | Python | ACL推理分类器模块, 封装AscendCL接口加载OM并推理 |
| `infer_single.py` | Python | 单张手写数字图片识别脚本 |
| `infer_batch.py` | Python | 批量手写数字图片识别脚本, 汇总识别成功率 |
| `requirements.txt` | Text | Python依赖库列表 |

## 文件功能详解

### setup_env.sh
香橙派开发板环境配置脚本。执行 `source setup_env.sh` 加载CANN Toolkit环境变量,
设置 `TE_PARALLEL_COMPILER=1` 和 `MAX_COMPILE_CORE_NUMBER=1` 限制ATC编译并行度,
防止开发板内存耗尽。同时打印NPU设备信息。

### atc_convert.sh
ATC模型转换脚本。将 `../models/` 目录下的ONNX模型通过ATC工具编译为OM格式,
输出到 `../output/` 目录。目标芯片为 `Ascend310B4` (香橙派)。
支持参数选择转换FP32或剪枝模型。

### preprocess.py
图像预处理模块。提供以下功能:
- `load_image(path)`: 加载图片为28x28灰度numpy数组
- `preprocess(array)`: 归一化(除255) + 标准化(均值0.1307, 标准差0.3081) + reshape为(1,1,28,28)
- `preprocess_image(path)`: 一步到位从文件路径预处理
- `extract_label(filename)`: 从文件名 `test_XX_labelY.png` 提取真实标签Y

### acl_classifier.py
ACL推理分类器模块。`ACLClassifier` 类封装了完整的AscendCL推理流程:
- `init()`: 初始化ACL运行时 (acl.init, set_device, create_context)
- `load_model()`: 加载OM模型, 查询输入输出规格, 分配Device内存
- `infer(input)`: 执行推理, 返回输出logits和耗时
- `predict(input)`: 推理并返回预测类别、置信度和耗时
- `release()`: 释放所有资源

### infer_single.py
单张图片识别脚本。加载OM模型, 对单张手写数字图片进行识别,
打印实际数字、识别结果、置信度、推理耗时和识别状态。

### infer_batch.py
批量图片识别脚本。加载OM模型, 对目录下所有测试图片逐张识别,
打印每张图片的识别结果, 最后汇总识别成功率、平均耗时、吞吐率等统计信息。

## 部署运行流程

```bash
# 1. 进入code目录
cd code

# 2. 加载CANN环境
source setup_env.sh

# 3. 安装Python依赖
pip3 install -r requirements.txt

# 4. ATC转换ONNX -> OM
bash atc_convert.sh

# 5. 单张图片识别
python3 infer_single.py --om ../output/simplecnn_mnist_fp32.om --image ../output/test_images/test_00_label0.png

# 6. 批量图片识别
python3 infer_batch.py --om ../output/simplecnn_mnist_fp32.om --dir ../output/test_images
```
