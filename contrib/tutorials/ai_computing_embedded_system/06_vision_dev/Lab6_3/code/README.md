# code 目录说明

本目录包含实验6.3 昇腾香橙派的图片和视频 YOLO 目标检测实验的所有代码，基于 YOLOv8 模型在香橙派 AIPro（Ascend 310B4 NPU）上实现端侧目标检测，涵盖 DVPP+AIPP 硬件加速与 OpenCV 纯 CPU 两条路径。

## 代码文件说明

| 文件 | 说明 |
| --- | --- |
| `export_om.sh` | YOLOv8 模型 ONNX → OM 转换脚本，使用 ATC 工具编译为昇腾 310B4 的 OM 模型 |
| `aipp_yolo.cfg` | AIPP 静态预处理配置文件（RGB888_U8 格式、640×640 输入） |
| `yolo_postprocess.py` | YOLOv8 后处理工具模块：letterbox 预处理、输出解码、NMS、坐标还原、可视化绘制 |
| `yolo_image_detect.py` | 图片 YOLO 目标检测（DVPP + AIPP 硬件加速路径），DVPP 硬件解码 JPEG → VPC 缩放 → NPU 推理 |
| `yolo_image_detect_opencv.py` | 图片 YOLO 目标检测（OpenCV 纯 CPU 路径），OpenCV 解码缩放 → NPU 推理 |
| `yolo_video_detect.py` | 视频 YOLO 目标检测（DVPP + AIPP 硬件加速路径），逐帧 DVPP 硬件解码 + VPC 缩放 |
| `yolo_video_detect_opencv.py` | 视频 YOLO 目标检测（OpenCV 纯 CPU 路径），逐帧 OpenCV letterbox 缩放 |
| `yolo_usb_camera_detect.py` | USB 摄像头实时 YOLO 目标检测 + HDMI 显示，视频采集 → NPU 检测 → 显示器输出 |
| `benchmark_compare.py` | 三种 YOLO 预处理路径性能对比（OpenCV+纯OM / OpenCV+AIPP-OM / DVPP+AIPP-OM） |
| `download_yolov8n.py` | YOLOv8n 模型获取脚本：优先从直连地址下载 `yolov8n.onnx`，失败则回退至 ultralytics 导出 |
| `yolov8n.onnx` | YOLOv8n ONNX 模型文件（**不再随仓库分发**，运行 `python3 download_yolov8n.py` 获取） |
| `fusion_result.json` | 算子融合结果文件 |

## 模型文件获取

ONNX 模型不属于仓库准入文件类型，请运行以下命令在本目录获取 `yolov8n.onnx`：

```bash
python3 download_yolov8n.py
```

脚本优先从直连地址下载预导出的 `yolov8n.onnx`；若直连失败，可安装 ultralytics 回退导出：

```bash
pip install ultralytics onnx
python3 download_yolov8n.py
```

获取后再执行 `bash export_om.sh yolov8n.onnx ../output` 完成 ATC 转换。
