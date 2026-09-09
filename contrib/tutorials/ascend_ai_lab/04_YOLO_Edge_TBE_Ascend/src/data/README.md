# YOLOv5s COCO80 Sample Data

本实验使用 YOLOv5s 预训练模型，类别空间为 COCO 80 类。端侧部署、PyACL 推理、CPU/NPU 后处理对比和 Profiling 不需要下载完整 COCO 数据集，只需要少量图片用于功能验证和性能采样。

推荐目录结构：

```text
src/data/
├── coco80.names
└── images/
    ├── bus.jpg
    └── zidane.jpg
```

下载样例图：

```bash
mkdir -p src/data/images
wget -c -O src/data/images/bus.jpg https://ultralytics.com/images/bus.jpg
wget -c -O src/data/images/zidane.jpg https://ultralytics.com/images/zidane.jpg
```

如果开发板没有外网，可以在 PC 或云服务器下载后上传：

```bash
scp -r src/data/images HwHiAiUser@192.168.137.100:/home/HwHiAiUser/edge_lab/04_YOLO_Edge_TBE_Ascend/src/data/
```

