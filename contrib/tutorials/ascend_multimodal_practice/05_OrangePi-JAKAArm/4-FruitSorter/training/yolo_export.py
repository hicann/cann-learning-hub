from ultralytics import YOLO

# 加载训练好的权重
model = YOLO("runs/detect/fruit_yolo11n_corrected/weights/best.pt")

# 导出 ONNX，opset=12 是通用稳定值
model.export(format="onnx", opset=12, imgsz=[640,640])