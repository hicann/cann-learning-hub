# 实验5.3 ONNX 模型到 OM 模型转换与验证云沙箱实验

> **实验平台**：GitCode CANNLab 云沙箱（昇腾 910B3）

本实验在 GitCode 昇腾 910B3 云沙箱环境中，以 SimpleCNN（MNIST 手写数字识别）为载体，完整走通模型部署的标准化链路：ONNX 模型 → ATC 编译生成 OM → AscendCL 接口 NPU 推理 → 性能评估。实验同时提供 Python 推理程序，并对比 FP32 全精度、剪枝等多种模型格式的部署效果。

## Notebook

- [lab5.3_cann_sandbox_onnx_to_om.ipynb](./lab5.3_cann_sandbox_onnx_to_om.ipynb)
