# 实验5.4 香橙派开发板 OM 模型转换与手写数字识别

> **实验平台**：香橙派开发板（昇腾 310B NPU）

本实验以 MNIST 手写数字识别任务为载体，在香橙派开发板（昇腾 310B NPU）上完成模型部署的最后两个环节：ATC 模型转换和 ACL 端侧推理。实验 5.3 已在云沙箱（昇腾 910B3）上完成模型训练并导出 ONNX 文件，本实验将这些 ONNX 文件部署到香橙派上，通过 ATC 编译为 OM 离线模型，再利用 AscendCL 接口在 NPU 上执行推理。

## Notebook

- [lab5.4_orangepi_om_model_conversion.ipynb](./lab5.4_orangepi_om_model_conversion.ipynb)
