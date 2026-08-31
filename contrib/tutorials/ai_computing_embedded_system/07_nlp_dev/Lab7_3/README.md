# 实验7.3 Qwen1.5-0.5B 大语言模型在昇腾香橙派部署实验

> **实验平台**：昇腾香橙派嵌入式开发板

本实验在昇腾香橙派嵌入式开发板上部署 Qwen1.5-0.5B-Chat 大语言模型。实验流程为：首先在昇腾云沙箱平台上将模型导出为 ONNX 格式，然后将 ONNX 模型传输到香橙派开发板，使用 ATC（Ascend Tensor Compiler）命令将 ONNX 模型编译为昇腾专用的 .om 离线模型，再利用 AscendCL 接口在 NPU 上执行推理。

## Notebook

- [lab7.3_nlp_deploy_orangepi.ipynb](./lab7.3_nlp_deploy_orangepi.ipynb)
