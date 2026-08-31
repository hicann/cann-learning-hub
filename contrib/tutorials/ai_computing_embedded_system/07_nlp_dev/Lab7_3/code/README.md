# code 目录说明

本目录包含实验7.3 Qwen1.5-0.5B 大语言模型在昇腾香橙派部署实验的代码，实现从 ONNX 模型到 OM 模型转换及 ACL 端侧推理，并通过 Gradio 提供 Web 聊天交互界面。

## 代码文件说明

| 文件 | 说明 |
| --- | --- |
| `convert_atc.sh` | Qwen1.5-0.5B-Chat ONNX → OM 模型转换脚本，使用 ATC 工具编译为昇腾 310B 的 OM 离线模型 |
| `qwen1.5-0.5b-chat.py` | Qwen1.5-0.5B-Chat 在昇腾 310B 上的推理脚本：ACL 加载 OM 模型、Transformers 分词、自回归生成、Gradio Web 界面 |
| `说明.txt` | 实验操作说明文档，描述模型转换与部署流程 |

## 子目录

- `onnx_export/` — 从云沙箱导出的 Qwen ONNX 模型文件
- `docx_images/` — 说明文档配图
