# CANN 基础知识课程

本课程是 CANN 生态的入门通识课，面向零基础初学者，用大量生活化比喻和图文讲解，让你像读故事一样理解 AI、NPU 和 CANN。课程以 Jupyter Notebook 形式提供，支持在线交互式运行。

##  软硬件配套说明

本课程涉及 NPU 设备信息查询、CANN 环境检查以及基于 PyTorch 的 NPU 计算示例，推荐使用以下软硬件环境：

| 项目      | 要求                                    |
| ------- | ------------------------------------- |
| 支持硬件    | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上                             |
| Python  | 3.11                                  |


## 在线体验环境

本课程支持以下在线体验环境：

| 体验环境                            | 镜像模板 / 版本                  | Python 内核      | 说明                     |
| ------------------------------- | -------------------------- | -------------- | ---------------------- |
| cann-learning-hub 在线体验 Notebook | `cann_9.0.0_py3.11-A2-arm` | Python 3.11.15 | 各课程表格中的“在线体验”链接可直接打开运行 |
| CANNLab 云开发环境                   | `cann_9.0.0_py3.11-A2-arm` | Python 3.11.15 | CANNLab 可用于课程代码运行及 NPU 环境实践，具体使用方法可参考 [CANNLab 体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。    |

**注意**： 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 CANN 安装指南，并选择对应CANN版本的文档。
##  本地环境体验

#### 安装 PyTorch 和 torch_npu

推荐创建独立的 Python 虚拟环境后安装课程所需依赖：

```bash
pip3 install torch
pip3 install torch-npu
```

安装完成后，可以通过以下代码检查 PyTorch 是否能够正常识别昇腾 NPU：

```python
import torch
import torch_npu

print("PyTorch version:", torch.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU device count:", torch.npu.device_count())
```


> **说明：**
> PyTorch、torch_npu 与 CANN 之间存在版本配套关系。安装前请根据当前 CANN 版本选择相匹配的 PyTorch 和 torch_npu 版本，避免因版本不匹配导致安装失败或 NPU 无法正常使用。
## 课程内容

| 序号 | 课程 | 内容概要 | 在线体验 |
|:----:|------|---------|-----|
| 1 | [人工智能基础](./01_ai_basics.ipynb) | AI 发展历程 → AI/ML/DL → 模型 → 训练与推理 → 常见架构 → 计算图 → 算子 → 张量 → 从模型到昇腾 NPU | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/01_ai_basics.ipynb) |
| 2 | [什么是 NPU](./02_what_is_npu.ipynb) | 为什么需要 NPU → CPU vs NPU 算力差距 → 昇腾产品全览 → Host/Device → NPU 内部 6 组件 → AI Core 三大部分 → 多核并行 → GPU 对比 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/02_what_is_npu.ipynb) |
| 3 | [什么是 CANN](./03_what_is_cann.ipynb) | 从 NPU 到 CANN → 架构总览 → 各组件详解（框架适配/算子库/通信库/图引擎/加速库/Ascend C/编译器/运行时/驱动）→ 一行代码的完整旅程 → CANN vs CUDA → 环境验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/03_what_is_cann.ipynb) |
| 4 | [Hello World：NPU 加法](./04_hello_world_npu.ipynb) | 体验从 PyTorch 到昇腾 NPU 的零门槛迁移：一行代码切换设备，亲手感受 NPU 加速计算 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/04_hello_world_npu.ipynb) |
| 5 | [课程总结与实践](./05_course_summary.ipynb) | 四节课知识串联成全景图，CANN/CUDA 速查对照，三道难度递进的 NPU 实践题 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/05_course_summary.ipynb) |



## 课件

| 课件 | 时长 | 说明 |
|------|:----:|------|
| [artificial_intelligence_basics.pptx](./slides/artificial_intelligence_basics.pptx) | 2h | 完整版授课课件，涵盖人工智能基础、NPU 架构、CANN 软件栈、Hello World 全部课程内容 |
| [artificial_intelligence_basics_light.pptx](./slides/artificial_intelligence_basics_light.pptx) | 1h | 精简版授课课件，聚焦核心概念，适合 1 课时快速导览或讲座场景 |

## 适用人群

- 对 AI 和昇腾 NPU 感兴趣的初学者（无需编程基础）
- 有 GPU/CUDA 经验、想快速了解 CANN 的开发者
- 需要向团队介绍 CANN 基础概念的工程师

## 学习建议

1. 按顺序学习，每课承上启下
2. 每课都穿插了可执行的代码段，建议在昇腾 NPU 环境下运行体验
3. 有 CUDA 经验的同学可重点看第 3 课的"CANN 与 CUDA 对照"部分，快速建立映射

## 目录结构

```text
cann_basics/
├── README.md
├── slides/       # 授课课件
├── images/       # 课程图片资源
└── answer/       # 练习答案与批改脚本
```
