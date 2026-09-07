# 算子开发与应用

本课程以昇腾 CANN 算子开发为主线，通过四个在线实验依次完成环境认知、算子原型分析、Host 侧调用链理解，以及 Kernel 工程编译、功能验证和性能采样。

## 课程适用学习人群

- 具备 Python、C/C++ 和 Linux 命令行基础的高年级本科生、研究生及初级开发者。
- 希望了解 CANN 算子原型、Host/Tiling、Ascend C Kernel 和 msProf 基础流程的学习者。

## 整体学习目标

- 识别 CANN 安装环境、NPU 状态和算子开发工具链。
- 理解算子接口中的输入输出、Shape、dtype 和 format 约束。
- 区分 Host 侧参数检查、Tiling、资源管理与 Kernel 启动职责。
- 理解 Ascend C 的 CopyIn、Compute、CopyOut 三阶段，并完成真实编译、测试和性能采样。

## 课程支持的硬件产品

| 硬件产品 | 验证状态 |
| --- | --- |
| Atlas A2 系列产品（910B3） | 已由课程评审环境验证 |

已验证软件版本：CANN 9.0.0，Python 3.11.4。

## 已验证的在线体验环境

- CANNLab 云开发环境
  - NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`
  - 规格：`1*NPU 910B3 16vCPUs 32GiB`
  - Python 内核：`cann_py311`

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

## 课程章节目录

| Notebook | 内容 |
| --- | --- |
| `01.01_environment_cloud_experience.ipynb` | 检查 CANNLAB 预置环境，运行最小 NPU 样例，并使用 msOpGen 生成算子工程脚手架。 |
| `01.02_operator_prototype.ipynb` | 以 Sinh 为例分析 CANN 算子接口、Shape/Type 推导和跨生态接口差异。 |
| `01.03_host_side_experience.ipynb` | 分析 Host 侧调用链、Tiling 数据组织和框架胶水代码。 |
| `01.04_kernel_development_performance.ipynb` | 观察 Ascend C Kernel 三阶段，编译运行完整样例，并使用 msProf 采集真实性能数据。 |

## 运行说明

1. 从本目录依次打开四个 Notebook。
2. 使用 CANNLab 的 `cann_py311` Python 内核执行。
3. Notebook 会在当前目录下创建 `.cann_course_work/` 临时工作区；该目录仅存放本次实验生成物。
4. 参考答案位于 `answer/`，用于完成实验后的自查，不应替代实际运行记录。
