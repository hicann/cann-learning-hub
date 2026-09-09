# 第 8 章：工程部署及性能分析

基于注意力算子（QKᵀ → scale → softmax → AV）打通 Ascend C 自定义算子工程全流程：**msOpGen 工程生成 → Host/Kernel 实现 → 编译打包部署 → aclnn 单算子调用 → msProf 性能分析 → msSanitizer 异常检测**，并完成大模型注意力算子的 **O(S²) 复杂度推导与实测验证**。

## 支持平台

| 平台 | 芯片 | SoC 版本 | 状态 |
|------|------|----------|------|
| 昇腾 910B3 | Ascend910B | ascend910b | ✅ 已验证 |

已验证软件版本：CANN 9.0.0+（Atlas A2 系列，40 个 AIV 核）。

## 在线体验环境

本课程已在 **CANNLab 云开发环境** 中完成运行验证：

| 环境 | 状态 | 说明 |
| -- | -- | -- |
| CANNLab 云开发环境 | ✅ 已验证 | NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`，规格：`1*NPU 910B3 16vCPUs 32GiB`，Python 内核：Python 3.11.15 |

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

## 核心教学目标

1. **msOpGen 工程生成**：基于 ops.json 原型定义一键生成算子工程
2. **Host + Kernel 全流程**：Tiling / InferShape / InferDataType + 朴素三步注意力实现
3. **编译 → 打包 → 部署**：build.sh → .run 安装包 → 用户目录安装 + aclnn 单算子调用
4. **msProf 性能分析**：上板采集 op_summary（OpBasicInfo / PipeUtilization 报告解读）
5. **复杂度演示**：多 seq_len 实测耗时曲线验证 O(S²)，Flash Attention 理论对比
6. **msSanitizer / msDebug**：异常检测与调试工具使用

## 为什么用"纯标量 + 多核"实现注意力？

本实验刻意采用**纯标量实现（GM 标量访问 + UB 普通数组）+ 40 核按行切分**，原因：

- 与已验证稳定的 API 子集同构，规避环境对高级特性（MIX 任务 / TQue / UB GetValue 与大循环组合）的限制；
- 复杂度演示更直接：O(S²) 增长与实现方式无关，标量版曲线（85 / 355 / 1386 / 5506 ms）清晰呈现 ×4 规律；
- 多核切分（SPMD）是独立教学点：`row = blockIdx; row < S; row += blockDim`，加速比 38×。

Matmul（Cube）与 Softmax 的向量化实现作为课后实践题（见 08.03）。

## 前置知识

建议先完成：
- `01`-`07` 全部课程（数据组织、内存访问、并行计算、矩阵乘法分块、栈的表达式求值、算子工程与性能基础等）
- 昇腾 C 基础：工程生成 / Tiling / Kernel 结构（如 add_custom 模板）

## 快速开始

```bash
# 设置环境
source $ASCEND_HOME_PATH/set_env.sh
# 或 source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子（约 2-3 分钟）
cd src/attention_op
bash scripts/build_ops.sh

# 编译 runner + 生成测试数据 + 运行 benchmark
source scripts/env_custom_opp.sh
bash scripts/build_runner.sh
python3 scripts/gen_data.py 512 1024 2048 4096 --dim 64
aclnn_runner/build/main_attention_benchmark data 512 64

# msProf 性能采集
bash scripts/run_profiling.sh 512 --output prof
```

## 章节目录

| Notebook | 内容 | 状态 |
| -- | -- | -- |
| [08.01 章节介绍](08.01_chapter_intro.ipynb) | 前置要求 / 章节目标 / 内容导航 / 数据流图 | ✅ 已验证 |
| [08.02 动手实验](08.02_attention_operator_lab.ipynb) | 全流程 6 步骤（环境 → 工程 → 实现 → 部署 → 性能 → 复杂度） | ✅ 已验证 |
| [08.03 章节实践](08.03_chapter_test.ipynb) | 综合编程实践 + 知识测验 | ✅ 已验证 |

## 目录结构

```
08_engineering_deployment_and_perf_analysis/
├── 08.01_chapter_intro.ipynb   # 章节介绍
├── 08.02_attention_operator_lab.ipynb  # 动手实验
├── 08.03_chapter_test.ipynb    # 章节实践与测试
├── answer/                     # 参考答案
├── images/                     # 示意图（数据流 / 工程流程 / O(S²) 曲线）
└── src/attention_op/           # 算子工程源码
    ├── custom_ops/             # ops.json + msOpGen 生成的算子工程
    ├── aclnn_runner/           # aclnn API 调用 runner（benchmark）
    ├── scripts/                # 构建 / 打包 / 部署 / 性能采集脚本
    └── data/                   # 测试数据与 torch 参考输出
```
