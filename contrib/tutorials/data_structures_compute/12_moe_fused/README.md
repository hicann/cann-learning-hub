# 第 12 章：MoE 融合算子和性能分析

以 MoE（Mixture of Experts）Router 路径（`matmul → softmax → topk → renorm`）为载体，完成一个 **4 合 1 融合算子**的设计、实现与性能分析：中间张量全部驻留片上（不落 GM）、4 次 kernel 发射合并为 1 次，并从**数据结构视角**分析融合的可行性判据、Top-K 的选择算法本质与融合后的性能形态。

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

1. **融合的数据结构判据**：子图中间张量容量 ≤ UB 且无跨核依赖 → 可融合（本算子中间量 < 0.5KB/行）
2. **中间张量介质降级**：scores/gate/topk_scores 从 GM 介质降级为 UB 介质，GM 流量 ↓100%
3. **Top-K = 选择问题**：K 轮 max+mask（selection，O(K·E)）而非全排序（O(E·log E)）
4. **工程全流程**：工程搭建 → Host Tiling → 纯标量 Kernel → 编译打包部署 → aclnn 调用 → 8 用例回归
5. **多核切分的硬件约束**：标量 GM 写经 L2（line=64B），多核共写同一 line 丢写 → 连续块 + 64B 对齐切分
6. **性能形态分析**：融合收益（访存/发射）与计算管线吞吐（标量流水）的定量归因

## 为什么是"纯标量"实现？

本章与第 8 章相同，刻意采用**纯标量实现（标量 GM 访问 + UB 普通数组）+ 多核连续块切分**：

- 与已验证稳定的 API 子集同构（第 2 章 StackExprOps、第 8 章 attention_custom 同一范式），规避本环境对高级特性的限制（MIX 任务 / TQue / reduce 指令缺陷 / 标量读+向量指令混用误编译等，详见 12.02 步骤 4）；
- 把教学焦点放在**融合的数据结构分析**上：可行性判据、中间张量流量核算、Top-K 选型、切分与 cache line 约束，而不是与工具链缺陷搏斗；
- 性能代价（标量流水吞吐低）本身是教学点：12.02 步骤 7 用实测数据说明"融合改变访存形态，不改变计算吞吐"。

## 前置知识

建议先完成：

- 第 2 章（栈的表达式求值）：UB 普通数组模拟数据结构、标量算子工程范式
- 第 8 章（工程部署及性能分析）：算子工程全流程（编译 → 打包 → 部署 → aclnn 调用）、纯标量多核切分

## 快速开始

```bash
# 设置环境
source $ASCEND_HOME_PATH/set_env.sh

# 生成测试数据（8 组边界用例，固定 seed=42）
python3 tools/gen_test_data.py

# 编译算子包（约 2-3 分钟）
cd src/custom_op && bash build.sh && cd ../..

# 一键回归（8 用例正确性）
bash tools/run_all.sh

# 性能测量（以 case_128_512_16_2 为例，事件计时 100 次取均值）
bash src/custom_op/test/run.sh data/case_128_512_16_2 --bench 100
```

## 章节目录

| Notebook | 内容 | 状态 |
| -- | -- | -- |
| [12.01 章节介绍](12.01_chapter_intro.ipynb) | 前置要求 / 章节目标 / 内容导航 / 融合数据流 | ✅ 已验证 |
| [12.02 动手实验](12.02_moe_router_fused_lab.ipynb) | 全流程 7 步骤（环境 → 基线 → 设计 → 源码 → 部署 → 回归 → 性能） | ✅ 已验证 |
| [12.03 章节实践](12.03_chapter_test.ipynb) | 编程实践（简单/中等/困难）+ 知识测验 | ✅ 已验证 |

## 目录结构

```
12_moe_fused/
├── 12.01_chapter_intro.ipynb        # 章节介绍
├── 12.02_moe_router_fused_lab.ipynb # 动手实验
├── 12.03_chapter_test.ipynb         # 章节实践与测试
├── README.md                        # 本章说明
├── 910b_guide.md                    # 910B3 平台运行指南
├── answer/                          # 参考答案（实践题 + 测验）
├── images/                          # 示意图（融合数据流 / 多核切分）
├── docs/design.md                   # 算子设计文档（定稿版）
├── data/                            # 测试数据（gitignore，gen_test_data.py 再生）
├── src/custom_op/                   # 算子工程（op_host / op_kernel / test）
└── tools/                           # 参考实现 / 数据生成 / 回归脚本
```
