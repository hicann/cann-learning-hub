# MC2融合算子开发系列教程
本教程将带你学习Ascend C MC2融合算子的核心概念、设计思路与开发方法，围绕Matmul+MTE通信典型算子的开发流程展开详细拆解，助力开发者精准掌握MC2融合算子开发的核心要点与实操技巧。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 gitcode提供的轻量级 notebook 上运行。也可自行搭建jupyter lab，在本地环境执行使用。
- SRC：包含课程中所有的源码，供开发者自行下载及修改。
- PPT（建设中）：包含各课时的授课内容。  

## Notebooks

### MC2基础融合算子开发
- 该教程当前仅针对Ascend 950系列产品进行验证，其它产品使用存在问题，欢迎开发者提出issue或PR进行共建。

| Notebook | Link | 状态 |
|--|--|--|
| 1.1 章节介绍 | 在线体验建设中 | ✅ 已发布 |
| 1.2 MC2融合算子概念介绍 | 在线体验建设中 | ✅ 已发布 |
| 1.3 MC2融合算子开发流程讲解 | 在线体验建设中 | ✅ 已发布 |
| 1.4 章节实践 | 在线体验建设中 | ✅ 已发布 |
| 2.1 章节介绍 | [在线体验](./02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 MoE 架构概述 | [在线体验](./02.02_moe_architexture_overview.ipynb) | ✅ 已发布 |
| 2.3 并行策略 | [在线体验](./02.03_parallel_strategy.ipynb) | ✅ 已发布 |
| 2.4 Dispatch/Combine 算子 | [在线体验](./02.04_dispatch_combine_operator.ipynb) | ✅ 已发布 |
| 2.5 算子逻辑概述 | [在线体验](./02.05_operator_logic_overview.ipynb) | ✅ 已发布 |
| 2.6 核心流程拆解 | [在线体验](./02.06_dispatch_combine_core_flow.ipynb) | ✅ 已发布 |
| 2.7 Win 区内存布局 | [在线体验](./02.07_win_memory_layout.ipynb) | ✅ 已发布 |
| 2.8 Tiling 指南 | [在线体验](./02.08_tiling_guide.ipynb) | ✅ 已发布 |
| 2.9 Kernel 阶段指南 | [在线体验](./02.09_kernel_stage_guide.ipynb) | ✅ 已发布 |
| 2.10 量化 Dispatch 实践 | [在线体验](./02.10_quant_dispatch_practice.ipynb) | ✅ 已发布 |