# AOT SuperKernel技术系列（一）：从图执行优化说起

## 什么是 AOT SuperKernel

> AOT SuperKernel，即 Ahead-of-Time SuperKernel，是面向模型的算子二进制融合优化能力。它在模型执行前识别可被融合的 Task，将这些 Task 合并为单个 SuperKernel Task，优化多个 Task 在执行时的调度开销。

本文先从图执行优化说起，建立理解 AOT SuperKernel 的基础视角。

![superkernel_task_fusion](images/superkernel_task_fusion.png)

图示展示了多个 Task 融合为 SuperKernel Task 前后的调度执行形态变化。

## 图执行的载体：Aclgraph

在图执行阶段，硬件调度器基于 Aclgraph 提供的 ModelRI (Model Runtime Instance) 启动不同流 (Stream) 的任务调度，并在这些流上依次执行对应的任务 (Task)，如下图所示。

![aclgraph_stream_task_scheduling](images/aclgraph_stream_task_scheduling.png)

图示抽象了硬件调度器基于 Aclgraph 组织多条 Stream 与 Task 的关系，并展示 Stream 内 Task 顺序和跨 Stream 依赖的表达方式。

- Task 是 Aclgraph 的基本调度单元。不同类型的 Task 承载不同的执行信息。

  - Kernel Task：承载 Kernel 入口、参数、Block 数、Kernel 类型等执行信息；

  - Event Task：承载同步语义，存在 record、wait、reset 三个子类型；

- Stream 是 Aclgraph 在 Device 上调度的 Task 队列。在同一条 Stream 中，Task 按照进入队列的顺序依次调度，不同 Stream 之间 Task 调度的先后关系通常由 Event Task 建立。

在模型执行中，单个 Task 的耗时只是性能组成的一部分。随着图规模扩大、Task 数量增加，硬件调度器需要在多条 Stream 上处理大量 Task，整图执行效率也会受到调度效率的影响。

当图中存在大量粒度较小的 Kernel Task，或者多条 Stream 之间存在频繁的 Task 相互依赖时，开销会集中体现在以下方面。

- Task 调度等待变多。 每个 Task 被调度时都会产生调度间隙。随着图规模扩大、Task 数量增加，调度等待会逐步成为整图执行成本的一部分。

- Kernel 频繁启动。 硬件每次启动 NPU 计算时，都会产生独立于计算本身的启动开销。推理场景小数据量计算时，硬件调度开销占比突出。

> SuperKernel 优化的重点，是在保持原有 Aclgraph 图语义和 Kernel 计算逻辑不变的前提下，从整图视角改变 Task 的调度方式，减少调度等待和启动开销。

## SuperKernel Task 生成

在模型执行前，Aclgraph 可提供模型中的 Stream、Task 信息，SuperKernel 利用这些信息，识别可被融合的 Task，并将这些 Task 合并为单个被调度的 SuperKernel Task，生成过程中主要关注以下四类信息：

1. Scope 边界：确定哪些 Task 可以在一次 SuperKernel Task 调度中统一执行。

2. SubTask 编排：将 Aclgraph 中 Task 调度转换为 SuperKernel Task 内部的 SubTask 执行。

3. 同步表达：保留原有 Stream 上的任务依赖关系，并生成必要的任务间同步。

4. 运行入口：生成可调度的 SuperKernel Entry 和参数信息。

![superkernel_task_generation](images/superkernel_task_generation.png)

图示说明了 AOT 阶段如何将 Aclgraph 提供的 Stream、Task 调度信息转换为融合后的 SuperKernel Task，供后续执行阶段使用。

## SuperKernel 性能收益来源

SuperKernel 的收益来自调度方式的变化。不同图的主要收益点不完全相同，通常集中在以下三类机制：

- 减少 Task 启动开销。 原本需要多次启动的 Task，可以通过一次 SuperKernel Task 启动统一完成，从而减少多次启动带来的额外开销。

- 减少 Task 调度等待。 多个独立 Task 之间通常存在硬件调度间隙。通过一次 SuperKernel 调度统一承接多个 Task，可以压缩 Task 逐个调度带来的等待时间。

- 算子间流水并行。 在满足依赖和正确性约束时，SuperKernel 可以通过任务间同步优化，让相邻 SubTask 在依赖满足后形成更细粒度的流水执行窗口，进一步压缩端到端耗时。

![superkernel_performance_benefits](images/superkernel_performance_benefits.png)

图示对比了 Task 依次执行和融合为 SuperKernel Task 后的执行调度形态，收益主要来自启动开销减少、调度等待减少和流水并行。

## SuperKernel 执行结果的正确性

在满足可融合条件并保留必要同步关系的前提下，SuperKernel 改变的是 Aclgraph 内 Task 的执行调度方式，不改变原有计算语义和基础依赖语义。

其正确性依赖以下约束：

- 原有 Task 在 SuperKernel 中仍以 SubTask 形式执行，关键信息被等价转换并保留语义。

- 原有 Task 的依赖关系在 SuperKernel 中仍通过任务间同步进行保留。

- 对于无法证明安全的调度重组，SuperKernel 保持保守处理，不改变原有 Task 的调度方式。

同时，SuperKernel 也提供了 SK Meta（包含 Fail Reason、Device Args、Profiling 等）观测产物。开发者可以基于这些产物，判断 SuperKernel 在 Aclgraph 中的使能与执行情况。

## 小结

> SuperKernel 面向图执行中的系统性问题。它从 Aclgraph 提供的 Stream 和 Task 调度信息出发，在正确性约束下将一段图转换为可被一次调度执行的 SuperKernel Task。

下一篇将以全景视图展开，围绕 AOT SuperKernel 的整体架构，介绍它如何承接 PyTorch 图编译输入，并串起 SuperKernel Task 生成与运行期执行衔接的完整链路。
