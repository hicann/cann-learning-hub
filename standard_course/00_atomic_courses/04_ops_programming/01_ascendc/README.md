# Ascend C 算子编程课件素材（L4 · 04_ops_programming/01_ascendc）

> 对应 [原子课程总纲](../../README.md) 4.1 节 Ascend C 算子编程方向（L4-01~17）的课件素材落位目录，与 [directory.md](../../directory.md) 规划的 `04_ops_programming/01_ascendc/` 子方向一致。

| 位置 | 硬件路线 | PPT 数 | 内容 |
|------|---------|--------|------|
| [a2a3/](./a2a3/) | Atlas A2/A3（路线 A） | 11 | 导论与 SIMD 编程模型 → Memory 矢量 → 矩阵 → 融合 → 调试调优 → PyTorch 单算子调用 → 图模式接入 |
| [ascend950/](./ascend950/) | Ascend 950（路线 B） | 12 | 导论与 SIMD 编程模型（编程模型/Reg 矢量/矩阵 Tensor API/融合）→ SIMT → 调试调优 → 优化案例 → PyTorch 单算子调用 → 高级编程 → AclNN 工程化 → 入图 |
| （根目录） | 双路线公共 | 3 | [NPU 架构与 Ascend C 编程导论](./npu_architecture_and_ascend_c_programming_introduction.pptx)（64 页三篇合并精编：计组专题 / 并行计算第 6 章主讲义）；[a2a3 矢量算子开发](./a2a3_ascend_c_simd_vector_operator_development.pptx)（启航营两天营 D2 讲义）；[ascend950 矢量算子开发](./ascend950_ascendc_vector_operator_development.pptx)（950 备用） |

> 素材源自 [03_new_courses/ai_computing_and_neural_network_architecture_practice](../../../03_new_courses/ai_computing_and_neural_network_architecture_practice/) 第二部分（36 学时，双路线各 9 讲），2026-10 迁入本目录统一维护；课程讲次与实践安排见该课程 [syllabus.md](../../../03_new_courses/ai_computing_and_neural_network_architecture_practice/syllabus.md) 第 4.2/4.3 节。
