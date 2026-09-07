# CANN 新增经典书籍目录（New Books）

> 依据 skill 1.2/1.3 + `ref/cann_books.md`（2026-08 版）：**两本教材**，每本新增书籍单独成文。

## 书籍清单

| 文件 | 书籍 | 定位（业界对标） | 状态 |
|------|------|----------------|------|
| [book1_ascend_c_parallel_programming.md](./book1_ascend_c_parallel_programming.md) | **教材一《Ascend C 并行程序设计》** | 对标《Programming Massively Parallel Processors》（PMPP）——昇腾算子开发权威教材 | 规划 |
| [book2_ai_chip_and_system.md](./book2_ai_chip_and_system.md) | **教材二《人工智能芯片与系统》** | 对标《AI Accelerators》类教材（Hennessy & Patterson DSA 章延伸）——NPU 架构与系统教材 | 规划 |

## 双教材分工与课程同构

```
教材一《Ascend C 并行程序设计》 ↔ 03_new_courses 课程三《加速计算》 ↔ 原子课 L4-01~30/L3 层
教材二《人工智能芯片与系统》   ↔ 02_traditional_courses 体系结构课三幕剧 ↔ 原子课 L4-47~49
```

- **教材一**（编程视角）：读者为算子开发者/高年级学生；章节与课程三"编程模型→并行模式→多卡→加速库"四章同构；
- **教材二**（架构视角）：读者为体系结构方向学生/芯片从业者；以"AI 时代对架构的要求"为叙事主线，覆盖负载画像→达芬奇应答→优化方法论→超节点（含 CIM/PIM 前沿）。

## 编写规范（每章三件套）

每章 = **正文（原理+图示）** + **可运行实验**（复用原子课 Notebook）+ **章末习题**（对标 Teaching Kits/DLI 教材形态）。

## 已有素材基础

- 《Ascend C 异构并行程序设计》（HIT1920 电子版，<https://gitcode.com/HIT1920/AscendCBook>）可作为教材一起点素材；
- 体系结构课三幕剧设计（02_traditional_courses）为教材二第 II 部分提供教学化叙事底稿。


## 进一步学习参考（skill 1.4）

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标/参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程包实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |
