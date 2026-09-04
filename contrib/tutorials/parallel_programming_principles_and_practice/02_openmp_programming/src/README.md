# 源码说明

本目录从工作区原实验 `Ascend-SpMV` 按课程所需文件白名单迁移 CPU/OpenMP 子集。课程迁移未修改原工程；未包含 `.git`、`build`、矩阵缓存、生成日志以及 NPU 模拟 backend。

为支持章节中的单变量实验，课程副本对主机端代码做了最小教学适配：

- 从公共头文件和构建目标中移除未迁移的 NPU backend；
- 将 `CpuOpenMp16Backend` 改名为 `CpuOpenMpBackend`；
- 构建脚本明确使用鲲鹏毕昇 Host 编译器 `clang++`；
- 原工程固定 16 线程和 `schedule(static)`，课程副本允许通过 `OMP_NUM_THREADS` 与 `OMP_SCHEDULE` 配置；未设置变量时仍默认 16 线程和 static。

这些改动不改变 CSR 数据结构和 SpMV 数值计算。基准结果文件由实验运行时生成，不随仓库提交，也不能作为当前环境实测结果。
