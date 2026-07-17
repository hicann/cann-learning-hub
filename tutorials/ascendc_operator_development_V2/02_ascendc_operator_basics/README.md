# 第2章 算子入门

## 章节概述
本章为面向 Ascend 950 的算子开发入门实践章节，介绍 SIMT 与 SIMD 编程模型及对应典型算子结构，通过 Hello World、SIMD 连续类矢量算子（Add）、SIMD 矩阵算子（Tensor 编程与 Matmul 高阶 API）、SIMT 离散类矢量算子（Gather）五个由浅入深的示例，带你完成从理论到代码的第一步落地。

## 章节大纲

### 2.1 章节介绍
- 本章目标与学习路径
- 950 多层级 API 总览：框架 API（Tpipe/Tque）/ 基础 API / 语言扩展层 C API / Tensor API / 高阶 API
- 本章各节选用 API 的原则与兼容性说明

### 2.2 SIMT 与 SIMD 介绍
- SIMD：单指令多数据，面向连续矢量，对应 Vector 计算单元
- SIMT：单指令多线程，warp/线程模型，类比 GPU CUDA，面向离散矢量
- 950 的 SIMD/SIMT 混合编程能力
- 各自适用的算子类型

### 2.3 对应典型算子结构
- 连续类矢量算子（Add/Relu，SIMD/Vector）的结构
- 矩阵算子（MatMul，Cube）的结构
- 离散类矢量算子（Gather/Scatter，SIMT）的结构
- 算子结构与硬件计算单元的映射关系

### 2.4 Ascend C 的 Hello World
- 环境准备与快速入门
- 绑定 `hello_world` 样例：核函数直调、NPU 侧运行验证
- 工程目录结构
- `<<<>>>` 核函数调用方式
- 验证
- 兼容性：910B/910C/950

### 2.5 SIMD 连续类矢量算子示例（Add 算子）
- 绑定 `c_api_async_add` 样例
- 语言扩展层纯 C 接口编程范式：指针型计算、数组 `[]` 内存分配
- 异步搬运 + 计算流程：GM→Local→Add→GM
- 同步管理
- 与 C++ Tensor 写法对比
- 兼容性：910B/910C/950

### 2.6 SIMD 矩阵算子示例（Tensor 编程）
- 绑定 `matmul_tensor_api` 样例：静态 Tensor API、多核 Matmul
- MakeTensor + Slice 分核分块逻辑
- Copy（GM2L1 / L12L0A/B / L0C2GM）自动 NZ/ZN 布局转换
- Mmad 矩阵乘加
- 编译选项 `-DCMAKE_ASC_ARCHITECTURES=dav-3510`、sim 仿真模式
- 兼容性：仅 950

### 2.7 SIMT 离散类矢量算子示例（Gather 算子）
- 绑定 `03_simt_api/.../01_add` 样例（以 Gather 为例）
- SIMT API（warp、基本数学、类型转换）使用
- 线程索引与数据映射
- SIMT 与 SIMD 写法对比
- 混合编程初探
- 兼容性：仅 950

### 2.8 章节测试
- 选择题 + 判断题
- 覆盖 API 层级选择、SIMD/SIMT 区别、C API 与 Tensor 写法、兼容性判断

### 2.9 SIMD 矩阵算子示例（Matmul 高阶 API）
- 绑定 `matmul_advanced_api` 样例
- Matmul 高阶 API 封装数据搬运与 Cube 计算调度
- 矩阵规格配置、tiling 生成、输入输出 Tensor 设置与结果写回
- CMake 构建流程
- 兼容性：910B/910C/950
