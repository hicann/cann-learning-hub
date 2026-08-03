# PyAsc 昇腾算子编译器开发教程

## 背景

本教程由 XX 学校 XX 老师及其团队开发，面向编译原理和昇腾 AI 开发者，以 PyAsc 开源编译框架为载体，系统讲解从 Python 前端、MLIR 中间表示（ASC-IR）到 Ascend C 目标代码生成的完整编译链路。

PyAsc 是华为 CANN 社区开源的昇腾算子编译器框架，基于 MLIR/LLVM 构建，支持使用 Python 语法编写昇腾 NPU Kernel，并通过 JIT 编译自动生成 Ascend C 代码。

## 适用对象

- 学习编译原理课程的高校学生
- 希望理解昇腾算子开发与编译器技术的开发者
- 对 MLIR/TableGen/代码生成感兴趣的研究人员

## 章节目录

| 章节 | 内容 | 建议学时 |
|------|------|:---:|
| [01 PyAsc 简单算子](./01_pyasc_simple_operator/01_pyasc_simple_operator.ipynb) | 使用 PyAsc 编写 Add 算子，理解 GlobalTensor/LocalTensor 内存建模、数据搬运、向量计算与事件同步，建立端到端编译链路认知 | 2学时 |
| [02 Python↔C 前端映射](./02_python_c_mapping/02_python_c_mapping.ipynb) | 分析 PyAsc Python 前端如何精确映射 Ascend C API，掌握 @overload、@require_jit、OverloadDispatcher 和 IR Builder 调用机制 | 2学时 |
| [03 MLIR 定义 ASC-IR](./03_mlir_asc_ir/03_mlir_asc_ir.ipynb) | 理解 ASC-IR 基于 MLIR Dialect 和 TableGen 的定义方式，分析 BinaryTemplateL0123Op 如何一次生成 L0~L3 四类 Operation | 2学时 |
| [04 ASC-IR→AscendC 代码生成](./04_ascir_to_ascendc/04_ascir_to_ascendc.ipynb) | 掌握 ASC-IR 到 Ascend C 的自动代码生成路径，理解 genEmitter、Traits、paramTypeLists 的作用，区分自动生成与手写路径 | 2学时 |

## 前置知识

1. **编译原理基础**：词法分析、语法分析、语义分析、中间表示和目标代码生成
2. **昇腾算子开发基础**：Global Memory、Local Memory、Vector 计算单元、数据搬运与事件同步
3. **Python 与 C++ 基础**：Python 装饰器、类型注解、C++ 模板接口
4. **MLIR 与 TableGen 基础**：MLIR Dialect、Operation、Type、Attribute；TableGen 声明式代码生成

## 实验环境

- 操作系统：支持 CANN 的 Linux 发行版
- Python：3.9 至 3.12
- CANN：社区版 8.5.0.alpha001 及以上
- PyAsc：v1.1.0 及以上
- LLVM/MLIR：19.1.7
- 硬件：Atlas A2 或 A3 训练/推理产品（仿真模式可不使用硬件）

## 环境准备

```bash
git clone https://gitcode.com/cann/pyasc.git
cd pyasc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-build.txt
pip install -r requirements-runtime.txt
pip install pyasc pytest lit torch
```

## 参考资料

- [PyAsc 项目仓库](https://gitcode.com/cann/pyasc)
- [CANN 社区](https://gitcode.com/cann/community)
- [PyAsc 快速入门](https://gitcode.com/cann/pyasc/blob/master/docs/quick_start.md)
- [PyAsc 架构介绍](https://gitcode.com/cann/pyasc/blob/master/docs/architecture_introduction.md)

## 许可证

本教程采用 [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议。
