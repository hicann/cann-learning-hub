# 第 2 章 编程基础：从 Hello World 到 Add

本章约 2 小时，承接第 1 章的 vecadd。学习者会把“用过”变成“会写”：先拆 kernel 四件套，再建立 GM/UB/寄存器直觉，随后完成 Add 的规格、分块、编译和验证，最后迁移到 Mul、SAXPY 与 2-D Add。

| 页面 | 内容 | 类型 |
|---|---|---|
| [02.01_chapter_intro.ipynb](02.01_chapter_intro.ipynb) | 承上启下、目标和导航 | 认知 |
| [02.02_kernel_structure.ipynb](02.02_kernel_structure.ipynb) | Host/kernel 边界、四件套、T.print | 可执行 |
| [02.03_core_programming_model.ipynb](02.03_core_programming_model.ipynb) | 内存位置、alloc_shared、copy、显式 SIMD、ReLU | 可执行 |
| [02.04_add_workflow.ipynb](02.04_add_workflow.ipynb) | Add 规格、均分、compile 与 jit | 可执行 |
| [02.05_chapter_practice.ipynb](02.05_chapter_practice.ipynb) | Mul / SAXPY / 2-D Add 三档实践 | 可执行 |

## 阅读方式

直接打开上方 Notebook 阅读正文、代码与保留的运行输出；自行执行需要按第1.4节准备950环境。折叠仅用于方便阅读，不影响代码单元的执行顺序。

## 运行方法

### 在终端运行

执行前加载课程环境脚本，并从本章根目录运行：

```bash
# 在启动内核前加载部署者提供的课程环境脚本
python src/02.02_print_probe.py
python src/02.03_relu_model.py
python src/02.04_add_workflow.py
python src/02.05_answer.py
```

### 在 Notebook 中运行

可执行页面均通过独立子进程启动脚本，NPU 初始化不放在 notebook 内核。先执行准备单元，再按顺序执行写入和运行单元。

## 本章代码约定

float32 的 SimdVF 向量维使用 64 的倍数；本章的逐元素计算均使用显式 `T.simd` 指令，串行循环推进每个 64-lane 向量块。Add 使用 `vadd`，Mul 使用 `vmul`，SAXPY 用 `BRC_B32` 广播 UB 中的一元素 `alpha`，再执行 `vmul/vadd`。T.print 放在 SimdVF 外，当前环境会显示 AIV block 消息。

## 文件说明

配图在 `images/`，答案在 `answer/`；运行脚本会在本地生成验证输出。
