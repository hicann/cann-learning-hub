# ElementWiseCompoundCompute Axpy样例

## 概述

本样例展示复合计算接口 `Axpy` 的使用方法。`Axpy` 将标量乘法与向量加法融合在一条指令中完成，可减少中间结果写回和再次读取的开销。

## 目录结构

```text
03.03.08_axpy
├── scripts
│   ├── gen_data.py
│   └── verify_result.py
├── CMakeLists.txt
├── data_utils.h
├── apxy.asc
└── README.md
```

## 样例描述

本样例输入输出 shape 均为 `[1, 512]`，format 为 ND，数据类型为 `half`。核函数名为 `element_wise_compound_compute_custom`。

计算公式如下：

```text
dst = dst + src * scalar
```

其中 `input0` 作为 `src`，`input1` 作为初始 `dst`，`scalar = 2.0`。

Kernel 侧流程如下：

- 使用 `DataCopy` 将 `input0` 和 `input1` 从 GM 搬运到 UB。
- 将初始 `dst` 写入本地输出 Tensor。
- 调用 `Axpy` 完成标量乘法与向量加法融合计算。
- 使用 `DataCopy` 将计算结果从 UB 搬回 GM。

## 编译运行

在本样例根目录下执行如下命令：

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build -j
python3 scripts/gen_data.py
./build/demo
python3 scripts/verify_result.py output/output.bin output/golden.bin
```

执行结果如下，说明精度对比成功：

```bash
test pass!
```
