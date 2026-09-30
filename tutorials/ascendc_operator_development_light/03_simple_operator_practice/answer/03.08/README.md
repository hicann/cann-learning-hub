# 3.8 PyTorch调用算子实践：参考答案

两道题均提供 Pybind11 和 Torch Library 两种完整接入工程，使用同一题的 Ascend C Kernel。固定目标架构为 Atlas A2/A3 的 `dav-2201`，运行环境需具备 CANN、配套 PyTorch / torch_npu、Python 开发头文件、pybind11、CMake 和 C++ 编译器。

## 目录

```text
03.08/
├── run_all.sh
├── softmax/
│   ├── common/softmax_custom.asc
│   ├── pybind/           # CMakeLists.txt、pybind_binding.cpp、softmax_test.py
│   └── torch_library/    # CMakeLists.txt、torch_binding.cpp、softmax_test.py
└── matmul/
    ├── common/mmad_custom.asc
    ├── pybind/           # CMakeLists.txt、pybind_binding.cpp、matmul_test.py
    └── torch_library/    # CMakeLists.txt、torch_binding.cpp、matmul_test.py
```

## 一次编译并验证全部答案

在本目录执行：

```bash
bash run_all.sh
```

脚本自动加载 CANN 环境，依次编译和测试四个工程。每个工程在独立 Python 进程中加载，避免两种绑定方式的同名算子互相干扰。任何一步失败都会停止并返回非零退出码。

可通过环境变量指定环境和构建目录：

```bash
ASCEND_SET_ENV=/usr/local/Ascend/cann/set_env.sh \
PYTHON_BIN=/path/to/python3 \
BUILD_ROOT=/tmp/03.08_build \
BUILD_JOBS=2 \
bash run_all.sh
```

`PYTHON_BIN` 应指向已安装 PyTorch、torch_npu 和 pybind11 的 Python 解释器。

## 单独编译并运行

以下命令在本目录执行，以 Softmax 的 Pybind11 工程为例：

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S softmax/pybind -B build/softmax/pybind \
    -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
    -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/softmax/pybind --parallel 2
python3 softmax/pybind/softmax_test.py --build-dir build/softmax/pybind
```

其余三个工程分别位于 `softmax/torch_library`、`matmul/pybind`、`matmul/torch_library`。对应测试脚本为 `softmax_test.py` 或 `matmul_test.py`，`--build-dir` 传入该工程的实际构建目录。

## Softmax

- 输入、输出：`(128, 128)`，`torch.float32`，连续 NPU Tensor。
- Python 接口：Pybind11 为 `ascendc_softmax.softmax(x)`；Torch Library 为 `torch.ops.ascendc_softmax.softmax(x)`。
- 参考 `softmax_high_performance` 的 Case 0，保留单行循环内 ReduceMax、Duplicate、Sub、Exp、ReduceSum、Duplicate、Div 的 MemBase 计算顺序。
- A2/A3 使用 `GetValue` 读取归约结果，再进行标量广播；为此添加 Vector/Scalar 同步。每次处理一行以控制 UB 占用。
- 校验随机输入、大正数、大负数、常量输入和逐行归一化；对照 CPU PyTorch Softmax，容差为 `rtol=1e-5, atol=1e-6`。

## Matmul

- 输入 A：`(M, K)`；输入 B_T：`(N, K)`；输出 C：`(M, N)`。固定 `M=K=N=8192`，输入输出均为 `torch.float16`、连续 NPU Tensor。
- 计算 `C = A @ B_T.T`，无 Bias，B_T 在物理内存中已按转置后的行优先布局存储。
- Python 接口：Pybind11 为 `ascendc_matmul.matmul(a, bt)`；Torch Library 为 `torch.ops.ascendc_matmul.matmul(a, bt)`。
- 参考样例标注 CANN 9.2.0 及以上；本答案将其中的 `ceil_div` 替换为等价的整数函数，以兼容 CANN 9.1。
- 参考 `matmul_basic_api_high_performance` 的 A2/A3 路径，采用 Scenario 1 的固定参数和 24 个逻辑计算块；保留双缓冲、大包搬运、UnitFlag 及尾块处理。
- 随机输入与完整 CPU FP32 矩阵乘后转换为 FP16 的结果比较，容差为 `rtol=2e-3, atol=2e-3`；非对称置换矩阵用例精确检查 B_T 转置语义。
- 完整矩阵校验会额外使用 CPU 内存并执行一次 CPU 矩阵乘，运行耗时取决于 CPU 性能。

两题还验证输入不被修改、非默认 NPU Stream 上运行，以及错误的设备、shape、dtype、非连续输入能被拒绝。本答案面向前向推理，不实现自动求导。

## 代码来源

算子参考 asc-devkit 中的以下样例，并保留其源码许可声明：

- `examples/01_simd_cpp_api/05_best_practices/02_reg_compute/softmax_high_performance`：Case 0。
- `examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_basic_api_high_performance`：A2/A3 分支。

PyTorch 接入工程参考本仓库 `02_AscendC_basic/src/02.07/kernel_pytorch_call`。答案可独立构建，不依赖外部 asc-devkit 源码目录。

## 实测结果

已在 Ascend 910B4、CANN 9.1.0、Python 3.12.13、PyTorch 2.7.1 和 torch_npu 2.7.1.post10 环境下执行 `run_all.sh`，四个工程均编译成功并通过全部测试。

| 算子 | Pybind11 | Torch Library | 数值检查 |
| --- | --- | --- | --- |
| Softmax | 通过 | 通过 | 四组输入最大绝对误差不超过 `3.58e-7`，归一化检查通过 |
| Matmul | 通过 | 通过 | 随机矩阵全量比较通过 `rtol=2e-3, atol=2e-3`；置换矩阵用例逐元素精确相等 |

Matmul 随机用例的最大绝对误差为 `0.125`，全量结果均满足上述相对与绝对容差组合；该值不表示逐元素精确相等。A3 未在本机实测。
