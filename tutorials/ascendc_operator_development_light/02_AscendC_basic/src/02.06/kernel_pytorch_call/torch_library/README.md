# torch.library 调用方式

## 概述

本样例使用 PyTorch 的 torch.library 机制将 `add_custom` Kernel 注册为自定义算子，Python 侧加载动态库后通过 `torch.ops` 调用。

## 目录结构

```
├── CMakeLists.txt        // 编译工程文件
├── torch_binding.cpp     // torch.library 封装与算子注册
├── add_custom_test.py    // PyTorch 调用与精度校验脚本
└── README.md             // 样例说明文档
```

> Kernel 实现位于上级目录 `../common/add_custom.asc`。

## 样例说明

- `torch_binding.cpp` 中 `ascendc_add` 通过 `c10_npu::getCurrentNPUStream()` 获取当前 NPU stream，并调用 `add_custom_impl` 发起 Kernel 执行。
- `TORCH_LIBRARY(ascendc_ops, m)` 定义算子 namespace 与 schema，`TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)` 将 NPU 实现注册到 `PrivateUse1` 调度键。
- `add_custom_test.py` 中通过 `torch.ops.load_library(...)` 加载动态库，调用 `torch.ops.ascendc_ops.ascendc_add(...)`，并与 `torch.add` 结果对比校验。

## 编译运行

```bash
source ${install_path}/cann/set_env.sh # `${install_path}` 为CANN包安装目录，未指定安装目录时默认安装至 `/usr/local/Ascend` 下。

mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
make -j
python3 ../add_custom_test.py    # 在 build 目录下执行
```

精度对比成功时输出 `OK`。
