# pybind11 调用方式

## 概述

本样例使用 pybind11 将 `add_custom` Kernel 封装成 Python 模块函数，Python 侧 `import` 后直接调用。

## 目录结构

```
├── CMakeLists.txt        // 编译工程文件
├── pybind_binding.cpp    // pybind11 封装与模块绑定
├── add_custom_test.py    // PyTorch 调用与精度校验脚本
└── README.md             // 样例说明文档
```

> Kernel 实现位于上级目录 `../common/add_custom.asc`。

## 样例说明

- `pybind_binding.cpp` 中 `ascendc_add` 通过 `c10_npu::getCurrentNPUStream()` 获取当前 NPU stream，并调用 `add_custom_impl` 发起 Kernel 执行；`PYBIND11_MODULE(ascendc_ops, m)` 将其导出为 Python 模块 `ascendc_ops` 的 `ascendc_add` 函数。
- `add_custom_test.py` 中 `import ascendc_ops` 后调用 `ascendc_ops.ascendc_add(...)`，并与 `torch.add` 结果对比校验。

## 编译运行

```bash
source ${install_path}/cann/set_env.sh # source ${install_path}/cann/set_env.sh # `${install_path}` 为CANN包安装目录，未指定安装目录时默认安装至 `/usr/local/Ascend` 下。

mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
make -j
python3 ../add_custom_test.py    # 在 build 目录下执行
```

精度对比成功时输出 `OK`。
