# PyTorch框架下Kernel直调样例

## 概述

本目录基于同一个 Ascend C `add_custom` Kernel，展示自定义算子接入 PyTorch 后进行 Kernel 直调的两种方式。

## 目录结构

```
├── common
│   └── add_custom.asc        // 可复用的 Ascend C Kernel（含 add_custom_impl 入口）
├── pybind                    // pybind11 调用方式
└── torch_library             // torch.library 调用方式
```

## 样例列表

| 目录 | 说明 |
| --- | --- |
| [pybind](./pybind) | 使用 pybind11 把 Kernel 封装成 Python 模块函数 |
| [torch_library](./torch_library) | 使用 torch.library 把 Kernel 注册为 PyTorch 自定义算子 |
