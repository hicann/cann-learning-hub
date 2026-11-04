# 自定义 AddRmsNorm 算子

本目录提供一个最小可用的PyTorch自定义AddRmsNorm算子（仅支持BFLOAT16类型）。用于演示PyTorch自定义算子开发、注册、调用的全过程。

## 功能说明

RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNorm算子将RmsNorm前的Add算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## 文件结构

```text
add_rms_norm_custom_bfloat16/
├── add_rms_norm_custom.asc        # device kernel + host 启动器
├── torch_binding.cpp              # torch.library 注册
├── CMakeLists.txt                 # 编译 libascendc_kernels.so 与 libascendc_ops.so
└── test_add_rms_norm_custom.py    # 直调测试
```

## 参数说明

| 参数名 | 输入/输出 | 数据类型 | 描述 |
| --- | --- | --- | --- |
| x1/x2 | 输入 | BFLOAT16 | 表示用于Add计算的第一个输入。对应公式中的`x1/x2`。 |
| gamma | 输入 | BFLOAT16 | 表示RmsNorm的缩放因子（权重）。对应公式中的`gamma`。 |
| epsilon | 输入 | FLOAT32 | 表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。 |

## 返回值说明
| 参数名 | 数据类型 | 描述 |
| --- | --- | --- |
| y | BFLOAT16 | 表示最后的输出。对应公式中的`RmsNorm(x)`。 |
| rstd | FLOAT32 | 表示归一化后的标准差的倒数。对应公式中`Rms(x)`的倒数。 |
| x | BFLOAT16 | 表示Add计算的结果。对应公式中的`x`。 |

## 构建与运行

```bash
cd add_rms_norm_custom_bfloat16
mkdir -p build && cd build
cmake ..
make -j
python3 ../test_add_rms_norm_custom.py
```
