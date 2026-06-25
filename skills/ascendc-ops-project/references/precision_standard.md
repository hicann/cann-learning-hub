# 算子精度验证标准速查

从 `ops-precision-standard` skill 整合的核心内容。

---

## 精度标准选择流程

```
随机数生成算子？
  ├─ 是 → 随机数生成类标准
  └─ 否 → 包含数值计算？
           ├─ 否 → 非计算类标准
           └─ 是 → 检查输入输出dtype
                    ├─ 均为整型 → 整数计算类标准
                    ├─ 整型↔浮点 → 量化计算类标准
                    └─ 均为浮点 → 浮点计算类标准（最常用）
```

---

## 浮点计算类算子精度标准（社区标准）

### 适用场景

**适用算子类型**：浮点计算类算子（所有使用浮点数进行数值计算的算子）

**精度指标**：MERE/MARE Threshold

### 各数据类型阈值

| 数据类型 | Threshold | 数值 |
|---------|-----------|------|
| **FLOAT16** | 2^-10 | 约0.000977 |
| **BFLOAT16** | 2^-7 | 约0.00781 |
| **FLOAT32** | 2^-13 | 约0.000122 |
| **HiFLOAT32** | 2^-11 | 约0.000488 |
| **FLOAT8 E4M3** | 2^-3 | 约0.125 |
| **FLOAT8 E5M2** | 2^-2 | 约0.25 |

### 判定条件

**通过标准**：
1. MERE < Threshold
2. MARE < 10 * Threshold

### 误差指标定义

**平均相对误差（MERE）**：
```
MERE = avg(|actual - golden| / (|golden| + 1e-7))
```

**最大相对误差（MARE）**：
```
MARE = max(|actual - golden| / (|golden| + 1e-7))
```

---

## 精度验证代码模板

### C++ 测试代码

```cpp
#include <cmath>
#include <vector>
#include <iostream>

template<typename T>
bool compare_result(const std::vector<T>& output, const std::vector<T>& expected, 
                   float rtol = 1e-3, float atol = 1e-3) {
    float mere_sum = 0.0f;
    float mare_max = 0.0f;
    
    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs((float)output[i] - (float)expected[i]);
        float rel_error = diff / (std::abs((float)expected[i]) + 1e-7f);
        mere_sum += rel_error;
        mare_max = std::max(mare_max, rel_error);
    }
    
    float mere = mere_sum / output.size();
    float mare = mare_max;
    
    // 根据数据类型选择阈值
    float threshold = 0.000122f;  // FP32
    if constexpr (std::is_same<T, half>::value) {
        threshold = 0.000977f;  // FP16
    }
    
    bool mere_pass = mere < threshold;
    bool mare_pass = mare < 10 * threshold;
    
    if (!mere_pass || !mare_pass) {
        std::cerr << "精度不达标: MERE=" << mere << ", MARE=" << mare 
                  << ", threshold=" << threshold << std::endl;
        return false;
    }
    
    return true;
}
```

### Python 验证脚本

```python
import numpy as np

def check_precision_threshold(npu_output, golden_output):
    """检查精度是否达标"""
    
    # 根据数据类型选择阈值
    dtype_thresholds = {
        np.float16: 2**-10,      # 0.000977
        np.float32: 2**-13,      # 0.000122
        np.bfloat16: 2**-7,      # 0.00781
    }
    
    threshold = dtype_thresholds.get(npu_output.dtype, 2**-13)
    mare_threshold = 10 * threshold
    
    # 计算误差
    diff = np.abs(npu_output - golden_output)
    rel_error = diff / (np.abs(golden_output) + 1e-7)
    
    mere = np.mean(rel_error)
    mare = np.max(rel_error)
    
    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold
    is_pass = mere_pass and mare_pass
    
    return {
        'is_pass': is_pass,
        'mere': float(mere),
        'mare': float(mare),
        'threshold': float(threshold),
        'mare_threshold': float(mare_threshold),
        'mere_pass': mere_pass,
        'mare_pass': mare_pass,
    }

# 使用示例
npu_output = run_operator_on_npu()
golden_output = run_reference_on_cpu()

result = check_precision_threshold(npu_output, golden_output)
if not result['is_pass']:
    print(f"精度不达标: MERE={result['mere']:.6f}, MARE={result['mare']:.6f}")
```

---

## 特殊场景处理

### 小值域问题

当数据值域很小时（如接近0），相对误差会放大。

**解决方案**：
```python
# 使用绝对误差 + 相对误差的组合
tolerance = atol + rtol * np.abs(golden_output)
is_pass = np.all(diff < tolerance)
```

### INF/NAN 处理

```python
# 检查 INF/NAN
if np.any(np.isinf(npu_output)) or np.any(np.isnan(npu_output)):
    print("输出包含 INF/NAN")
    is_pass = False

# 检查 INF/NAN 位置是否一致
inf_match = np.equal(np.isinf(npu_output), np.isinf(golden_output))
nan_match = np.equal(np.isnan(npu_output), np.isnan(golden_output))
if not (np.all(inf_match) and np.all(nan_match)):
    print("INF/NAN 位置不匹配")
    is_pass = False
```

### FP16 精度问题

FP16 精度较低，容易出现累积误差。

**解决方案**：
- 在 Kernel 中使用 FP32 中间计算
- 放宽 FP16 的精度阈值

---

## 整数计算类算子精度标准

### 适用场景

输入输出均为整型的算子（如整数加减乘除、位运算等）

### 验证标准

**精确匹配**：输出必须与标杆完全一致

```python
is_pass = np.array_equal(npu_output, golden_output)
```

---

## 非计算类算子精度标准

### 适用场景

不涉及数值计算的算子（如 Reshape、Transpose、Concat、Split 等）

### 验证标准

**精确匹配**：输出必须与标杆完全一致

```python
is_pass = np.array_equal(npu_output, golden_output)
```

---

## 测试用例设计建议

### Shape 覆盖

- 小 shape：[2, 3], [4, 5, 6]
- 中等 shape：[32, 64], [16, 32, 64]
- 大 shape：[1024, 1024], [256, 512, 512]
- 边界 shape：[1], [1, 1, 1]

### 数据类型覆盖

- FP32（默认）
- FP16（常见）
- BF16（可选）
- INT32（如适用）

### 数值范围覆盖

- 正常值：随机生成
- 边界值：0, 1, -1, 最大值, 最小值
- 特殊值：INF, NAN, 非常小的值

### 属性组合覆盖

- 不同 axis 值
- 不同属性组合（如 exclusive, reverse）
- 边界属性值

---

## 检查清单

### 精度验证完成检查

- [ ] 确定算子类别（浮点/整数/非计算）
- [ ] 选择正确的精度标准
- [ ] 实现参考算法（Golden）
- [ ] 设计测试用例（shape/dtype/数值/属性）
- [ ] 运行测试并收集结果
- [ ] 分析失败用例，定位问题
- [ ] 所有测试用例通过

### 测试报告内容

- 总测试用例数
- 通过/失败数量
- 失败用例详情（shape, dtype, 误差值）
- 性能数据（可选）
