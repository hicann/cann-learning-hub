---
name: ascendc-ops-project
description: Ascend C 算子工程化开发技能。提供从工程创建、编译打包、安装部署到 aclnn API 测试的完整流程，包含 Tiling 模板编程、属性、TBuf、Workspace 使用。触发：创建算子工程、算子打包安装、工程化开发、aclnn API 测试、Tiling 模板编程、属性使用。
---

# Ascend C 算子工程化开发指南

## 触发场景

1. 从零创建标准算子工程
2. 算子编译、打包、安装部署
3. 完整开发流程指导（设计→实现→测试→部署）
4. 使用 aclnn 二段式接口进行测试验证
5. Tiling 模板编程、属性、TBuf、Workspace 使用

---

## ⚠️ 铁律：严格按照算子原型开发

**这是最重要的规则，违反将导致算子无法正确集成到网络中！**

### 什么是算子原型

算子原型定义了算子的：
- 输入名称、类型、数量
- 输出名称、类型、数量
- 属性名称、类型、默认值
- API 签名

### 必须遵守的规则

| 规则 | 说明 | 错误示例 | 正确做法 |
|------|------|----------|----------|
| **输入不能当属性** | 输入是运行时数据，属性是编译时常量 | 把 `axis` 输入改成属性 | 保持 `axis` 为输入，使用 `ValueDepend` 在 Tiling 阶段读取 |
| **属性不能当输入** | 属性值在编译时确定，不能动态变化 | 把 `dim` 属性改成输入张量 | 保持 `dim` 为属性，通过 `GetAttrs()` 获取 |
| **不能增删输入输出** | 必须与原型完全一致 | 删除某个输入 | 保留所有输入，即使不使用 |
| **不能修改数据类型** | dtype 必须与原型一致 | 原型是 float16，改成 float | 按原型支持的数据类型实现 |
| **不能修改参数顺序** | API 参数顺序由原型决定 | 调换输入顺序 | 保持原型定义的顺序 |

### 为什么必须严格遵守

1. **API 签名由原型决定**：`aclnnOpGetWorkspaceSize` 的参数顺序和类型由原型自动生成
2. **网络集成依赖原型**：框架调用算子时按照原型传参
3. **测试代码依赖原型**：测试用例按照原型 API 编写

---

## ⚠️ 铁律：深入理解参考算子

**必须完全理解参考算子的所有行为，包括特殊语义！**

### 为什么这很重要

题目通常会指定参考算子（如 `torch.histc`、`torch.nn.functional.softmax`），要求完全复现其行为。如果对参考算子理解不完整，会导致：
- 测试用例失败
- 边界情况处理错误
- 默认值语义错误

### 关键步骤

1. **查阅官方文档**
   - 不要仅凭直觉理解参数含义
   - 特别注意默认值的特殊语义
   - 理解所有边界情况

2. **理解默认值的真实含义**
   - 默认值可能是一个**标志**，而非有效值
   - 常见模式：
     - `value=0` 表示"自动计算"
     - `value=-1` 表示"使用默认行为"
     - `value=None` 表示"可选参数"

3. **测试边界情况**（如可能）
   - 用参考算子测试特殊参数值
   - 验证自己的理解是否正确

### 实战案例

#### 案例 1：torch.histc 的默认值语义

**题目**：实现 Histogram 算子，参考 `torch.histc`

**错误理解**：
```cpp
// ❌ 认为 min=0.0, max=0.0 是有效范围
// 使用默认范围 [0.0, 1.0]
```

**正确理解**（查阅 PyTorch 文档）：
```python
# torch.histc(input, bins=100, min=0, max=0)
# 如果 min == 0 and max == 0，自动使用 input 的 min 和 max
```

**正确实现**：
```cpp
// ✅ 检测特殊标志
if (min_val == 0.0f && max_val == 0.0f) {
    // 动态计算数据的实际范围
    need_compute_range = true;
}

// 在 Kernel 中：
if (need_compute_range) {
    // 遍历数据计算 min 和 max
    ComputeDataRange();
}
```

#### 案例 2：axis=-1 的特殊含义

**常见模式**：`axis=-1` 表示最后一维

```cpp
// ✅ 正确处理负值
int32_t axis = attrs->GetInt(0);
int32_t rank = shape.GetDimNum();
if (axis < 0) {
    axis = axis + rank;  // 转换为正值
}
```

### 检查清单

- [ ] 查阅参考算子的官方文档
- [ ] 理解所有参数的含义和默认值
- [ ] 识别默认值的特殊语义（是否是标志）
- [ ] 理解边界情况和特殊值处理
- [ ] 用参考算子验证理解（如可能）

### 常见错误案例

```cpp
// ❌ 错误：把 axis 输入当成属性
this->Attr("axis").Int(0);  // 错误！

// ✅ 正确：axis 是输入，使用 ValueDepend 在 Tiling 阶段读取
this->Input("axis")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT32, ge::DT_INT64})
    .ValueDepend(REQUIRED, DependScope::TILING);  // 正确！
```

### 如何获取算子原型

1. **题目描述**：从题目或需求文档获取
2. **框架参考**：参考 PyTorch/TensorFlow 对应算子的原型
3. **已有实现**：参考 CANN 内置算子的原型定义

---

## Part 1: 快速开始 - 完整开发流程

### 开发流程图

```
需求分析 → 原型定义 → 工程生成 → Tiling设计 → Host实现 → Kernel实现 → 编译安装 → 测试验证
```

### 关键步骤速查

| 步骤 | 文件 | 关键点 |
|------|------|--------|
| 1. 原型定义 | `op.json` | 输入输出 dtype 数量一致 |
| 2. 工程生成 | `msopgen` | 指定目标芯片 |
| 3. Tiling结构 | `*_tiling.h` | 包含所有需要传递的参数 |
| 4. Host实现 | `op_host/*.cpp` | Tiling计算、属性读取、ValueDepend |
| 5. Kernel实现 | `op_kernel/*.cpp` | 模板类、dtype判断、多核切分 |
| 6. 编译安装 | `build.sh` | 检查编译错误 |
| 7. 测试验证 | `test.cpp` | 多shape、多dtype、多属性组合 |

---

## Part 2: 常见问题与解决方案（重要！）

### 2.1 ValueDepend - 输入张量在 Tiling 阶段读取

**问题**：当输入是张量（如 axis）而非属性时，Tiling 阶段无法读取其值。

**⚠️ 重要限制**：ValueDepend 输入的 dtype 必须相同，且只能是 **float、int64、bool** 之一！

```
错误示例：ValueDepend input dtype must be the same and must be float, int64, or bool.
```

**解决方案**：
```cpp
// ❌ 错误：int32 不支持 ValueDepend
this->Input("axis")
    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
    .ValueDepend(REQUIRED, DependScope::TILING);  // 编译报错！

// ❌ 错误：dtype 不一致
this->Input("axis")
    .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
    .ValueDepend(REQUIRED, DependScope::TILING);  // 编译报错！

// ✅ 正确：使用 int64，dtype 一致
this->Input("axis")
    .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
    .ValueDepend(REQUIRED, DependScope::TILING);  // 正确！
```

**完整示例**：
```cpp
// Host 侧算子注册
this->Input("axis")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .ValueDepend(REQUIRED, DependScope::TILING);

// Tiling 函数中读取
auto axis_tensor = context->GetInputTensor(1);
int32_t axis = 0;
if (axis_tensor != nullptr) {
    const int64_t* data = axis_tensor->GetData<int64_t>();
    if (data != nullptr) {
        axis = static_cast<int32_t>(data[0]);
    }
}
```

**注意**：配置 ValueDepend 后，API 签名会变化：
- 原：`aclnnOpGetWorkspaceSize(x, axis_tensor, ...)`
- 变：`aclnnOpGetWorkspaceSize(x, axis_intarray, ...)`

### 2.2 多数据类型支持 - 模板实现

**问题**：Kernel 不支持 RTTI（虚函数、继承），无法用运行时多态。

**解决方案**：使用模板 + dtype 参数

```cpp
// Tiling 数据中添加 dtype
struct TilingData {
    uint32_t totalLength;
    int32_t dtype;  // 数据类型
};

// Host 侧设置 dtype
auto input_dtype = context->GetInputDesc(0)->GetDataType();
tiling->dtype = static_cast<int32_t>(input_dtype);

// Kernel 侧模板类
template<typename T>
class KernelOp {
    // 使用 T 类型实现
};

// dtype 值参考
// DT_FLOAT = 0, DT_FLOAT16 = 1, DT_INT8 = 2, DT_INT32 = 3

// Kernel 入口函数
extern "C" __global__ __aicore__ void op_kernel(...)
{
    int32_t dtype = tilingData.dtype;
    if (dtype == 0) {      // DT_FLOAT
        KernelOp<float> op;
        op.Init(...);
        op.Process();
    } else if (dtype == 1) { // DT_FLOAT16
        KernelOp<half> op;
        op.Init(...);
        op.Process();
    }
}
```

### 2.3 Float16 精度问题

**问题**：`half precision operation is not allowed in aicore function`

**解决方案**：转换为 float32 计算

```cpp
template<>
__aicore__ inline void KernelOp<half>::ProcessForward(...)
{
    float sum = 0.0f;  // 使用 float 累积
    for (...) {
        half value_half = xGm.GetValue(offset);
        float value = (float)value_half;  // 转换
        sum = sum + value;
        yGm.SetValue(offset, (half)sum);  // 写回 half
    }
}
```

### 2.4 多核切分问题

**问题**：多核执行时结果错误或竞争。

**解决方案**：

```cpp
// 方案1：单核模式（简单场景）
uint32_t blockDim = 1;
context->SetBlockDim(blockDim);

// 方案2：正确切分（需要仔细设计）
// Tiling 中传递 blockDim
tiling->blockDim = blockDim;

// Kernel 中使用
uint32_t blockIdx_ = AscendC::GetBlockIdx();
uint32_t outerPerCore = (outerLength + blockDim - 1) / blockDim;
uint32_t outerStart = blockIdx_ * outerPerCore;
uint32_t outerEnd = std::min(outerStart + outerPerCore, outerLength);
```

### 2.5 原型定义 dtype 数量问题

**问题**：`msopgen` 要求所有输入的 type 数量必须一致。

**解决方案**：
```json
{
    "input_desc": [
        {"name": "x", "type": ["float16", "float", "int32", "int8"]},
        {"name": "axis", "type": ["int32", "int32", "int32", "int32"]}  // 数量一致
    ]
}
```

---

## Part 3: 创建算子工程

### Step 1: 创建算子原型定义文件

**基础示例**：
```json
[{
    "op": "AddCustom",
    "input_desc": [
        {"name": "x", "param_type": "required", "format": ["ND", "ND"], "type": ["float16", "float"]},
        {"name": "y", "param_type": "required", "format": ["ND", "ND"], "type": ["float16", "float"]}
    ],
    "output_desc": [
        {"name": "z", "param_type": "required", "format": ["ND", "ND"], "type": ["float16", "float"]}
    ]
}]
```

**带属性示例**：
```json
[{
    "op": "Clamp",
    "input_desc": [{"name": "x", "param_type": "required", "format": ["ND"], "type": ["float"]}],
    "output_desc": [{"name": "y", "param_type": "required", "format": ["ND"], "type": ["float"]}],
    "attr": [
        {"name": "min", "type": "float", "param_type": "optional", "default_value": 0},
        {"name": "max", "type": "float", "param_type": "optional", "default_value": 1}
    ]
}]
```

### Step 2: 创建算子工程

```bash
msopgen gen -i op.json -c ai_core-ascend910b -lan cpp -out custom_op
```

### Step 3: 工程目录结构

```
custom_op/
├── op_host/
│   ├── op.cpp              # Host 侧实现（Tiling、形状推导）
│   └── CMakeLists.txt
├── op_kernel/
│   ├── op_tiling.h         # Tiling 数据结构
│   ├── op.cpp              # Kernel 侧实现
│   └── CMakeLists.txt
├── CMakeLists.txt
├── CMakePresets.json
└── build.sh
```

---

## Part 4: Host 侧实现模板

### 4.1 完整 Host 代码模板

```cpp
#include "../op_kernel/op_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    TilingData *tiling = context->GetTilingData<TilingData>();
    
    // 1. 获取输入 shape
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    auto shape_dims = x_shape->GetStorageShape();
    int32_t rank = shape_dims.GetDimNum();
    
    // 2. 计算总元素数
    uint32_t totalLength = 1;
    for (int i = 0; i < rank; i++) {
        totalLength *= shape_dims.GetDim(i);
    }
    
    // 3. 获取属性（如果有）
    auto attrs = context->GetAttrs();
    bool attr1 = false;
    if (attrs != nullptr && attrs->GetBool(0) != nullptr) {
        attr1 = *attrs->GetBool(0);
    }
    
    // 4. 获取输入张量数据（如果配置了 ValueDepend）
    auto input_tensor = context->GetInputTensor(1);
    int32_t input_value = 0;
    if (input_tensor != nullptr) {
        const int32_t* data = input_tensor->GetData<int32_t>();
        if (data != nullptr) input_value = data[0];
    }
    
    // 5. 获取数据类型
    auto input_dtype = context->GetInputDesc(0)->GetDataType();
    
    // 6. 设置 Tiling 数据
    tiling->totalLength = totalLength;
    tiling->dtype = static_cast<int32_t>(input_dtype);
    // ... 其他参数
    
    // 7. 设置核数
    uint32_t coreNum = platform.GetCoreNum();
    uint32_t blockDim = 1;  // 或根据计算量设置
    context->SetBlockDim(blockDim);
    tiling->blockDim = blockDim;
    
    // 8. 设置 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;  // 或设置需要的大小
    
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Op : public OpDef {
public:
    explicit Op(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 可选属性
        this->Attr("attr1").AttrType(OPTIONAL).Bool(false);
        
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Op);
}
```

---

## Part 5: Kernel 侧实现模板

### 5.1 完整 Kernel 代码模板

```cpp
#include "kernel_operator.h"
#include "op_tiling.h"

template<typename T>
class KernelOp {
public:
    __aicore__ inline KernelOp() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t totalLength, uint32_t blockDim)
    {
        this->totalLength = totalLength;
        this->blockDim = blockDim;
        blockIdx_ = AscendC::GetBlockIdx();
        
        xGm.SetGlobalBuffer((__gm__ T *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)y, totalLength);
    }
    
    __aicore__ inline void Process()
    {
        uint32_t lengthPerCore = (totalLength + blockDim - 1) / blockDim;
        uint32_t start = blockIdx_ * lengthPerCore;
        uint32_t end = (start + lengthPerCore < totalLength) ? 
                       (start + lengthPerCore) : totalLength;
        
        for (uint32_t i = start; i < end; i++) {
            T value = xGm.GetValue(i);
            // 计算逻辑
            yGm.SetValue(i, value);
        }
    }

private:
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t totalLength;
    uint32_t blockDim;
    uint32_t blockIdx_;
};

// Float16 特化（需要 float32 中间计算）
template<>
__aicore__ inline void KernelOp<half>::Process()
{
    uint32_t lengthPerCore = (totalLength + blockDim - 1) / blockDim;
    uint32_t start = blockIdx_ * lengthPerCore;
    uint32_t end = (start + lengthPerCore < totalLength) ? 
                   (start + lengthPerCore) : totalLength;
    
    for (uint32_t i = start; i < end; i++) {
        half value_half = xGm.GetValue(i);
        float value = (float)value_half;
        // 使用 float 计算
        yGm.SetValue(i, (half)value);
    }
}

extern "C" __global__ __aicore__ void op_kernel(GM_ADDR x, GM_ADDR y, 
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingData);
    GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tiling);
    
    int32_t dtype = tilingData.dtype;
    
    if (dtype == 0) {  // DT_FLOAT
        KernelOp<float> op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockDim);
        op.Process();
    } else if (dtype == 1) {  // DT_FLOAT16
        KernelOp<half> op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockDim);
        op.Process();
    } else if (dtype == 3) {  // DT_INT32
        KernelOp<int32_t> op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockDim);
        op.Process();
    } else if (dtype == 2) {  // DT_INT8
        KernelOp<int8_t> op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockDim);
        op.Process();
    }
}
```

---

## Part 6: 测试代码模板

### 6.1 完整测试代码框架

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <acl/acl.h>
#include "aclnn/acl_meta.h"
#include "aclnn_op.h"

#define CHECK_ACL(ret) do { \
    if (ret != ACL_SUCCESS) { \
        std::cerr << "ACL 错误: " << ret << " 在第 " << __LINE__ << " 行" << std::endl; \
        return false; \
    } \
} while(0)

// 参考实现
template<typename T>
std::vector<T> ref_impl(const std::vector<T>& input, /* params */) {
    std::vector<T> output(input.size());
    // 实现参考算法
    return output;
}

// 结果比较
template<typename T>
bool compare_result(const std::vector<T>& output, const std::vector<T>& expected, 
                    float rtol = 1e-3, float atol = 1e-3) {
    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs((float)output[i] - (float)expected[i]);
        float tolerance = atol + rtol * std::abs((float)expected[i]);
        if (diff > tolerance) {
            std::cerr << "不匹配在索引 " << i << ": 得到 " << (float)output[i] 
                      << ", 期望 " << (float)expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 辅助函数
std::vector<int64_t> compute_stride(const std::vector<int64_t>& shape) {
    std::vector<int64_t> stride(shape.size());
    stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
}

int64_t compute_total(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (auto d : shape) total *= d;
    return total;
}

// 测试用例
bool test_case(const std::vector<int64_t>& shape, int64_t axis, 
               bool exclusive, bool reverse, const std::string& name) {
    int64_t total = compute_total(shape);
    
    // 准备数据
    std::vector<float> input(total);
    for (int64_t i = 0; i < total; i++) input[i] = (float)i;
    
    auto expected = ref_impl(input, axis, exclusive, reverse, shape);
    
    // 分配设备内存
    float* x_dev = nullptr;
    float* y_dev = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&x_dev, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&y_dev, total * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(x_dev, total * sizeof(float), input.data(), 
                          total * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    
    // 创建张量
    auto stride = compute_stride(shape);
    aclTensor* x_tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, 
                                          stride.data(), 0, ACL_FORMAT_ND, 
                                          shape.data(), shape.size(), x_dev);
    aclIntArray* axis_array = aclCreateIntArray(&axis, 1);
    aclTensor* y_tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, 
                                          stride.data(), 0, ACL_FORMAT_ND, 
                                          shape.data(), shape.size(), y_dev);
    
    // 调用算子
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnOpGetWorkspaceSize(x_tensor, axis_array, exclusive, reverse, 
                                      y_tensor, &workspaceSize, &executor));
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclnnOp(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // 获取结果
    std::vector<float> output(total);
    CHECK_ACL(aclrtMemcpy(output.data(), total * sizeof(float), y_dev, 
                          total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    
    // 验证
    bool pass = compare_result(output, expected);
    std::cout << name << ": " << (pass ? "✓" : "✗") << std::endl;
    
    // 清理
    if (workspace) CHECK_ACL(aclrtFree(workspace));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtFree(x_dev));
    CHECK_ACL(aclrtFree(y_dev));
    
    return pass;
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, 0));
    CHECK_ACL(aclrtSetCurrentContext(context));
    
    int passed = 0, failed = 0;
    
    // 测试不同 shape
    if (test_case({2, 3, 4}, 1, false, false, "basic")) passed++; else failed++;
    if (test_case({128, 256}, 1, false, false, "large_2d")) passed++; else failed++;
    if (test_case({16, 32, 64}, 1, false, false, "large_3d")) passed++; else failed++;
    
    // 测试不同 axis
    if (test_case({2, 3, 4}, 0, false, false, "axis0")) passed++; else failed++;
    if (test_case({2, 3, 4}, -1, false, false, "axis_neg")) passed++; else failed++;
    
    // 测试不同属性
    if (test_case({2, 3, 4}, 1, true, false, "exclusive")) passed++; else failed++;
    if (test_case({2, 3, 4}, 1, false, true, "reverse")) passed++; else failed++;
    
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    
    std::cout << "通过: " << passed << ", 失败: " << failed << std::endl;
    return failed == 0 ? 0 : -1;
}
```

---

## Part 7: 编译打包安装测试

### 7.1 编译安装命令

```bash
# 编译
cd custom_op && bash build.sh

# 安装
./build_out/custom_opp_ubuntu_aarch64.run --install-path=${HOME} --quiet

# 设置环境
source ${HOME}/vendors/customize/bin/set_env.bash
```

### 7.2 测试编译命令

```bash
source ${HOME}/vendors/customize/bin/set_env.bash

g++ -o test_op test_op.cpp \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/include/acl \
    -I${HOME}/vendors/customize/op_api/include \
    -L${ASCEND_HOME_PATH}/aarch64-linux/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    -L${HOME}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64 \
    -lascendcl -lnnopbase -lcust_opapi -lcust_opmaster_rt2.0 \
    -std=c++17 \
    -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64 \
    -Wl,-rpath,${HOME}/vendors/customize/op_api/lib \
    -Wl,-rpath,${HOME}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64

./test_op
```

---

## Part 8: Tiling 模板编程（高级）

### 8.1 什么是 TilingKey

TilingKey 用于区分不同的 kernel 实现分支，编译时会根据不同 TilingKey 形成不同二进制算子 om 文件。

### 8.2 Tiling 模板文件示例

**op_kernel/tiling_key_op.h**:
```cpp
#ifndef TILING_KEY_OP_H
#define TILING_KEY_OP_H
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Op,
ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),
ASCENDC_TPL_UINT_DECL(TILE_NUM, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_MIX, 1, 2, 4, 8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)),
);
#endif
```

---

## Part 9: Workspace 和 TBuf

### 9.1 Workspace 设置

```cpp
// Host 侧
auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
size_t userWorkspaceSize = 256 * 4;
size_t systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
```

### 9.2 TBuf 使用

```cpp
class KernelOp {
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
};

__aicore__ inline void Init() {
    pipe.InitBuffer(tmpBuf, 256 * sizeof(float));
}

__aicore__ inline void Compute() {
    AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
    // 使用 tmp，不要 EnQue/DeQue/FreeTensor
}
```

---

## Part 10: 数据类型枚举值

| 枚举名 | 值 | C++类型 |
|--------|-----|---------|
| DT_FLOAT | 0 | float |
| DT_FLOAT16 | 1 | half |
| DT_INT8 | 2 | int8_t |
| DT_INT32 | 3 | int32_t |
| DT_INT64 | 4 | int64_t |
| DT_UINT8 | 5 | uint8_t |
| DT_BOOL | 6 | bool |

---

## Part 11: 检查清单

### ⚠️ 算子原型合规检查（最重要！）
- [ ] **输入/输出数量与原型一致**
- [ ] **输入/输出名称与原型一致**
- [ ] **数据类型与原型一致**
- [ ] **属性名称、类型、默认值与原型一致**
- [ ] **没有把输入当成属性**
- [ ] **没有把属性当成输入**
- [ ] **API 调用参数顺序正确**

### 工程创建
- [ ] 算子原型 json 文件正确（dtype 数量一致）
- [ ] 工程目录结构完整

### Host 侧
- [ ] Tiling 数据结构包含所有参数
- [ ] ValueDepend 配置正确（如需要）
- [ ] 属性获取索引正确
- [ ] blockDim 设置合理

### Kernel 侧
- [ ] 使用模板支持多数据类型
- [ ] Float16 使用 float32 中间计算
- [ ] 多核切分逻辑正确
- [ ] 无虚函数/继承

### 测试验证
- [ ] 多种 shape 测试
- [ ] 多种 dtype 测试
- [ ] 多种属性组合测试
- [ ] 边界情况测试

---

## Part 12: API 最佳实践速查

详细内容见 [references/api_best_practices.md](references/api_best_practices.md)

### 核心要点

1. **API 黑名单**：禁止使用 `GlobalTensor::SetValue()` 和 `GetValue()`
2. **DataCopy vs DataCopyPad**：优先使用 DataCopyPad，自动处理非对齐
3. **TBuf vs TQue**：MTE 搬运用 TQue，纯计算用 TBuf
4. **Double Buffer**：在 `InitBuffer` 的 `num` 参数中设置，与模板 `depth` 无关
5. **repeatTime 限制**：uint8_t 类型最大 255，超过需分批处理

---

## Part 13: Tiling 设计速查

详细内容见 [references/tiling_design.md](references/tiling_design.md)

### 算子分类

| 类别 | 特征 | 典型算子 | Tiling 复杂度 |
|------|------|---------|--------------|
| Elementwise | Shape相同，逐元素独立 | Sin, Cos, Abs | ⭐ 简单 |
| Reduction | 沿轴归约 | Softmax, LayerNorm | ⭐⭐⭐ 复杂 |
| Broadcast | Shape不同，需广播 | Add, Mul | ⭐⭐ 中等 |

### 通用设计要素

所有算子必须完成：
1. **多核切分**：任务分配给多个 AI Core
2. **UB 切分**：单次处理数据量（A2/A3: 192KB）
3. **Buffer 规划**：输入/输出/中间 buffer
4. **分支覆盖**：dtype/shape/对齐/边界

---

## Part 14: 精度验证标准速查

详细内容见 [references/precision_standard.md](references/precision_standard.md)

### 精度标准选择

- **浮点计算类**：MERE/MARE Threshold（最常用）
- **整数计算类**：精确匹配
- **非计算类**：精确匹配

### 浮点精度阈值

| 数据类型 | Threshold | 数值 |
|---------|-----------|------|
| FLOAT16 | 2^-10 | 0.000977 |
| FLOAT32 | 2^-13 | 0.000122 |
| BFLOAT16 | 2^-7 | 0.00781 |

### 通过标准

1. MERE < Threshold
2. MARE < 10 * Threshold

---

## 参考资源

### 内部文档
- [API 最佳实践速查](references/api_best_practices.md) - API 使用规范、黑名单、常见错误
- [Tiling 设计速查](references/tiling_design.md) - 算子分类、设计要素、模板
- [精度验证标准速查](references/precision_standard.md) - 精度标准、验证方法、测试设计
- [CMake 配置指南](references/cmake_guide.md) - CMakeLists.txt 和 CMakePresets.json 配置

### 完整示例
- [AddCustom 示例](examples/add_custom_example.md) - 简单逐元素算子
- [Clamp 示例](examples/clamp_example.md) - 带属性算子

### 代码模板
- `templates/op_host_template.cpp` - Host 侧实现模板
- `templates/op_kernel_template.cpp` - Kernel 侧实现模板
- `templates/tiling_header_template.h` - Tiling 数据结构模板
- `templates/test_template.cpp` - 测试代码模板
- `templates/build.sh.template` - 编译脚本模板
- `templates/CMakeLists.txt.template` - CMakeLists.txt 模板
- `templates/CMakePresets.json.template` - CMakePresets.json 模板
