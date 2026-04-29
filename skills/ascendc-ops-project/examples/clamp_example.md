# 完整示例：带 Tiling 模板、属性、Workspace 的 Clamp 算子

## 1. 算子原型定义

**op.json**:
```json
[{
    "op": "Clamp",
    "input_desc": [{
        "name": "x",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int32", "float"]
    }],
    "output_desc": [{
        "name": "y",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int32", "float"]
    }],
    "attr": [{
        "name": "min",
        "type": "float",
        "param_type": "optional",
        "default_value": 0
    }]
}]
```

## 2. 创建工程

```bash
msopgen gen -i op.json -c ai_core-ascend910b1 -lan cpp -out custom_op
```

## 3. Tiling 模板文件

**op_kernel/tiling_key_clamp.h**:
```cpp
#ifndef TILING_KEY_CLAMP_H
#define TILING_KEY_CLAMP_H
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Clamp,
ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_INT32, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)), 
ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_INT32, C_DT_FLOAT, ASCENDC_TPL_OUTPUT(0)),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT32),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_INT32),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT),
    ),
);
#endif
```

## 4. Tiling 结构体

**op_kernel/clamp_tiling.h**:
```cpp
#ifndef CLAMP_TILING_H
#define CLAMP_TILING_H
#include <cstdint>

struct ClampTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
    float min;  // 属性值
};
#endif
```

## 5. Host 侧实现

**op_host/clamp.cpp**:
```cpp
#include "../op_kernel/clamp_tiling.h"
#include "register/op_def_registry.h"
#include "../op_kernel/tiling_key_clamp.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ClampTilingData *tiling = context->GetTilingData<ClampTilingData>();
    
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    
    // 获取数据类型用于模板参数
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetOutputDesc(0)->GetDataType();
    uint32_t D_T_X = static_cast<int>(dtype_x);
    uint32_t D_T_Y = static_cast<int>(dtype_y);
   
    // 获取属性值
    float min_value = *context->GetAttrs()->GetFloat(0);
    
    // 设置 Tiling 数据
    tiling->totalLength = totalLength;
    tiling->tileNum = 8;
    tiling->min = min_value;
    context->SetBlockDim(8);
    
    // 配置模板参数
    ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y); 
    
    // 设置 Workspace
    size_t userWorkspaceSize = 256 * 4;  // 用户 workspace
    size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();  // 系统 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Clamp : public OpDef {
public:
    explicit Clamp(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min").Float();  // 注册属性
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Clamp);
}
```

## 6. Kernel 侧实现

**op_kernel/clamp.cpp**:
```cpp
#include "kernel_operator.h"
#include "clamp_tiling.h"
#include "tiling_key_clamp.h"

constexpr int32_t BUFFER_NUM = 1;

template <class dtypeX, class dtypeY>
class KernelClamp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, 
                                uint32_t totalLength, uint32_t tileNum, float min)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->min = min;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        tmpGm.SetGlobalBuffer((__gm__ float *)workspace);  // 使用 workspace
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueTmp, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum * BUFFER_NUM; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // ... CopyIn, Compute, CopyOut 实现 ...

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY, outQueueTmp;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<float> tmpGm;  // workspace
    uint32_t blockLength, tileNum, tileLength;
    float min;
};

// 模板参数核函数（去掉 extern "C"）
template <typename D_T_X, typename D_T_Y>
__global__ __aicore__ void clamp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClampTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClampTilingData, tilingData, tiling);
    
    KernelClamp<D_T_X, D_T_Y> op;
    op.Init(x, y, workspace, tilingData.totalLength, tilingData.tileNum, tilingData.min);

    // 根据模板参数选择处理逻辑
    if constexpr (std::is_same_v<D_T_X, float> && std::is_same_v<D_T_Y, float>) {
        op.Process();  // float 类型处理
    } else {
        op.Process2(); // int32 类型处理
    }
}
```

## 7. 测试代码

```cpp
int main()
{
    // ... 初始化 ...
    
    // 设置属性值
    double min = 3.0;
    
    // 调用带属性的 aclnn API
    aclnnClampGetWorkspaceSize(inputX, min, outputY, &workspaceSize, &executor);
    aclnnClamp(workspaceAddr, workspaceSize, executor, stream);
    
    // ... 同步、获取结果、验证 ...
}
```
