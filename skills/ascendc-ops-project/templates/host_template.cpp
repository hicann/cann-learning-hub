/**
 * Ascend C 算子 Host 侧代码模板
 * 
 * 使用方法：
 * 1. 替换 "Op" 为实际算子名
 * 2. 修改 TilingData 结构体
 * 3. 实现 TilingFunc 逻辑
 * 4. 修改算子注册部分
 */

#include "../op_kernel/op_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    OpTilingData *tiling = context->GetTilingData<OpTilingData>();
    
    // ==================== 1. 获取输入 shape ====================
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    auto shape_dims = x_shape->GetStorageShape();
    int32_t rank = shape_dims.GetDimNum();
    
    // ==================== 2. 计算总元素数 ====================
    uint32_t totalLength = 1;
    for (int i = 0; i < rank; i++) {
        totalLength *= shape_dims.GetDim(i);
    }
    
    // ==================== 3. 获取属性（如果有） ====================
    auto attrs = context->GetAttrs();
    bool attr_bool = false;
    int32_t attr_int = 0;
    float attr_float = 0.0f;
    
    if (attrs != nullptr) {
        // Bool 属性
        if (attrs->GetBool(0) != nullptr) {
            attr_bool = *attrs->GetBool(0);
        }
        // Int 属性
        if (attrs->GetInt(1) != nullptr) {
            attr_int = *attrs->GetInt(1);
        }
        // Float 属性
        if (attrs->GetFloat(2) != nullptr) {
            attr_float = *attrs->GetFloat(2);
        }
    }
    
    // ==================== 4. 获取输入张量数据（ValueDepend） ====================
    // 如果配置了 ValueDepend，可以在 Tiling 阶段读取输入张量的值
    // auto input_tensor = context->GetInputTensor(1);
    // int32_t input_value = 0;
    // if (input_tensor != nullptr) {
    //     auto dtype = input_tensor->GetDataType();
    //     if (dtype == ge::DT_INT32) {
    //         const int32_t* data = input_tensor->GetData<int32_t>();
    //         if (data != nullptr) input_value = data[0];
    //     } else if (dtype == ge::DT_INT64) {
    //         const int64_t* data = input_tensor->GetData<int64_t>();
    //         if (data != nullptr) input_value = static_cast<int32_t>(data[0]);
    //     }
    // }
    
    // ==================== 5. 获取数据类型 ====================
    auto input_dtype = context->GetInputDesc(0)->GetDataType();
    
    // ==================== 6. 计算 Tiling 参数 ====================
    // TODO: 根据算子逻辑计算需要的参数
    
    // ==================== 7. 设置 Tiling 数据 ====================
    tiling->totalLength = totalLength;
    tiling->dtype = static_cast<int32_t>(input_dtype);
    // tiling->其他参数 = ...;
    
    // ==================== 8. 设置核数 ====================
    uint32_t coreNum = platform.GetCoreNum();
    uint32_t blockDim = 1;  // 简单场景使用单核
    // 或者根据计算量设置：blockDim = std::min(outerLength, coreNum);
    context->SetBlockDim(blockDim);
    tiling->blockDim = blockDim;
    
    // ==================== 9. 设置 workspace ====================
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    // 如果需要 workspace:
    // size_t userWorkspaceSize = ...;
    // size_t systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    // currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    currentWorkspace[0] = 0;  // 不需要 workspace
    
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// ==================== 形状推导 ====================

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;  // 输出形状与输入相同
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);  // 输出类型与输入相同
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

// ==================== 算子注册 ====================

namespace ops {

class Op : public OpDef {
public:
    explicit Op(const char* name) : OpDef(name)
    {
        // 输入 x
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 如果有更多输入（如 axis），需要配置 ValueDepend
        // this->Input("axis")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        //                ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        //              ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        //                          ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .ValueDepend(REQUIRED, DependScope::TILING);
        
        // 输出 y
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 可选属性
        this->Attr("attr_bool").AttrType(OPTIONAL).Bool(false);
        this->Attr("attr_int").AttrType(OPTIONAL).Int(0);
        this->Attr("attr_float").AttrType(OPTIONAL).Float(0.0f);
        
        // 设置形状推导和数据类型推导
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        
        // 设置 Tiling 函数和目标芯片
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Op);

} // namespace ops
