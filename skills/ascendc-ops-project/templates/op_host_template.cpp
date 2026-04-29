#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/${OPERATOR_NAME}_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 获取Tiling数据结构
    auto tiling = context->GetTilingData<${OPERATOR_CLASS}TilingData>();
    
    // 2. 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = platform.GetCoreNum();
    
    // 3. 获取输入输出shape
    auto input_shape = context->GetInputShape(0);
    uint32_t inputTotalElements = input_shape->GetStorageShape().GetShapeSize();
    
    auto output_shape = context->GetOutputShape(0);
    uint32_t totalElements = output_shape->GetStorageShape().GetShapeSize();
    
    // 4. 获取属性（根据实际情况修改）
    // auto attrs = context->GetAttrs();
    // auto param1 = attrs->GetListInt(0)->GetData();
    // auto param2 = *attrs->GetInt(1);
    
    // 5. 设置Tiling数据
    tiling->totalElements = totalElements;
    tiling->inputTotalElements = inputTotalElements;
    // ... 设置其他参数 ...
    
    // 6. 计算切分策略（简单情况：单核）
    uint32_t blockNum = 1;
    if (totalElements > 8192) {
        blockNum = std::min(aivNum, (totalElements + 8191) / 8192);
    }
    
    uint32_t elementsPerCore = (totalElements + blockNum - 1) / blockNum;
    uint32_t tailElements = totalElements - elementsPerCore * (blockNum - 1);
    
    tiling->blockNum = blockNum;
    tiling->numPerCore = elementsPerCore;
    tiling->tailNumLastCore = tailElements;
    
    // 7. 设置blockDim（必须！）
    context->SetBlockDim(blockNum);
    
    // 8. 设置workspace（可选）
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    // TODO: 实现shape推导
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // TODO: 实现数据类型推导
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ${OPERATOR_CLASS} : public OpDef {
public:
    explicit ${OPERATOR_CLASS}(const char *name) : OpDef(name) {
        // 输入定义（根据实际情况修改）
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 输出定义
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        // 属性定义（根据实际情况添加）
        // this->Attr("param1")
        //     .AttrType(REQUIRED)
        //     .ListInt();
        // this->Attr("param2")
        //     .AttrType(REQUIRED)
        //     .Int();
        
        // 注册函数
        this->SetInferShape(ge::InferShape)
              .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(${OPERATOR_CLASS});
} // namespace ops
