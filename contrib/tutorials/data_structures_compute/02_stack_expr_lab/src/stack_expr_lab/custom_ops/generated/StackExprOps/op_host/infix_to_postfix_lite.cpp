#include "../op_kernel/infix_to_postfix_lite_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    InfixToPostfixLiteTilingData *tiling = context->GetTilingData<InfixToPostfixLiteTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int32_t totalLen = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++)
        totalLen *= x_shape->GetStorageShape().GetDim(i);

    uint32_t blockDim = 8;
    uint32_t blockLength = totalLen / blockDim;

    const int64_t *exprLenPtr = context->GetAttrs()->GetInt(0);
    uint32_t exprLength = (exprLenPtr != nullptr) ? static_cast<uint32_t>(*exprLenPtr) : blockLength;

    tiling->totalLength = totalLen;
    tiling->blockLength = blockLength;
    tiling->exprLength = exprLength;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
class InfixToPostfixLite : public OpDef {
public:
    explicit InfixToPostfixLite(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("exprLength").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(InfixToPostfixLite);
}
