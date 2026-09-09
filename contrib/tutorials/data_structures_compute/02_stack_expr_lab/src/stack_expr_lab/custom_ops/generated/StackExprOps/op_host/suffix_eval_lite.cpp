#include "../op_kernel/suffix_eval_lite_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SuffixEvalLiteTilingData *tiling = context->GetTilingData<SuffixEvalLiteTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int32_t totalLen = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++)
        totalLen *= x_shape->GetStorageShape().GetDim(i);

    uint32_t blockDim = 8;
    uint32_t blockLength = totalLen / blockDim;

    const int64_t *tokenCountPtr = context->GetAttrs()->GetInt(0);
    uint32_t tokenCount = (tokenCountPtr != nullptr) ? static_cast<uint32_t>(*tokenCountPtr) : blockLength;

    tiling->totalLength = totalLen;
    tiling->blockLength = blockLength;
    tiling->tokenCount = tokenCount;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, 8);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SuffixEvalLite : public OpDef {
public:
    explicit SuffixEvalLite(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("tokenCount").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SuffixEvalLite);
}
