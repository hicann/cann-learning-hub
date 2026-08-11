#include "moe_sort_quick_sort_lite_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 16;
static constexpr uint32_t TOP_K = 2;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &shape = context->GetInputShape(0)->GetStorageShape();
    const uint32_t numTokens = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t numExperts = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t tokensPerCore = (numTokens + BLOCK_DIM - 1) / BLOCK_DIM;
    MoeSortQuickSortLiteTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_numExperts(numExperts);
    tiling.set_topK(TOP_K);
    tiling.set_tokensPerCore(tokensPerCore);
    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const int64_t tokens = context->GetInputShape(0)->GetDim(0);
    *context->GetOutputShape(0) = gert::Shape({tokens, optiling::TOP_K});
    *context->GetOutputShape(1) = gert::Shape({tokens, optiling::TOP_K});
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MoeSortQuickSortLite : public OpDef {
public:
    explicit MoeSortQuickSortLite(const char *name) : OpDef(name)
    {
        Input("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("topkIndicesOut").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("topkProbsOut").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(ge::InferShape);
        AICore().SetTiling(optiling::TilingFunc);
        AICore().AddConfig("ascend310b");
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MoeSortQuickSortLite);
}
