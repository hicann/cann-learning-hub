#include "moe_token_unpermute_lite_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 16;
static constexpr uint32_t TOP_K = 2;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &expertShape = context->GetInputShape(0)->GetStorageShape();
    const uint32_t totalRows = static_cast<uint32_t>(expertShape.GetDim(0));
    const uint32_t hiddenSize = static_cast<uint32_t>(expertShape.GetDim(1));
    const uint32_t numTokens = totalRows / TOP_K;
    const uint32_t tokensPerCore = (numTokens + BLOCK_DIM - 1) / BLOCK_DIM;
    MoeTokenUnpermuteLiteTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_topK(TOP_K);
    tiling.set_totalRows(totalRows);
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
    const gert::Shape *expert = context->GetInputShape(0);
    const int64_t totalRows = expert->GetDim(0);
    const int64_t hiddenSize = expert->GetDim(1);
    *context->GetOutputShape(0) = gert::Shape({totalRows / optiling::TOP_K, hiddenSize});
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MoeTokenUnpermuteLite : public OpDef {
public:
    explicit MoeTokenUnpermuteLite(const char *name) : OpDef(name)
    {
        Input("expertOut").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("sortedIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("permutedProbs").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(ge::InferShape);
        AICore().SetTiling(optiling::TilingFunc);
        AICore().AddConfig("ascend310b");
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MoeTokenUnpermuteLite);
}
