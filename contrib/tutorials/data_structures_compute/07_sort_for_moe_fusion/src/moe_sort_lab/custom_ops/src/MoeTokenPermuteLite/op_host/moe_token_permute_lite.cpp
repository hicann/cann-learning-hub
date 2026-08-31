#include "moe_token_permute_lite_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 16;
static constexpr uint32_t TOP_K = 2;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &tokenShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape &orderShape = context->GetInputShape(1)->GetStorageShape();
    const uint32_t numTokens = static_cast<uint32_t>(tokenShape.GetDim(0));
    const uint32_t hiddenSize = static_cast<uint32_t>(tokenShape.GetDim(1));
    const uint32_t totalRows = static_cast<uint32_t>(orderShape.GetShapeSize());
    const uint32_t rowsPerCore = (totalRows + BLOCK_DIM - 1) / BLOCK_DIM;
    MoeTokenPermuteLiteTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_topK(TOP_K);
    tiling.set_totalRows(totalRows);
    tiling.set_rowsPerCore(rowsPerCore);
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
    const gert::Shape *tokens = context->GetInputShape(0);
    const int64_t totalRows = context->GetInputShape(1)->GetShapeSize();
    const int64_t hiddenSize = tokens->GetDim(1);
    *context->GetOutputShape(0) = gert::Shape({totalRows, hiddenSize});
    *context->GetOutputShape(1) = gert::Shape({totalRows});
    *context->GetOutputShape(2) = gert::Shape({totalRows});
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MoeTokenPermuteLite : public OpDef {
public:
    explicit MoeTokenPermuteLite(const char *name) : OpDef(name)
    {
        Input("tokens").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("sortedOrder").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("probs").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("permutedTokens").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("sortedIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("permutedProbs").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(ge::InferShape);
        AICore().SetTiling(optiling::TilingFunc);
        AICore().AddConfig("ascend310b");
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MoeTokenPermuteLite);
}
