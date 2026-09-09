#include "moe_top_k_lite_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 16;  // 910B 示例值；310B 由 build_ops.sh 改为 8
static constexpr uint32_t TOP_K = 2;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Shape &shape = context->GetInputShape(0)->GetStorageShape();
    const uint32_t numTokens = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t numExperts = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t blockDim = BLOCK_DIM;
    const uint32_t tokensPerCore = (numTokens + blockDim - 1) / blockDim;

    MoeTopKLiteTilingData tiling;
    tiling.set_numTokens(numTokens);
    tiling.set_numExperts(numExperts);
    tiling.set_topK(TOP_K);
    tiling.set_tokensPerCore(tokensPerCore);
    context->SetBlockDim(blockDim);
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
    const gert::Shape *xShape = context->GetInputShape(0);
    const int64_t tokens = xShape->GetDim(0);
    *context->GetOutputShape(0) = gert::Shape({tokens, optiling::TOP_K});
    *context->GetOutputShape(1) = gert::Shape({tokens, optiling::TOP_K});
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MoeTopKLite : public OpDef {
public:
    explicit MoeTopKLite(const char *name) : OpDef(name)
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
OP_ADD(MoeTopKLite);
}
