#include "top_k_reduce_lite_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
/*
 * 【跨芯片适配说明 — BLOCK_DIM 是写死的吗？】
 *
 * 当前 BLOCK_DIM=8 是硬编码的，原因是 CANN 8.0 SDK 没有动态获取核数的 API。
 *
 * 【推荐写法】在 CANN 8.2+ 中，应使用 PlatformAscendC 动态获取：
 *   #include "platform/platform_infos_lite_def.h"
 *   platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
 *   uint32_t blockDim = platform.GetCoreNumAiv();  // 310B=8, 910B=20+, A2=24, A3=24
 *   context->SetBlockDim(blockDim);
 *
 * 【核心原则】跨芯片适配的关键是：
 *   1. BLOCK_DIM / blockDim 在 Host/Tiling 侧计算，不要在 Kernel 侧硬编码
 *   2. totalLength 从输入 shape 动态计算，不要写死
 *   3. Kernel 只按 tiling 参数执行，不关心具体芯片型号
 */
static constexpr uint32_t BLOCK_DIM = 8;  // ⚠️ 310B=8, 910B=20+, A2=24, A3=24
static constexpr uint32_t TILE_LEN = 1024;
static constexpr uint32_t TOP_K = 4;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TopKReduceLiteTilingData tiling;
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::Shape &shape = xShape->GetStorageShape();
    uint32_t totalLength = static_cast<uint32_t>(shape.GetShapeSize());

    uint32_t blockDim = BLOCK_DIM;
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
    uint32_t tileLength = TILE_LEN;
    uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;
    uint32_t lastTileLength = blockLength - (tileNum - 1) * tileLength;
    if (tileNum == 0) { tileNum = 1; lastTileLength = 0; }

    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_topK(TOP_K);
    tiling.set_blockLength(blockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(tileNum);
    tiling.set_lastTileLength(lastTileLength);
    tiling.set_blockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    int64_t N = xShape->GetDim(0);
    int64_t K = 4;
    gert::Shape *valuesShape = context->GetOutputShape(0);
    gert::Shape *indicesShape = context->GetOutputShape(1);
    *valuesShape = gert::Shape({static_cast<int64_t>(optiling::BLOCK_DIM) * K});
    *indicesShape = gert::Shape({static_cast<int64_t>(optiling::BLOCK_DIM) * K});
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TopKReduceLite : public OpDef {
public:
    explicit TopKReduceLite(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("values").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(TopKReduceLite);
}