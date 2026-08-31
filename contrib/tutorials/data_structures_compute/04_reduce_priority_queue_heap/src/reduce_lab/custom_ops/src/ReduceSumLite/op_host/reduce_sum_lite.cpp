/*
 * ============================================================
 * ReduceSumLite — Host 侧代码（教学版）
 * ============================================================
 *
 * 【Host 侧的职责】
 * 1. TilingFunc: 从输入 tensor shape 计算分块策略
 * 2. InferShape: 定义输出 tensor 的 shape
 * 3. OP_ADD: 注册算子到 CANN 框架
 *
 * 【Tiling 机制】
 *   Host 端运行在 CPU 上，负责"怎么分"
 *   Kernel 端运行在 AI Core 上，负责"算什么"
 *   Tiling 结构体是两者之间的桥梁
 */
#include "reduce_sum_lite_tiling.h"
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
 * 【当前写法】硬编码 BLOCK_DIM=8，兼容 310B。
 *   在 910B/A2/A3 上运行时，需修改此值为对应核数。
 *
 * 【核心原则】跨芯片适配的关键是：
 *   1. BLOCK_DIM / blockDim 在 Host/Tiling 侧计算，不要在 Kernel 侧硬编码
 *   2. totalLength 从输入 shape 动态计算，不要写死
 *   3. Kernel 只按 tiling 参数执行，不关心具体芯片型号
 */
static constexpr uint32_t BLOCK_DIM = 8;  // ⚠️ 310B=8, 910B=20+, A2=24, A3=24
static constexpr uint32_t TILE_LEN = 1024;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ReduceSumLiteTilingData tiling;
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::Shape &shape = xShape->GetStorageShape();
    uint32_t totalLength = static_cast<uint32_t>(shape.GetShapeSize());

    uint32_t blockDim = BLOCK_DIM;

    // 【分块计算】
    // blockLength: 每个核负责的元素数 = ceil(N / blockDim)
    // tileLength: 每次循环处理 1024 个元素（减少 UB 压力）
    // tileNum: 每个核需要循环的次数
    // lastTileLength: 最后一次循环可能不满 1024
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
    uint32_t tileLength = TILE_LEN;
    uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;
    uint32_t lastTileLength = blockLength - (tileNum - 1) * tileLength;
    if (tileNum == 0) { tileNum = 1; lastTileLength = 0; }

    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(tileNum);
    tiling.set_lastTileLength(lastTileLength);
    tiling.set_outputSize(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);  // TilingKey 必须与 kernel 入口一致

    // workspace = 0：结果直接写入输出 tensor，不需要 workspace
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
/*
 * 【InferShape】定义输出 tensor 的 shape
 * 输出 y 的 shape 是 [8]（BLOCK_DIM 个 partial sum）
 * Host 端读取 y[0..7] 并求和得到最终结果
 */
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = gert::Shape({static_cast<int64_t>(optiling::BLOCK_DIM)});  // 输出 size = BLOCK_DIM
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
/*
 * 【算子注册】
 * - 定义输入输出的 dtype 和 format
 * - 绑定 InferShape 和 TilingFunc
 * - AddConfig 指定目标平台
 */
class ReduceSumLite : public OpDef {
public:
    explicit ReduceSumLite(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReduceSumLite);
}