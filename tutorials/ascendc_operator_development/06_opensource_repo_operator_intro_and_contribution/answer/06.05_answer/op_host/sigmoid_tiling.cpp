#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"

namespace optiling {

struct SigmoidCompileInfo {};

// tiling 分发入口
static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(8);
    SigmoidTilingData *tiling = context->GetTilingData<SigmoidTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileNum = 2;

    ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
 
    uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{   
    // AscendC 算子可以直接返回SUCCESS提升性能，硬件信息可在运行时获取
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);
} // namespace optiling
