
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sub_custom_tiling_data.h"
#include "../op_kernel/sub_custom_tiling_key.h"

namespace optiling {

struct SubCustomCompileInfo {};

// tiling 分发入口
static ge::graphStatus SubCustomTilingFunc(gert::TilingContext* context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(8);
    SubCustomTilingData *tiling = context->GetTilingData<SubCustomTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileNum = 1;
    uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSubCustom([[maybe_unused]] gert::TilingParseContext* context)
{   
    // AscendC 算子可以直接返回SUCCESS提升性能，硬件信息可在运行时获取
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SubCustom).Tiling(SubCustomTilingFunc).TilingParse<SubCustomCompileInfo>(TilingParseForSubCustom);
} // namespace optiling
