#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/add_custom_tiling_data.h"
#include "../op_kernel/add_custom_tiling_key.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(8);
    AddCustomTilingData *tiling = context->GetTilingData<AddCustomTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileNum = 1;
    uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddCustom).Tiling(TilingFunc);
}  // namespace optiling