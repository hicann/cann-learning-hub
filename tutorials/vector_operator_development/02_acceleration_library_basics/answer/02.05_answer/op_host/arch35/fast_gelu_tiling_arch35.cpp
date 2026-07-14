#include "fast_gelu_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

namespace ge {

IMPLEMT_VERIFIER(FastGelu, FastGeluVerify) {
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INIT_WITH_TILING_DATA_BEGIN(FastGelu)
    uint32_t totalLength = op_desc->GetInputDesc(0).GetShape().GetShapeSize();
    auto tilingData = reinterpret_cast<FastGeluTilingData*>(context->GetRawTilingData());
    tilingData->totalLength = totalLength;
    tilingData->tileNum = 8;
IMPLEMT_COMMON_INIT_WITH_TILING_DATA_END

}  // namespace ge
