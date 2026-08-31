#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TreeQueuePipelineLiteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, taskCount);
TILING_DATA_FIELD_DEF(uint32_t, queueDepth);
TILING_DATA_FIELD_DEF(uint32_t, computeLanes);
TILING_DATA_FIELD_DEF(float, copyIn);
TILING_DATA_FIELD_DEF(float, copyOut);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TreeQueuePipelineLite, TreeQueuePipelineLiteTilingData);
}
