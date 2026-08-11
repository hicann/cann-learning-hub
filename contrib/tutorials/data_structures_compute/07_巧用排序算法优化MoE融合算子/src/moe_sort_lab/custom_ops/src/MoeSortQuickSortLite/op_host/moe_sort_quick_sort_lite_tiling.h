#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeSortQuickSortLiteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, numExperts);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeSortQuickSortLite, MoeSortQuickSortLiteTilingData);
}
