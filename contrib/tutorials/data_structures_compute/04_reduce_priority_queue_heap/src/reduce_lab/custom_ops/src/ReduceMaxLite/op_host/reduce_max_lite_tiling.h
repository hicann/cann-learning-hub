#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceMaxLiteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
TILING_DATA_FIELD_DEF(uint32_t, outputSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ReduceMaxLite, ReduceMaxLiteTilingData);
}
