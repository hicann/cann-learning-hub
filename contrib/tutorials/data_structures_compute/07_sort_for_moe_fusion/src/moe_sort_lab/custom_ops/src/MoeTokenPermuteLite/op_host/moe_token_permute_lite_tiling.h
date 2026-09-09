#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeTokenPermuteLiteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenPermuteLite, MoeTokenPermuteLiteTilingData);
}
