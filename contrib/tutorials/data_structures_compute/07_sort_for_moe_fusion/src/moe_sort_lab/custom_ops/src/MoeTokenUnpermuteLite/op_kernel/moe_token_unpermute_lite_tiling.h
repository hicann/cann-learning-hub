#ifndef MOE_TOKEN_UNPERMUTE_LITE_TILING_H
#define MOE_TOKEN_UNPERMUTE_LITE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteLiteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenUnpermuteLite, MoeTokenUnpermuteLiteTilingData)
}
#endif
