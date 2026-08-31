#ifndef TOP_K_REDUCE_LITE_TILING_H
#define TOP_K_REDUCE_LITE_TILING_H

#ifndef TOPK_REDUCE_LITE_KERNEL_UNDEF_DT_MACROS
#define TOPK_REDUCE_LITE_KERNEL_UNDEF_DT_MACROS
#ifdef DT_FLOAT
#undef DT_FLOAT
#endif
#ifdef DT_FLOAT16
#undef DT_FLOAT16
#endif
#ifdef DT_INT8
#undef DT_INT8
#endif
#ifdef DT_INT16
#undef DT_INT16
#endif
#ifdef DT_INT32
#undef DT_INT32
#endif
#ifdef DT_INT64
#undef DT_INT64
#endif
#ifdef DT_UINT8
#undef DT_UINT8
#endif
#ifdef DT_UINT16
#undef DT_UINT16
#endif
#ifdef DT_UINT32
#undef DT_UINT32
#endif
#ifdef DT_UINT64
#undef DT_UINT64
#endif
#ifdef DT_BOOL
#undef DT_BOOL
#endif
#endif

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TopKReduceLiteTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, topK);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TopKReduceLite, TopKReduceLiteTilingData)
}

#endif