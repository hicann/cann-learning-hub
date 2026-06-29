#ifndef QMM_CUSTOM_TILING_H
#define QMM_CUSTOM_TILING_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) QmmCustomTilingData {
    AscendC::tiling::TCubeTiling cubeTilingData;
    uint32_t isPertoken;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t singleCoreM;
    int32_t singleCoreN;
    uint64_t workspaceSize;
};
#pragma pack(pop)

#endif // QMM_CUSTOM_TILING_H
