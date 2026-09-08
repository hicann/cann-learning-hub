#ifndef RMSNORM_TILING_H
#define RMSNORM_TILING_H

#include <cstdint>

#pragma pack(push, 1)
struct RmsNormBaselineTiling {
    uint32_t rows = 0;
    uint32_t hidden = 0;
    uint32_t coreNum = 1;
    uint32_t rowsPerCore = 0;
    float eps = 1e-6f;
    float invHidden = 1.0f;
};
#pragma pack(pop)

static_assert(sizeof(RmsNormBaselineTiling) == 24, "RmsNormBaselineTiling size must be 24 bytes");

#endif
