#ifndef SWIGLU_TILING_H
#define SWIGLU_TILING_H

#include <cstdint>

#pragma pack(push, 1)
struct SwiGluTiling {
    uint32_t totalSize = 0;
    uint32_t coreNum = 1;
    uint32_t elementsPerCore = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(SwiGluTiling) == 16, "SwiGluTiling size must be 16 bytes");

#endif