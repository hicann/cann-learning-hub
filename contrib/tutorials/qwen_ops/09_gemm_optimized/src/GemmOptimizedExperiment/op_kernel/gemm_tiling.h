#ifndef GEMM_OPTIMIZED_TILING_H
#define GEMM_OPTIMIZED_TILING_H

#include <cstdint>

#pragma pack(push, 1)
struct GemmOptimizedTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t coreNum = 1;
    uint32_t tileM = 8;
    uint32_t tileN = 8;
    uint32_t tileK = 128;
    uint32_t totalTiles = 0;
};
#pragma pack(pop)

static_assert(sizeof(GemmOptimizedTiling) == 32, "GemmOptimizedTiling size must be 32 bytes");

#endif