#ifndef GEMM_BASELINE_TILING_H
#define GEMM_BASELINE_TILING_H

#include <cstdint>

#pragma pack(push, 1)
struct GemmBaselineTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t coreNum = 1;
    uint32_t rowsPerCore = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
};
#pragma pack(pop)

static_assert(sizeof(GemmBaselineTiling) == 32, "GemmBaselineTiling size must be 32 bytes");

#endif