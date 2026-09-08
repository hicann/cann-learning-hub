#ifndef SWIGLU_TILING_H
#define SWIGLU_TILING_H

#include <cstdint>

struct SwiGluTilingData {
    uint32_t totalSize = 0;
    uint32_t elementsPerCore = 0;
    uint32_t coreNum = 1;
    uint32_t tileLength = 1024;
};

#endif