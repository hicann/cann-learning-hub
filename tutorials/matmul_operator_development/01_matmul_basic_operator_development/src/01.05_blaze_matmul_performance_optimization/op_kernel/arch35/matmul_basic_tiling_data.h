#ifndef MATMUL_BASIC_TUTORIAL_TILING_DATA_H
#define MATMUL_BASIC_TUTORIAL_TILING_DATA_H

#include <cstdint>

#pragma pack(push, 8)

struct MatmulBasicTutorialTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t mL1 = 0;
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 1;
    uint32_t nTailMain = 1;
    uint8_t isHf32 = 0;
    uint8_t l1BufferNum = 0;
    uint8_t l0cDB = 1;
    uint8_t ubDB = 1;
    uint8_t l2CacheDisable = 0;
    uint32_t sliceM = 1;
    uint32_t srcNdStride = 1;
    uint32_t innerBatch = 1;
};

#pragma pack(pop)

#endif
