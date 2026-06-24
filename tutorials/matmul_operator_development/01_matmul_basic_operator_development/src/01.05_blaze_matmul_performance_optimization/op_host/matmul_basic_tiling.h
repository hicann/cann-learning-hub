#ifndef MATMUL_BASIC_TUTORIAL_HOST_TILING_H
#define MATMUL_BASIC_TUTORIAL_HOST_TILING_H

#include "matmul_basic_tiling_data.h"
#include <cstdint>
#include <algorithm>

namespace MatmulBasicTutorial {

class MatmulBasicTiling {
public:
    static MatmulBasicTutorialTilingData DoOpTiling(
        int64_t m, int64_t n, int64_t k, int64_t coreNum)
    {
        MatmulBasicTutorialTilingData tiling;
        tiling.m = static_cast<uint32_t>(m);
        tiling.n = static_cast<uint32_t>(n);
        tiling.k = static_cast<uint32_t>(k);
        tiling.usedCoreNum = static_cast<uint32_t>(coreNum);

        tiling.baseM = 128;
        tiling.baseN = 256;
        tiling.baseK = 64;
        tiling.mL1 = 128;
        tiling.nL1 = 256;
        tiling.kL1 = 64;

        tiling.mTailCnt = static_cast<uint32_t>(m % tiling.baseM);
        tiling.nTailCnt = static_cast<uint32_t>(n % tiling.baseN);

        if (m % tiling.mL1 > 0) {
            uint32_t tailL1M = static_cast<uint32_t>(m % tiling.mL1);
            if (tailL1M <= tiling.baseM) {
                tiling.mBaseTailSplitCnt = 1;
                tiling.mTailMain = tailL1M;
            } else {
                tiling.mBaseTailSplitCnt = CeilDiv(tailL1M, tiling.baseM);
                tiling.mTailMain = tiling.baseM;
            }
        }

        if (n % tiling.nL1 > 0) {
            uint32_t tailL1N = static_cast<uint32_t>(n % tiling.nL1);
            if (tailL1N <= tiling.baseN) {
                tiling.nBaseTailSplitCnt = 1;
                tiling.nTailMain = tailL1N;
            } else {
                tiling.nBaseTailSplitCnt = CeilDiv(tailL1N, tiling.baseN);
                tiling.nTailMain = tiling.baseN;
            }
        }

        tiling.isHf32 = 0;
        tiling.l1BufferNum = 0;
        tiling.l0cDB = 1;
        tiling.ubDB = 1;
        tiling.l2CacheDisable = 0;
        tiling.sliceM = 1;
        tiling.srcNdStride = 1;
        tiling.innerBatch = 1;

        return tiling;
    }

private:
    static uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }
};

} // namespace MatmulBasicTutorial

#endif
