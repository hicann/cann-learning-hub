#ifndef TILING_DATA_H
#define TILING_DATA_H

#include <stdint.h>
#include <kernel_tiling/kernel_tiling.h>

namespace MatmulAllGatherImpl {

constexpr static int64_t MAX_RANK_NUM = 64;

struct HcclA5OpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[MAX_RANK_NUM];
    uint64_t windowsOut[MAX_RANK_NUM];
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};

struct MatmulAllGatherTilingInfo {
    uint64_t mDim;       // M 维度（每个 rank 计算完整 M × K × N）
    uint64_t kDim;
    uint64_t nDim;
    uint64_t aivNum;
    uint64_t totalWinSize;
    uint64_t baseM;
    uint64_t baseK;
    uint64_t baseN;
};

struct MatmulAllGatherTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MatmulAllGatherTilingInfo tilingInfo;
};

}

#endif