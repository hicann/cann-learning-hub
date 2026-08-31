// ============================== 参考答案 ==============================
//   本文件是 op_host/cube_matmul_custom.cpp 的完整实现，TODO 1 已补全。
//   Host 仅构造进程内的 Tiling 信息；Kernel 边界传递 GM 指针和基础整数参数。
//   建议先独立完成后再对照阅读。
// ====================================================================

#include "experiment_config.h"

#include <algorithm>
#include <cstdint>

struct CubeMatmulCustomTiling {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t mTiles = 0;
    uint32_t nTiles = 0;
    uint32_t kTiles = 0;
    uint32_t usedCoreNum = 0;
};

inline bool BuildCubeMatmulCustomTiling(uint32_t M, uint32_t N, uint32_t K,
                                        uint32_t cubeCoreNum,
                                        CubeMatmulCustomTiling* tiling)
{
    if (tiling == nullptr || cubeCoreNum == 0 || M == 0 || N == 0 || K == 0) {
        return false;
    }

    if (LAB_BASE_M % 16 != 0 || LAB_BASE_N % 16 != 0 || LAB_BASE_K % 16 != 0 ||
        M % LAB_BASE_M != 0 || N % LAB_BASE_N != 0 || K % LAB_BASE_K != 0) {
        return false;
    }

    // ---------------- 答案 TODO 1：计算 M/N/K 方向的 Tile 数量 ----------------
    // baseM/baseN/baseK 分别切分 M/N/K。一个输出 Tile 对应 C 的一个 (mTile, nTile)
    // 区域，K 方向的 Tile 在同一输出区域内累加归约，不是独立输出任务，
    // 因此下面的 outputTileCount = mTiles * nTiles，不乘 kTiles。
    const uint32_t mTiles = M / LAB_BASE_M;
    const uint32_t nTiles = N / LAB_BASE_N;
    const uint32_t kTiles = K / LAB_BASE_K;

    const uint32_t outputTileCount = mTiles * nTiles;
    const uint32_t usedCoreNum = std::max(1U, std::min(outputTileCount, cubeCoreNum));

    tiling->M = M;
    tiling->N = N;
    tiling->K = K;
    tiling->mTiles = mTiles;
    tiling->nTiles = nTiles;
    tiling->kTiles = kTiles;
    tiling->usedCoreNum = usedCoreNum;
    return true;
}
