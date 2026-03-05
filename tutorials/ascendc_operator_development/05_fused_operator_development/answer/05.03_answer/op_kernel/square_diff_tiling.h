
#ifndef SQUARE_DIFF_TILING_H
#define SQUARE_DIFF_TILING_H
#include <cstdint>

struct SquareDiffTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
};

#endif // SQUARE_DIFF_TILING_H
