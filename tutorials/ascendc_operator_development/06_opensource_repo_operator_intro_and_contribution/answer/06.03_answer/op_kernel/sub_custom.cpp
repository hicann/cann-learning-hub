
#include "sub_custom.h"

enum class SubTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void sub_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(SubCustomTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(SubTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsSub::Sub<float> op;
        op.Init(x1, x2, y, tilingData.totalLength, tilingData.tileNum);
        op.Process();
    }
}
