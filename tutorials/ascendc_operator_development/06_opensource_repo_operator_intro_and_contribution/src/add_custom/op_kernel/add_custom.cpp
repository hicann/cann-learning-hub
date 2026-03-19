#include "add_custom.h"

enum class AddTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE = 0,
};

template <uint32_t schMode>
 __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTilingData, tiling_data, tiling);
    if constexpr (schMode == static_cast<uint32_t>(AddTilingKey::TILING_KEY_EXAMPLE)) {
        KernelAdd<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
        op.Process();
    }
}