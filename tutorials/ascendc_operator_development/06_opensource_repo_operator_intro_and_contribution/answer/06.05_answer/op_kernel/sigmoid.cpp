#include "sigmoid.h"

enum class SigmoidTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidTilingData, tilingData, tiling);

    // 场景1
    if constexpr (schMode == static_cast<uint32_t>(SigmoidTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsSigmoid::Sigmoid<float> op; // 算子kernel实例获取
        op.Init(x, y, tilingData.totalLength, tilingData.tileNum);      // 算子kernel实例初始化
        op.Process();                       // 算子kernel实例执行
    }

    // 场景2
    if constexpr (schMode == static_cast<uint32_t>(SigmoidTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        NsSigmoid::Sigmoid<int32_t> op; // 算子kernel实例获取
        op.Init(x, y, tilingData.totalLength, tilingData.tileNum);        // 算子kernel实例初始化
        op.Process2();                         // 算子kernel实例执行
    }
}
