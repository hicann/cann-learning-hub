#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/rope_optimized_tiling.h"

// extern 声明 kernel 入口（kernel 侧为 extern "C"，此处需保持一致）
extern "C" {
// FP32 入口
void rope_optimized_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                           uint8_t *x, uint8_t *cos, uint8_t *sin,
                           uint8_t *output, uint8_t *tiling);
// FP16 入口
void rope_optimized_kernel_fp16(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                uint8_t *x, uint8_t *cos, uint8_t *sin,
                                uint8_t *output, uint8_t *tiling);
}

namespace ascend_kernel {

at::Tensor rope_optimized_torch(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin)
{
    // 参数校验
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
                "only FP32/FP16 supported");
    TORCH_CHECK(x.scalar_type() == cos.scalar_type(), "x and cos must have the same dtype");
    TORCH_CHECK(x.scalar_type() == sin.scalar_type(), "x and sin must have the same dtype");
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(cos.is_privateuseone(), "cos must be on NPU");
    TORCH_CHECK(sin.is_privateuseone(), "sin must be on NPU");

    // x: [B, S, H, D], cos: [S, D], sin: [S, D]
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, S, H, D]");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2D [S, D]");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D [S, D]");
    TORCH_CHECK(x.size(1) == cos.size(0), "S dimension mismatch");
    TORCH_CHECK(x.size(1) == sin.size(0), "S dimension mismatch");
    TORCH_CHECK(x.size(3) == cos.size(1), "D dimension mismatch");
    TORCH_CHECK(x.size(3) == sin.size(1), "D dimension mismatch");

    at::Tensor y = at::empty_like(x);

    int64_t B = x.size(0);
    int64_t S = x.size(1);
    int64_t H = x.size(2);
    int64_t D = x.size(3);
    int64_t totalElements = x.numel();
    TORCH_CHECK(totalElements > 0, "input tensors must not be empty");
    TORCH_CHECK(D % 2 == 0, "D must be even for rotate_half");

    // stream(true) 在返回 ACL stream 前会清 queue，确保与之前 NPU 操作的正确同步
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 查询核数
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // dtype 标识
    int32_t dtypeFlag = (x.scalar_type() == at::kFloat) ? DTYPE_FLOAT : DTYPE_HALF;
    int32_t elemBytes = (dtypeFlag == DTYPE_FLOAT) ? 4 : 2;

    // Tiling 计算（baseRows/remainRows 负载均衡 + sTile 动态计算）
    RopeOptimizedTilingData tiling;
    tiling.B = B;
    tiling.S = S;
    tiling.H = H;
    tiling.D = D;
    tiling.BH = B * H;
    tiling.dtype = dtypeFlag;

    // 多核切分：baseRows/remainRows 负载均衡策略
    tiling.coreNum = (int32_t)std::min((int64_t)tiling.BH, (int64_t)availableCoreNum);
    tiling.baseRows = (int32_t)(tiling.BH / tiling.coreNum);
    tiling.remainRows = (int32_t)(tiling.BH % tiling.coreNum);
    tiling.blockNum = tiling.coreNum;

    // UB 切分：单缓冲，bufferNum=6（4 TQue + 2 TBuf）
    int32_t totalBufferNum = 6;
    int64_t maxElemNum = UB_SIZE / ((int64_t)totalBufferNum * elemBytes);
    int64_t maxSTile = maxElemNum / D;

    int32_t sTile = (int32_t)((maxSTile / 8) * 8);
    if (sTile > (int32_t)S) sTile = (int32_t)S;
    if (sTile < 1) sTile = 1;
    if (sTile > 255) sTile = 255;

    tiling.sTile = sTile;
    tiling.sLoop = ((int32_t)S + sTile - 1) / sTile;
    tiling.sTail = (int32_t)S - (tiling.sLoop - 1) * sTile;

    uint32_t blockNum = (uint32_t)tiling.blockNum;

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(RopeOptimizedTilingData))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(RopeOptimizedTilingData),
        &tiling, sizeof(RopeOptimizedTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 根据 dtype 选择 kernel 入口
    if (dtypeFlag == DTYPE_FLOAT) {
        rope_optimized_kernel(blockNum, nullptr, aclStream,
            reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(cos.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(sin.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));
    } else {
        rope_optimized_kernel_fp16(blockNum, nullptr, aclStream,
            reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(cos.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(sin.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));
    }

    return y;
}

} // namespace ascend_kernel
