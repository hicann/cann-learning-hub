#include "kernel_operator.h"
#include "${OPERATOR_NAME}_tiling.h"

constexpr uint32_t ALIGN_NUM = 8;  // 32字节对齐
constexpr uint32_t DOUBLE_BUFFER = 2;

template<typename T>
class Kernel${OPERATOR_CLASS} {
public:
    __aicore__ inline Kernel${OPERATOR_CLASS}() {}
    
    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR output,
        const ${OPERATOR_CLASS}TilingData* tiling,
        uint32_t inputTotalElements)
    {
        tiling_ = tiling;
        blockIdx_ = AscendC::GetBlockIdx();
        
        // 计算当前核处理的元素数
        if (blockIdx_ < tiling->blockNum - 1) {
            totalElements_ = tiling->numPerCore;
        } else {
            totalElements_ = tiling->tailNumLastCore;
        }
        
        inputTotalElements_ = inputTotalElements;
        
        // 32字节对齐处理
        uint32_t bufferSize = totalElements_;
        if (bufferSize < ALIGN_NUM) bufferSize = ALIGN_NUM;
        bufferSize = (bufferSize + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        
        // 设置GlobalBuffer
        inputGm_.SetGlobalBuffer((__gm__ T*)input, inputTotalElements);
        
        uint32_t outputOffset = blockIdx_ * tiling->numPerCore;
        outputGm_.SetGlobalBuffer((__gm__ T*)output + outputOffset, totalElements_);
        
        // 初始化Queue
        pipe_.InitBuffer(inQueue_, DOUBLE_BUFFER, bufferSize * sizeof(T));
        pipe_.InitBuffer(outQueue_, DOUBLE_BUFFER, bufferSize * sizeof(T));
    }
    
    __aicore__ inline void Process() {
        if (totalElements_ == 0) return;
        
        // TODO: 实现计算逻辑
        // 1. CopyIn: 从GlobalMemory拷贝到LocalMemory
        // 2. Compute: 在LocalMemory上计算
        // 3. CopyOut: 从LocalMemory拷贝到GlobalMemory
        
        // 示例：简单的数据拷贝
        CopyIn();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();
        
        // 对齐count
        uint32_t alignCount = (totalElements_ + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        AscendC::DataCopy(inLocal, inputGm_, alignCount);
        
        inQueue_.EnQue(inLocal);
    }
    
    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<T> inLocal = inQueue_.DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        
        uint32_t alignCount = (totalElements_ + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        AscendC::DataCopy(outLocal, inLocal, alignCount);
        
        inQueue_.FreeTensor(inLocal);
        outQueue_.EnQue<T>(outLocal);
        
        // 输出到GlobalMemory
        outLocal = outQueue_.DeQue<T>();
        
        if (totalElements_ == alignCount) {
            AscendC::DataCopy(outputGm_, outLocal, totalElements_);
        } else {
            AscendC::DataCopyPad(outputGm_, outLocal,
                {1, static_cast<uint16_t>(totalElements_ * sizeof(T)),
                 static_cast<uint16_t>(alignCount * sizeof(T)), 0});
        }
        
        outQueue_.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe_;
    const ${OPERATOR_CLASS}TilingData* tiling_;
    
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    
    AscendC::TQue<AscendC::QuePosition::VECIN, DOUBLE_BUFFER> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, DOUBLE_BUFFER> outQueue_;
    
    uint32_t blockIdx_;
    uint32_t totalElements_;
    uint32_t inputTotalElements_;
};

__global__ __aicore__ void ${OPERATOR_NAME}(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(${OPERATOR_CLASS}TilingData);
    GET_TILING_DATA_WITH_STRUCT(${OPERATOR_CLASS}TilingData, tiling_data, tiling);
    
    Kernel${OPERATOR_CLASS}<DTYPE_INPUT_X> op;
    op.Init(input_x, output, &tiling_data, tiling_data.inputTotalElements);
    op.Process();
}
