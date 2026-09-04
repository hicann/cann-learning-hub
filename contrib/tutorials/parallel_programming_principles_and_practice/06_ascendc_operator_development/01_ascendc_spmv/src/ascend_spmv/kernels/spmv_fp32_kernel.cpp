#include "kernel_operator.h"

// Row-parallel FP32 CSR SpMV baseline. Output rows are split into contiguous
// per-core ranges: every range starts at a 64-byte-aligned position (16
// floats per cacheline), non-final ranges use the aligned stride, and only
// the final range truncates at the output end. Cores therefore never write
// into the same 64 B cacheline, which keeps multi-core
// GlobalTensor::SetValue on the float output free of cacheline overwrites.
extern "C" __global__ __vector__ __aicore__ void spmv_fp32(
    GM_ADDR rowPtr, GM_ADDR colIdx, GM_ADDR values, GM_ADDR x, GM_ADDR y,
    int64_t rows) {
    constexpr int64_t kFloatsPerCacheLine = 16;
    const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t elementsPerBlock =
        ((rows + blockCount - 1) / blockCount + kFloatsPerCacheLine - 1) /
        kFloatsPerCacheLine * kFloatsPerCacheLine;
    const int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
    const int64_t last = first + elementsPerBlock < rows ? first + elementsPerBlock : rows;
    AscendC::GlobalTensor<int32_t> rowPtrGm;
    AscendC::GlobalTensor<int32_t> colIdxGm;
    AscendC::GlobalTensor<float> valuesGm;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    rowPtrGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr));
    colIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(colIdx));
    valuesGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values));
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    for (int64_t row = first; row < last; ++row) {
        float sum = 0.0F;
        const int32_t begin = rowPtrGm.GetValue(static_cast<int32_t>(row));
        const int32_t end = rowPtrGm.GetValue(static_cast<int32_t>(row + 1));
        for (int32_t index = begin; index < end; ++index) {
            sum += valuesGm.GetValue(index) * xGm.GetValue(colIdxGm.GetValue(index));
        }
        yGm.SetValue(static_cast<int32_t>(row), sum);
    }
}
