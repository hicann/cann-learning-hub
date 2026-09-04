#include "kernel_operator.h"

// Local-row-parallel FP32 CSR SpMV baseline. Local output rows are split
// into contiguous per-core ranges: every range starts at a 64-byte-aligned
// position (16 floats per cacheline), non-final ranges use the aligned
// stride, and only the final range truncates at the output end. Cores
// therefore never write into the same 64 B cacheline, which keeps
// multi-core GlobalTensor::SetValue on the float output free of cacheline
// overwrites.
extern "C" __global__ __vector__ __aicore__ void spmv_local_fp32(
    GM_ADDR rowPtr, GM_ADDR colIdx, GM_ADDR values, GM_ADDR x, GM_ADDR localY,
    int64_t firstRow, int64_t lastRow) {
    constexpr int64_t kFloatsPerCacheLine = 16;
    const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t localRows = lastRow - firstRow;
    const int64_t elementsPerBlock =
        ((localRows + blockCount - 1) / blockCount + kFloatsPerCacheLine - 1) /
        kFloatsPerCacheLine * kFloatsPerCacheLine;
    const int64_t first = static_cast<int64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
    const int64_t last = first + elementsPerBlock < localRows ? first + elementsPerBlock : localRows;
    AscendC::GlobalTensor<int32_t> rowPtrGm, colIdxGm;
    AscendC::GlobalTensor<float> valuesGm, xGm, yGm;
    rowPtrGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rowPtr));
    colIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(colIdx));
    valuesGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values));
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localY));
    for (int64_t localRow = first; localRow < last; ++localRow) {
        float sum = 0.0F;
        const int64_t row = firstRow + localRow;
        const int32_t begin = rowPtrGm.GetValue(static_cast<int32_t>(row));
        const int32_t end = rowPtrGm.GetValue(static_cast<int32_t>(row + 1));
        for (int32_t p = begin; p < end; ++p) {
            sum += valuesGm.GetValue(p) * xGm.GetValue(colIdxGm.GetValue(p));
        }
        yGm.SetValue(static_cast<int32_t>(localRow), sum);
    }
}
