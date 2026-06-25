// Kernel侧核函数实现
#include "kernel_operator.h"

#include "as_strided_tiling.h"

#include "tiling_key_as_strided.h"

using namespace AscendC;

template <typename DT_INPUT_X>
class KernelAsStrided {
public:
    __aicore__ inline KernelAsStrided() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                                GM_ADDR tiling_addr,
                                uint32_t totalOutputElementsVal,
                                uint32_t tileSizeVal,
                                uint32_t inputTotalElementsVal,
                                uint32_t ndimVal,
                                uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3,
                                int32_t stride0, int32_t stride1, int32_t stride2, int32_t stride3,
                                int32_t storageOffsetVal,
                                uint32_t dimStride0, uint32_t dimStride1, uint32_t dimStride2, uint32_t dimStride3,
                                uint32_t pathFlagVal,
                                uint32_t blockDimVal,
                                uint32_t offsetTableInTilingVal) {
        // Store tiling data
        totalOutputElements = totalOutputElementsVal;
        tileSize = tileSizeVal;
        inputTotalElements = inputTotalElementsVal;
        ndim = ndimVal;
        sizeArr[0] = size0; sizeArr[1] = size1; sizeArr[2] = size2; sizeArr[3] = size3;
        strideArr[0] = stride0; strideArr[1] = stride1; strideArr[2] = stride2; strideArr[3] = stride3;
        storageOffset = storageOffsetVal;
        dimStrideArr[0] = dimStride0; dimStrideArr[1] = dimStride1;
        dimStrideArr[2] = dimStride2; dimStrideArr[3] = dimStride3;
        pathFlag = pathFlagVal;
        blockDimValue = blockDimVal;
        offsetTableInTiling = offsetTableInTilingVal;

        // Set global buffers (1D flattened view)
        inputGlobal.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x);
        outputGlobal.SetGlobalBuffer((__gm__ DT_INPUT_X*)output);

        // Set offset table global buffer (located after AsStridedTilingData in tiling GM)
        if (offsetTableInTiling == 1) {
            offsetTableGlobal.SetGlobalBuffer(
                (__gm__ uint32_t*)(tiling_addr + sizeof(AsStridedTilingData)));
        }

        // Compute core task range
        uint32_t blockIdx = GetBlockIdx();
        coreElements = CeilDiv(totalOutputElements, blockDimValue);
        coreOffset = blockIdx * coreElements;
        if (coreOffset >= totalOutputElements) {
            coreElements = 0;
            return;
        }
        if (coreOffset + coreElements > totalOutputElements) {
            coreElements = totalOutputElements - coreOffset;
        }

        // Initialize buffers based on path
        if (pathFlag == 0) {
            // Path A: full input in UB + offset table + output tile (Double Buffer)
            uint32_t inputBytes = AlignUp32(inputTotalElements * (uint32_t)sizeof(DT_INPUT_X));
            uint32_t offsetBytes = AlignUp32(tileSize * (uint32_t)sizeof(uint32_t));
            uint32_t outputBytes = AlignUp32(tileSize * (uint32_t)sizeof(DT_INPUT_X));

            pipe.InitBuffer(inQueueSrc, 1, inputBytes);
            pipe.InitBuffer(tmpQueueOffset, 1, offsetBytes);
            pipe.InitBuffer(outQueueDstA, 2, outputBytes);  // Double Buffer for Path A

            // Copy full input to UB (only once)
            LocalTensor<DT_INPUT_X> inputLocal = inQueueSrc.AllocTensor<DT_INPUT_X>();
            DataCopyExtParams inParams{1, inputTotalElements * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
            DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, (DT_INPUT_X)0};
            DataCopyPad(inputLocal, inputGlobal, inParams, padParams);
            inQueueSrc.EnQue(inputLocal);
        } else {
            // Path B: single element temp + srcIdx table + output tile
            uint32_t srcIdxBytes = AlignUp32(tileSize * (uint32_t)sizeof(uint32_t));
            uint32_t outputBytes = AlignUp32(tileSize * (uint32_t)sizeof(DT_INPUT_X));

            pipe.InitBuffer(inQueueTmp, 1, 32);
            pipe.InitBuffer(tmpQueueSrcIdx, 1, srcIdxBytes);
            pipe.InitBuffer(outQueueDstB, 1, outputBytes);
        }
    }

    __aicore__ inline void Process() {
        if (coreElements == 0) return;

        if (pathFlag == 0) {
            ProcessPathA();
        } else {
            ProcessPathB();
        }
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) {
        return (value + 31) / 32 * 32;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
        return (a + b - 1) / b;
    }

    // Initialize incremental index state for on-the-fly computation
    // Sets multiIdx[] and curSrcIdx for the given flat output index
    __aicore__ inline void initIncrementalState(uint32_t flatIdx) {
        curSrcIdx = storageOffset;
        uint32_t remaining = flatIdx;
        for (uint32_t d = 0; d < ndim; d++) {
            multiIdx[d] = remaining / dimStrideArr[d];
            remaining = remaining % dimStrideArr[d];
            curSrcIdx += (int32_t)multiIdx[d] * strideArr[d];
        }
        // Defensive clamp
        if (curSrcIdx < 0 || (uint32_t)curSrcIdx >= inputTotalElements) {
            curSrcIdx = 0;
        }
    }

    // Advance incremental state to next output element (no division/modulo)
    __aicore__ inline void advanceIncrementalState() {
        // Mixed-radix counter: increment last dimension, carry as needed
        int32_t d = (int32_t)ndim - 1;
        multiIdx[d]++;
        curSrcIdx += strideArr[d];
        while (d > 0 && multiIdx[d] >= sizeArr[d]) {
            // Dimension d overflowed: reset and carry to d-1
            curSrcIdx -= (int32_t)sizeArr[d] * strideArr[d];
            multiIdx[d] = 0;
            d--;
            multiIdx[d]++;
            curSrcIdx += strideArr[d];
        }
        // Defensive clamp
        if (curSrcIdx < 0 || (uint32_t)curSrcIdx >= inputTotalElements) {
            curSrcIdx = 0;
        }
    }

    // Path A: input fully in UB + Gather
    __aicore__ inline void ProcessPathA() {
        // Input is already in UB (copied in Init)
        LocalTensor<DT_INPUT_X> inputLocal = inQueueSrc.DeQue<DT_INPUT_X>();

        uint32_t tilesPerCore = CeilDiv(coreElements, tileSize);

        if (offsetTableInTiling == 1) {
            // Preferred path: offset table pre-computed in Host, stored in tiling data
            // DataCopyPad from tiling GM to UB, then Gather — no SetValue, no division/modulo
            for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
                uint32_t tileStart = coreOffset + tile * tileSize;
                uint32_t curTileSize = tileSize;
                if (tileStart + curTileSize > coreOffset + coreElements) {
                    curTileSize = coreOffset + coreElements - tileStart;
                }

                // 1. DataCopyPad offset table from tiling GM to UB
                LocalTensor<uint32_t> offsetLocal = tmpQueueOffset.AllocTensor<uint32_t>();
                DataCopyExtParams offParams{1, curTileSize * (uint32_t)sizeof(uint32_t), 0, 0, 0};
                DataCopyPadExtParams<uint32_t> offPadParams{false, 0, 0, (uint32_t)0};
                DataCopyPad(offsetLocal, offsetTableGlobal[tileStart], offParams, offPadParams);
                tmpQueueOffset.EnQue(offsetLocal);

                // 2. Gather: collect elements from inputLocal using offsetLocal
                LocalTensor<uint32_t> offLocal = tmpQueueOffset.DeQue<uint32_t>();
                LocalTensor<DT_INPUT_X> outputLocal = outQueueDstA.AllocTensor<DT_INPUT_X>();
                Gather(outputLocal, inputLocal, offLocal, 0, curTileSize);
                outQueueDstA.EnQue(outputLocal);
                tmpQueueOffset.FreeTensor(offLocal);

                // 3. CopyOut: write output tile to GM
                LocalTensor<DT_INPUT_X> outLocal = outQueueDstA.DeQue<DT_INPUT_X>();
                DataCopyExtParams outParams{1, curTileSize * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
                DataCopyPad(outputGlobal[tileStart], outLocal, outParams);
                outQueueDstA.FreeTensor(outLocal);
            }
        } else {
            // Fallback path: on-the-fly computation using incremental index (no division/modulo)
            // Still uses SetValue for offset table construction, but avoids expensive div/mod
            for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
                uint32_t tileStart = coreOffset + tile * tileSize;
                uint32_t curTileSize = tileSize;
                if (tileStart + curTileSize > coreOffset + coreElements) {
                    curTileSize = coreOffset + coreElements - tileStart;
                }

                // 1. Compute byte offsets for this tile using incremental state
                if (tile == 0) {
                    initIncrementalState(tileStart);
                }

                LocalTensor<uint32_t> offsetLocal = tmpQueueOffset.AllocTensor<uint32_t>();
                for (uint32_t i = 0; i < curTileSize; i++) {
                    offsetLocal.SetValue(i, (uint32_t)curSrcIdx * (uint32_t)sizeof(DT_INPUT_X));
                    if (i + 1 < curTileSize) {
                        advanceIncrementalState();
                    }
                }
                tmpQueueOffset.EnQue(offsetLocal);

                // 2. Gather: collect elements from inputLocal using offsetLocal
                LocalTensor<uint32_t> offLocal = tmpQueueOffset.DeQue<uint32_t>();
                LocalTensor<DT_INPUT_X> outputLocal = outQueueDstA.AllocTensor<DT_INPUT_X>();
                Gather(outputLocal, inputLocal, offLocal, 0, curTileSize);
                outQueueDstA.EnQue(outputLocal);
                tmpQueueOffset.FreeTensor(offLocal);

                // 3. CopyOut: write output tile to GM
                LocalTensor<DT_INPUT_X> outLocal = outQueueDstA.DeQue<DT_INPUT_X>();
                DataCopyExtParams outParams{1, curTileSize * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
                DataCopyPad(outputGlobal[tileStart], outLocal, outParams);
                outQueueDstA.FreeTensor(outLocal);

                // Advance to next tile start
                if (tilesPerCore > 1 && tile + 1 < tilesPerCore) {
                    advanceIncrementalState();
                }
            }
        }

        inQueueSrc.FreeTensor(inputLocal);
    }

    // Path B: per-element DataCopyPad from GM (fallback for large inputs)
    __aicore__ inline void ProcessPathB() {
        uint32_t tilesPerCore = CeilDiv(coreElements, tileSize);

        if (offsetTableInTiling == 1) {
            // Preferred path: srcIdx table pre-computed in Host, stored in tiling data
            // DataCopyPad from tiling GM to UB, then per-element read — no division/modulo
            for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
                uint32_t tileStart = coreOffset + tile * tileSize;
                uint32_t curTileSize = tileSize;
                if (tileStart + curTileSize > coreOffset + coreElements) {
                    curTileSize = coreOffset + coreElements - tileStart;
                }

                // 1. DataCopyPad srcIdx table from tiling GM to UB
                LocalTensor<uint32_t> srcIdxLocal = tmpQueueSrcIdx.AllocTensor<uint32_t>();
                DataCopyExtParams idxParams{1, curTileSize * (uint32_t)sizeof(uint32_t), 0, 0, 0};
                DataCopyPadExtParams<uint32_t> idxPadParams{false, 0, 0, (uint32_t)0};
                DataCopyPad(srcIdxLocal, offsetTableGlobal[tileStart], idxParams, idxPadParams);
                tmpQueueSrcIdx.EnQue(srcIdxLocal);

                LocalTensor<uint32_t> idxLocal = tmpQueueSrcIdx.DeQue<uint32_t>();

                // 2. Per-element: read srcIdx from table, DataCopyPad from GM, SetValue to output
                LocalTensor<DT_INPUT_X> outputLocal = outQueueDstB.AllocTensor<DT_INPUT_X>();
                for (uint32_t i = 0; i < curTileSize; i++) {
                    uint32_t srcIdx = idxLocal.GetValue(i);

                    // DataCopyPad single element from GM to UB
                    LocalTensor<DT_INPUT_X> tmpLocal = inQueueTmp.AllocTensor<DT_INPUT_X>();
                    DataCopyExtParams inParams{1, (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
                    DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, (DT_INPUT_X)0};
                    DataCopyPad(tmpLocal, inputGlobal[srcIdx], inParams, padParams);
                    inQueueTmp.EnQue(tmpLocal);

                    // Read value and write to output position
                    LocalTensor<DT_INPUT_X> valLocal = inQueueTmp.DeQue<DT_INPUT_X>();
                    DT_INPUT_X val = valLocal.GetValue(0);
                    outputLocal.SetValue(i, val);
                    inQueueTmp.FreeTensor(valLocal);
                }

                tmpQueueSrcIdx.FreeTensor(idxLocal);

                // 3. CopyOut: write output tile to GM
                outQueueDstB.EnQue(outputLocal);
                LocalTensor<DT_INPUT_X> outLocal = outQueueDstB.DeQue<DT_INPUT_X>();
                DataCopyExtParams outParams{1, curTileSize * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
                DataCopyPad(outputGlobal[tileStart], outLocal, outParams);
                outQueueDstB.FreeTensor(outLocal);
            }
        } else {
            // Fallback path: on-the-fly computation using incremental index (no division/modulo)
            for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
                uint32_t tileStart = coreOffset + tile * tileSize;
                uint32_t curTileSize = tileSize;
                if (tileStart + curTileSize > coreOffset + coreElements) {
                    curTileSize = coreOffset + coreElements - tileStart;
                }

                // Initialize incremental state at tile start
                if (tile == 0) {
                    initIncrementalState(tileStart);
                }

                // Per-element: use incremental srcIdx, DataCopyPad from GM, SetValue to output
                LocalTensor<DT_INPUT_X> outputLocal = outQueueDstB.AllocTensor<DT_INPUT_X>();
                for (uint32_t i = 0; i < curTileSize; i++) {
                    // DataCopyPad single element from GM to UB
                    LocalTensor<DT_INPUT_X> tmpLocal = inQueueTmp.AllocTensor<DT_INPUT_X>();
                    DataCopyExtParams inParams{1, (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
                    DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, (DT_INPUT_X)0};
                    DataCopyPad(tmpLocal, inputGlobal[(uint32_t)curSrcIdx], inParams, padParams);
                    inQueueTmp.EnQue(tmpLocal);

                    // Read value and write to output position
                    LocalTensor<DT_INPUT_X> valLocal = inQueueTmp.DeQue<DT_INPUT_X>();
                    DT_INPUT_X val = valLocal.GetValue(0);
                    outputLocal.SetValue(i, val);
                    inQueueTmp.FreeTensor(valLocal);

                    // Advance to next element
                    if (i + 1 < curTileSize) {
                        advanceIncrementalState();
                    }
                }

                // CopyOut: write output tile to GM
                outQueueDstB.EnQue(outputLocal);
                LocalTensor<DT_INPUT_X> outLocal = outQueueDstB.DeQue<DT_INPUT_X>();
                DataCopyExtParams outParams{1, curTileSize * (uint32_t)sizeof(DT_INPUT_X), 0, 0, 0};
                DataCopyPad(outputGlobal[tileStart], outLocal, outParams);
                outQueueDstB.FreeTensor(outLocal);

                // Advance to next tile start
                if (tilesPerCore > 1 && tile + 1 < tilesPerCore) {
                    advanceIncrementalState();
                }
            }
        }
    }

    // Pipe and Queues
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc;        // Path A: full input
    TQue<TPosition::VECIN, 1> inQueueTmp;        // Path B: single element temp
    TQue<TPosition::VECCALC, 1> tmpQueueOffset;  // Path A: Gather offset table
    TQue<TPosition::VECCALC, 1> tmpQueueSrcIdx;  // Path B: srcIdx table
    TQue<TPosition::VECOUT, 2> outQueueDstA;     // Path A: output tile (Double Buffer)
    TQue<TPosition::VECOUT, 1> outQueueDstB;     // Path B: output tile

    // Global tensors
    GlobalTensor<DT_INPUT_X> inputGlobal;
    GlobalTensor<DT_INPUT_X> outputGlobal;
    GlobalTensor<uint32_t> offsetTableGlobal;    // Offset/srcIdx table in tiling GM

    // Tiling data
    uint32_t totalOutputElements;
    uint32_t tileSize;
    uint32_t inputTotalElements;
    uint32_t ndim;
    uint32_t sizeArr[4];
    int32_t strideArr[4];
    int32_t storageOffset;
    uint32_t dimStrideArr[4];
    uint32_t pathFlag;
    uint32_t blockDimValue;
    uint32_t offsetTableInTiling;

    // Core task range
    uint32_t coreElements;
    uint32_t coreOffset;

    // Incremental index state (for on-the-fly fallback)
    int32_t curSrcIdx;
    uint32_t multiIdx[4];
};

template <typename DT_INPUT_X>
__global__ __aicore__ void as_strided(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AsStridedTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tiling_data, tiling);
    KernelAsStrided<DT_INPUT_X> op;
    op.Init(input_x, output, tiling,
            tiling_data.totalOutputElements,
            tiling_data.tileSize,
            tiling_data.inputTotalElements,
            tiling_data.ndim,
            tiling_data.size[0], tiling_data.size[1], tiling_data.size[2], tiling_data.size[3],
            tiling_data.stride[0], tiling_data.stride[1], tiling_data.stride[2], tiling_data.stride[3],
            tiling_data.storageOffset,
            tiling_data.dimStride[0], tiling_data.dimStride[1], tiling_data.dimStride[2], tiling_data.dimStride[3],
            tiling_data.pathFlag,
            tiling_data.blockDim,
            tiling_data.offsetTableInTiling);
    op.Process();
}
