#include "moe_token_permute_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

class KernelMoeTokenPermuteLite {
public:
    __aicore__ inline KernelMoeTokenPermuteLite() {}

    __aicore__ inline void Init(GM_ADDR tokens, GM_ADDR sortedOrder, GM_ADDR probs,
                                GM_ADDR permutedTokens, GM_ADDR sortedIndices, GM_ADDR permutedProbs,
                                uint32_t inNumTokens, uint32_t inHiddenSize, uint32_t inTopK, uint32_t inTotalRows, uint32_t inRowsPerCore)
    {
        numTokens = inNumTokens;
        hiddenSize = inHiddenSize;
        topK = inTopK;
        totalRows = inTotalRows;
        rowsPerCore = inRowsPerCore;

        tokensGm.SetGlobalBuffer((__gm__ half *)tokens, numTokens * hiddenSize);
        orderGm.SetGlobalBuffer((__gm__ int32_t *)sortedOrder, totalRows);
        probsGm.SetGlobalBuffer((__gm__ half *)probs, numTokens * topK);
        outTokensGm.SetGlobalBuffer((__gm__ half *)permutedTokens, totalRows * hiddenSize);
        outIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sortedIndices, totalRows);
        outProbsGm.SetGlobalBuffer((__gm__ half *)permutedProbs, totalRows);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t startRow = coreId * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > totalRows) {
            endRow = totalRows;
        }

        for (uint32_t row = startRow; row < endRow; ++row) {
            int32_t pairId = orderGm.GetValue(row);
            uint32_t pair = static_cast<uint32_t>(pairId);
            uint32_t tokenId = pair / topK;
            uint32_t kId = pair - tokenId * topK;

            outIndicesGm.SetValue(row, static_cast<int32_t>(tokenId));
            half p = probsGm.GetValue(tokenId * topK + kId);
            outProbsGm.SetValue(row, p);

            uint32_t src = tokenId * hiddenSize;
            uint32_t dst = row * hiddenSize;
            for (uint32_t h = 0; h < hiddenSize; ++h) {
                half v = tokensGm.GetValue(src + h);
                outTokensGm.SetValue(dst + h, v);
            }
        }
    }

private:
    GlobalTensor<half> tokensGm;
    GlobalTensor<int32_t> orderGm;
    GlobalTensor<half> probsGm;
    GlobalTensor<half> outTokensGm;
    GlobalTensor<int32_t> outIndicesGm;
    GlobalTensor<half> outProbsGm;
    uint32_t numTokens;
    uint32_t hiddenSize;
    uint32_t topK;
    uint32_t totalRows;
    uint32_t rowsPerCore;
};

__aicore__ inline void RunMoeTokenPermuteLite(GM_ADDR tokens, GM_ADDR sortedOrder, GM_ADDR probs,
                                              GM_ADDR permutedTokens, GM_ADDR sortedIndices, GM_ADDR permutedProbs,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelMoeTokenPermuteLite op;
    op.Init(tokens, sortedOrder, probs, permutedTokens, sortedIndices, permutedProbs,
            tilingData.numTokens, tilingData.hiddenSize, tilingData.topK, tilingData.totalRows, tilingData.rowsPerCore);
    op.Process();
}

extern "C" __global__ __aicore__ void moe_token_permute_lite(GM_ADDR tokens, GM_ADDR sortedOrder, GM_ADDR probs,
                                                              GM_ADDR permutedTokens, GM_ADDR sortedIndices, GM_ADDR permutedProbs,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        RunMoeTokenPermuteLite(tokens, sortedOrder, probs, permutedTokens, sortedIndices, permutedProbs, workspace, tiling);
    } else if (TILING_KEY_IS(1)) {
        RunMoeTokenPermuteLite(tokens, sortedOrder, probs, permutedTokens, sortedIndices, permutedProbs, workspace, tiling);
    }
}