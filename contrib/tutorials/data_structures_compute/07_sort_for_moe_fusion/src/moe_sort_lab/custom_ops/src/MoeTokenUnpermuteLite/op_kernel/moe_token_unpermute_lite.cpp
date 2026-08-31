#include "moe_token_unpermute_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

class KernelMoeTokenUnpermuteLite {
public:
    __aicore__ inline KernelMoeTokenUnpermuteLite() {}

    __aicore__ inline void Init(GM_ADDR expertOut, GM_ADDR sortedIndices, GM_ADDR permutedProbs, GM_ADDR out,
                                uint32_t inNumTokens, uint32_t inHiddenSize, uint32_t inTopK, uint32_t inTotalRows, uint32_t inTokensPerCore)
    {
        numTokens = inNumTokens;
        hiddenSize = inHiddenSize;
        topK = inTopK;
        totalRows = inTotalRows;
        tokensPerCore = inTokensPerCore;

        expertGm.SetGlobalBuffer((__gm__ half *)expertOut, totalRows * hiddenSize);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)sortedIndices, totalRows);
        probsGm.SetGlobalBuffer((__gm__ half *)permutedProbs, totalRows);
        outGm.SetGlobalBuffer((__gm__ half *)out, numTokens * hiddenSize);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t tokenStart = coreId * tokensPerCore;
        uint32_t tokenEnd = tokenStart + tokensPerCore;
        if (tokenEnd > numTokens) {
            tokenEnd = numTokens;
        }

        for (uint32_t tokenId = tokenStart; tokenId < tokenEnd; ++tokenId) {
            for (uint32_t h = 0; h < hiddenSize; ++h) {
                float acc = 0.0f;
                for (uint32_t row = 0; row < totalRows; ++row) {
                    int32_t rowToken = indicesGm.GetValue(row);
                    if (static_cast<uint32_t>(rowToken) == tokenId) {
                        half pH = probsGm.GetValue(row);
                        half vH = expertGm.GetValue(row * hiddenSize + h);
                        float p = static_cast<float>(pH);
                        float v = static_cast<float>(vH);
                        acc += p * v;
                    }
                }
                outGm.SetValue(tokenId * hiddenSize + h, static_cast<half>(acc));
            }
        }
    }

private:
    GlobalTensor<half> expertGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<half> probsGm;
    GlobalTensor<half> outGm;
    uint32_t numTokens;
    uint32_t hiddenSize;
    uint32_t topK;
    uint32_t totalRows;
    uint32_t tokensPerCore;
};

__aicore__ inline void RunMoeTokenUnpermuteLite(GM_ADDR expertOut, GM_ADDR sortedIndices, GM_ADDR permutedProbs,
                                                GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelMoeTokenUnpermuteLite op;
    op.Init(expertOut, sortedIndices, permutedProbs, out,
            tilingData.numTokens, tilingData.hiddenSize, tilingData.topK, tilingData.totalRows, tilingData.tokensPerCore);
    op.Process();
}

extern "C" __global__ __aicore__ void moe_token_unpermute_lite(GM_ADDR expertOut, GM_ADDR sortedIndices, GM_ADDR permutedProbs,
                                                                GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        RunMoeTokenUnpermuteLite(expertOut, sortedIndices, permutedProbs, out, workspace, tiling);
    } else if (TILING_KEY_IS(1)) {
        RunMoeTokenUnpermuteLite(expertOut, sortedIndices, permutedProbs, out, workspace, tiling);
    }
}