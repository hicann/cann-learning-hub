#include "moe_sort_quick_sort_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;
static constexpr uint32_t MAX_E = 256;
static constexpr uint32_t MAX_K = 8;
class KernelMoeSortQuickSortLite {
public:
    __aicore__ inline KernelMoeSortQuickSortLite() {}
    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs,
                                uint32_t inNumTokens, uint32_t inNumExperts, uint32_t inTopK, uint32_t inTokensPerCore) {
        numTokens = inNumTokens; numExperts = inNumExperts; topK = inTopK; tokensPerCore = inTokensPerCore;
        logitsGm.SetGlobalBuffer((__gm__ half *)logits, numTokens * numExperts);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices, numTokens * topK);
        probsGm.SetGlobalBuffer((__gm__ half *)topkProbs, numTokens * topK);
    }
    __aicore__ inline void Process() {
        uint32_t ts = GetBlockIdx() * tokensPerCore;
        uint32_t te = ts + tokensPerCore;
        if (te > numTokens) te = numTokens;
        for (uint32_t t = ts; t < te; ++t) {
            float v[MAX_E]; int32_t id[MAX_E];
            for (uint32_t e = 0; e < numExperts; ++e) { v[e] = static_cast<float>(logitsGm.GetValue(t*numExperts+e)); id[e] = static_cast<int32_t>(e); }
            int32_t sLo[32], sHi[32], sp = 0;
            sLo[sp] = 0; sHi[sp] = static_cast<int32_t>(numExperts) - 1; sp++;
            while (sp > 0) {
                sp--; int32_t lo = sLo[sp], hi = sHi[sp];
                if (lo >= hi) continue;
                float pv = v[hi]; int32_t pi = id[hi]; int32_t i = lo;
                for (int32_t j = lo; j < hi; ++j) {
                    if (v[j] > pv || (v[j] == pv && id[j] < pi)) {
                        float tv = v[i]; v[i] = v[j]; v[j] = tv;
                        int32_t ti = id[i]; id[i] = id[j]; id[j] = ti; ++i;
                    }
                }
                float tv = v[i]; v[i] = v[hi]; v[hi] = tv;
                int32_t ti = id[i]; id[i] = id[hi]; id[hi] = ti;
                if (sp < 30) { sLo[sp] = lo; sHi[sp] = i - 1; sp++; sLo[sp] = i + 1; sHi[sp] = hi; sp++; }
            }
            float sv[MAX_K]; for (uint32_t k = 0; k < topK; ++k) { sv[k] = v[k]; indicesGm.SetValue(t*topK+k, id[k]); }
            float mv = sv[0]; for (uint32_t k = 1; k < topK; ++k) if (sv[k] < mv) mv = sv[k];
            float sm = 0.0f; float sh[MAX_K]; for (uint32_t k = 0; k < topK; ++k) { sh[k] = sv[k] - mv + 1.0f; sm += sh[k]; }
            for (uint32_t k = 0; k < topK; ++k) probsGm.SetValue(t*topK+k, static_cast<half>(sh[k]/sm));
        }
    }
private:
    GlobalTensor<half> logitsGm; GlobalTensor<int32_t> indicesGm; GlobalTensor<half> probsGm;
    uint32_t numTokens, numExperts, topK, tokensPerCore;
};
__aicore__ inline void RunMoeSortQuickSortLite(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMoeSortQuickSortLite op;
    op.Init(logits, topkIndices, topkProbs, tilingData.numTokens, tilingData.numExperts, tilingData.topK, tilingData.tokensPerCore);
    op.Process();
}
extern "C" __global__ __aicore__ void moe_sort_quick_sort_lite(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs, GM_ADDR workspace, GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) { RunMoeSortQuickSortLite(logits, topkIndices, topkProbs, workspace, tiling); }
    else if (TILING_KEY_IS(1)) { RunMoeSortQuickSortLite(logits, topkIndices, topkProbs, workspace, tiling); }
}
