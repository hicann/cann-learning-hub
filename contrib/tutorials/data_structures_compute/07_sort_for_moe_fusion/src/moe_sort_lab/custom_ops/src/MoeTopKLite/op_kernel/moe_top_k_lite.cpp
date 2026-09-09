#include "moe_top_k_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

/*
 * MoeTopKLite: min-heap top-K selection (O(E·logK)).
 * Flat arrays (no struct), all heap ops inlined, GM writes only after loop.
 */
static constexpr uint32_t MAX_E = 256;
static constexpr uint32_t MAX_K = 8;

class KernelMoeTopKLite {
public:
    __aicore__ inline KernelMoeTopKLite() {}
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
            float vals[MAX_E];
            for (uint32_t e = 0; e < numExperts; ++e) {
                vals[e] = static_cast<float>(logitsGm.GetValue(t * numExperts + e));
            }
            float hS[MAX_K]; int32_t hI[MAX_K]; uint32_t hN = 0;
            for (uint32_t e = 0; e < numExperts; ++e) {
                float cs = vals[e]; int32_t ci = static_cast<int32_t>(e);
                if (hN < topK) {
                    hS[hN] = cs; hI[hN] = ci;
                    int32_t idx = static_cast<int32_t>(hN);
                    while (idx > 0) {
                        int32_t p = (idx - 1) / 2;
                        if (!(hS[idx] < hS[p] || (hS[idx] == hS[p] && hI[idx] > hI[p]))) break;
                        float ts2 = hS[idx]; hS[idx] = hS[p]; hS[p] = ts2;
                        int32_t ti = hI[idx]; hI[idx] = hI[p]; hI[p] = ti;
                        idx = p;
                    }
                    hN++;
                } else {
                    if (cs > hS[0] || (cs == hS[0] && ci < hI[0])) {
                        hS[0] = cs; hI[0] = ci;
                        int32_t idx = 0;
                        while (true) {
                            int32_t l = 2*idx+1, r = 2*idx+2, w = idx;
                            if (l < static_cast<int32_t>(hN) && (hS[l] < hS[w] || (hS[l] == hS[w] && hI[l] > hI[w]))) w = l;
                            if (r < static_cast<int32_t>(hN) && (hS[r] < hS[w] || (hS[r] == hS[w] && hI[r] > hI[w]))) w = r;
                            if (w == idx) break;
                            float ts2 = hS[idx]; hS[idx] = hS[w]; hS[w] = ts2;
                            int32_t ti = hI[idx]; hI[idx] = hI[w]; hI[w] = ti;
                            idx = w;
                        }
                    }
                }
            }
            for (uint32_t i = 1; i < hN; ++i) {
                float ks = hS[i]; int32_t ki = hI[i];
                int32_t j = static_cast<int32_t>(i) - 1;
                while (j >= 0 && (ks > hS[j] || (ks == hS[j] && ki < hI[j]))) {
                    hS[j+1] = hS[j]; hI[j+1] = hI[j]; j--;
                }
                hS[j+1] = ks; hI[j+1] = ki;
            }
            float mv = hS[0]; for (uint32_t k = 1; k < hN; ++k) if (hS[k] < mv) mv = hS[k];
            float sm = 0.0f; float sh[MAX_K];
            for (uint32_t k = 0; k < hN; ++k) { sh[k] = hS[k] - mv + 1.0f; sm += sh[k]; }
            for (uint32_t k = 0; k < hN; ++k) {
                indicesGm.SetValue(t * topK + k, hI[k]);
                probsGm.SetValue(t * topK + k, static_cast<half>(sh[k] / sm));
            }
        }
    }
private:
    GlobalTensor<half> logitsGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<half> probsGm;
    uint32_t numTokens, numExperts, topK, tokensPerCore;
};

__aicore__ inline void RunMoeTopKLite(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMoeTopKLite op;
    op.Init(logits, topkIndices, topkProbs, tilingData.numTokens, tilingData.numExperts, tilingData.topK, tilingData.tokensPerCore);
    op.Process();
}

extern "C" __global__ __aicore__ void moe_top_k_lite(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs, GM_ADDR workspace, GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) { RunMoeTopKLite(logits, topkIndices, topkProbs, workspace, tiling); }
    else if (TILING_KEY_IS(1)) { RunMoeTopKLite(logits, topkIndices, topkProbs, workspace, tiling); }
}