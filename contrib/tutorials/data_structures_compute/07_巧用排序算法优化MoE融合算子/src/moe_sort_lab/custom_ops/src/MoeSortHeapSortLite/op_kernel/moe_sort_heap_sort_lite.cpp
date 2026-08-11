#include "moe_sort_heap_sort_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;
static constexpr uint32_t MAX_E = 256;
static constexpr uint32_t MAX_K = 8;
class KernelMoeSortHeapSortLite {
public:
    __aicore__ inline KernelMoeSortHeapSortLite() {}
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
            float hv[MAX_E]; int32_t hi[MAX_E]; uint32_t hs = numExperts;
            for (uint32_t e = 0; e < numExperts; ++e) { hv[e] = static_cast<float>(logitsGm.GetValue(t*numExperts+e)); hi[e] = static_cast<int32_t>(e); }
            for (int32_t i = static_cast<int32_t>(hs/2)-1; i >= 0; --i) { int32_t idx=i; while(true){int32_t la=idx,l=2*idx+1,r=2*idx+2;
                if(l<static_cast<int32_t>(hs)&&(hv[l]>hv[la]||(hv[l]==hv[la]&&hi[l]<hi[la])))la=l;
                if(r<static_cast<int32_t>(hs)&&(hv[r]>hv[la]||(hv[r]==hv[la]&&hi[r]<hi[la])))la=r;
                if(la==idx)break;float tv=hv[idx];hv[idx]=hv[la];hv[la]=tv;int32_t ti=hi[idx];hi[idx]=hi[la];hi[la]=ti;idx=la;}}
            float sv[MAX_K];
            for (uint32_t k = 0; k < topK; ++k) {
                sv[k]=hv[0]; indicesGm.SetValue(t*topK+k,hi[0]);
                float tv=hv[0];hv[0]=hv[hs-1];hv[hs-1]=tv;int32_t ti=hi[0];hi[0]=hi[hs-1];hi[hs-1]=ti;hs--;
                int32_t idx=0;while(true){int32_t la=idx,l=2*idx+1,r=2*idx+2;
                if(l<static_cast<int32_t>(hs)&&(hv[l]>hv[la]||(hv[l]==hv[la]&&hi[l]<hi[la])))la=l;
                if(r<static_cast<int32_t>(hs)&&(hv[r]>hv[la]||(hv[r]==hv[la]&&hi[r]<hi[la])))la=r;
                if(la==idx)break;float tv2=hv[idx];hv[idx]=hv[la];hv[la]=tv2;int32_t ti2=hi[idx];hi[idx]=hi[la];hi[la]=ti2;idx=la;}}
            float mv=sv[0];for(uint32_t k=1;k<topK;++k)if(sv[k]<mv)mv=sv[k];
            float sm=0.0f;float sh[MAX_K];for(uint32_t k=0;k<topK;++k){sh[k]=sv[k]-mv+1.0f;sm+=sh[k];}
            for(uint32_t k=0;k<topK;++k)probsGm.SetValue(t*topK+k,static_cast<half>(sh[k]/sm));
        }
    }
private:
    GlobalTensor<half> logitsGm; GlobalTensor<int32_t> indicesGm; GlobalTensor<half> probsGm;
    uint32_t numTokens, numExperts, topK, tokensPerCore;
};
__aicore__ inline void RunMoeSortHeapSortLite(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMoeSortHeapSortLite op;
    op.Init(logits, topkIndices, topkProbs, tilingData.numTokens, tilingData.numExperts, tilingData.topK, tilingData.tokensPerCore);
    op.Process();
}
extern "C" __global__ __aicore__ void moe_sort_heap_sort_lite(GM_ADDR logits, GM_ADDR topkIndices, GM_ADDR topkProbs, GM_ADDR workspace, GM_ADDR tiling) {
    if (TILING_KEY_IS(0)) { RunMoeSortHeapSortLite(logits, topkIndices, topkProbs, workspace, tiling); }
    else if (TILING_KEY_IS(1)) { RunMoeSortHeapSortLite(logits, topkIndices, topkProbs, workspace, tiling); }
}
