#ifndef GQA_ATTENTION_BASELINE_KERNEL_H
#define GQA_ATTENTION_BASELINE_KERNEL_H

#include "kernel_operator.h"
#include "gqa_attention_tiling.h"

using namespace AscendC;

// Range reduction keeps the baseline self-contained and deliberately scalar.
__aicore__ inline float GqaBaselineExp(float x)
{
    if (x < -80.0f) return 0.0f;
    if (x > 0.0f) x = 0.0f;
    const float invLn2 = 1.4426950409f;
    const float ln2 = 0.6931471806f;
    const int32_t n = static_cast<int32_t>(x * invLn2) - 1;
    const float r = x - static_cast<float>(n) * ln2;
    const float r2 = r * r;
    const float poly = 1.0f + r + r2 * (0.5f + r * (0.16666667f + r * (0.04166667f + r * 0.00833333f)));
    float pow2 = 1.0f;
    for (int32_t i = 0; i > n; --i) pow2 *= 0.5f;
    return poly * pow2;
}

class KernelGqaAttentionBaseline {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output,
        uint32_t batch, uint32_t queryHeads, uint32_t kvHeads, uint32_t queryLen,
        uint32_t keyLen, uint32_t headDim, uint32_t coreNum, uint32_t queriesPerCore,
        uint32_t causal, float scale)
    {
        batch_ = batch; queryHeads_ = queryHeads; kvHeads_ = kvHeads; queryLen_ = queryLen;
        keyLen_ = keyLen; headDim_ = headDim; coreNum_ = coreNum; queriesPerCore_ = queriesPerCore;
        causal_ = causal; scale_ = scale;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(query), batch * queryHeads * queryLen * headDim);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(key), batch * kvHeads * keyLen * headDim);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value), batch * kvHeads * keyLen * headDim);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), batch * queryHeads * queryLen * headDim);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        if (core >= coreNum_) return;
        const uint32_t begin = core * queriesPerCore_;
        const uint32_t total = batch_ * queryHeads_ * queryLen_;
        uint32_t end = begin + queriesPerCore_;
        if (end > total) end = total;
        for (uint32_t index = begin; index < end; ++index) ComputeOne(index);
    }

private:
    __aicore__ inline void ComputeOne(uint32_t index)
    {
        const uint32_t qPos = index % queryLen_;
        const uint32_t qHead = (index / queryLen_) % queryHeads_;
        const uint32_t batch = index / (queryLen_ * queryHeads_);
        const uint32_t kvHead = qHead / (queryHeads_ / kvHeads_);
        const uint32_t qBase = ((batch * queryHeads_ + qHead) * queryLen_ + qPos) * headDim_;
        const uint32_t kvBase = (batch * kvHeads_ + kvHead) * keyLen_ * headDim_;
        uint32_t validKeys = keyLen_;
        if (causal_ != 0) {
            const int32_t visible = static_cast<int32_t>(qPos) + static_cast<int32_t>(keyLen_) - static_cast<int32_t>(queryLen_) + 1;
            validKeys = visible <= 0 ? 0 : (static_cast<uint32_t>(visible) < keyLen_ ? static_cast<uint32_t>(visible) : keyLen_);
        }

        float maxScore = -3.402823e+38f;
        float normalizer = 0.0f;
        for (uint32_t d = 0; d < headDim_; ++d) outputGm_.SetValue(qBase + d, 0.0f);
        for (uint32_t keyIndex = 0; keyIndex < validKeys; ++keyIndex) {
            const uint32_t kBase = kvBase + keyIndex * headDim_;
            float score = 0.0f;
            for (uint32_t d = 0; d < headDim_; ++d) score += queryGm_.GetValue(qBase + d) * keyGm_.GetValue(kBase + d);
            score *= scale_;
            const float nextMax = score > maxScore ? score : maxScore;
            const float oldFactor = GqaBaselineExp(maxScore - nextMax);
            const float newFactor = GqaBaselineExp(score - nextMax);
            normalizer = normalizer * oldFactor + newFactor;
            for (uint32_t d = 0; d < headDim_; ++d) {
                const float oldValue = outputGm_.GetValue(qBase + d);
                outputGm_.SetValue(qBase + d, oldValue * oldFactor + newFactor * valueGm_.GetValue(kBase + d));
            }
            maxScore = nextMax;
        }
        const float invNorm = normalizer > 0.0f ? 1.0f / normalizer : 0.0f;
        for (uint32_t d = 0; d < headDim_; ++d) outputGm_.SetValue(qBase + d, outputGm_.GetValue(qBase + d) * invNorm);
    }

    GlobalTensor<float> queryGm_, keyGm_, valueGm_, outputGm_;
    uint32_t batch_ = 0, queryHeads_ = 0, kvHeads_ = 0, queryLen_ = 0, keyLen_ = 0, headDim_ = 0;
    uint32_t coreNum_ = 1, queriesPerCore_ = 1, causal_ = 1;
    float scale_ = 1.0f;
};
#endif
