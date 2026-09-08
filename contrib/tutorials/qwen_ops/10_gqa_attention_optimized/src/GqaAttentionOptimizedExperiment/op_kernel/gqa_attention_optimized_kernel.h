#ifndef GQA_ATTENTION_OPTIMIZED_KERNEL_H
#define GQA_ATTENTION_OPTIMIZED_KERNEL_H

#include "kernel_operator.h"
#include "gqa_attention_tiling.h"

using namespace AscendC;

__aicore__ inline float GqaOptimizedExp(float x)
{
    if (x < -80.0f) return 0.0f;
    if (x > 0.0f) x = 0.0f;
    const float invLn2 = 1.4426950409f, ln2 = 0.6931471806f;
    const int32_t n = static_cast<int32_t>(x * invLn2) - 1;
    const float r = x - static_cast<float>(n) * ln2, r2 = r * r;
    const float poly = 1.0f + r + r2 * (0.5f + r * (0.16666667f + r * (0.04166667f + r * 0.00833333f)));
    float pow2 = 1.0f;
    for (int32_t i = 0; i > n; --i) pow2 *= 0.5f;
    return poly * pow2;
}

class KernelGqaAttentionOptimized {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output,
        uint32_t batch, uint32_t queryHeads, uint32_t kvHeads, uint32_t queryLen, uint32_t keyLen,
        uint32_t headDim, uint32_t coreNum, uint32_t queriesPerCore, uint32_t causal, float scale)
    {
        batch_ = batch; queryHeads_ = queryHeads; kvHeads_ = kvHeads; queryLen_ = queryLen; keyLen_ = keyLen;
        headDim_ = headDim; coreNum_ = coreNum; queriesPerCore_ = queriesPerCore; causal_ = causal; scale_ = scale;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(query), batch * queryHeads * queryLen * headDim);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(key), batch * kvHeads * keyLen * headDim);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value), batch * kvHeads * keyLen * headDim);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), batch * queryHeads * queryLen * headDim);
        const uint32_t bytes = headDim * sizeof(float);
        pipe_.InitBuffer(queryQue_, 1, bytes); pipe_.InitBuffer(keyQue_, 1, bytes); pipe_.InitBuffer(valueQue_, 1, bytes);
        pipe_.InitBuffer(productBuf_, bytes); pipe_.InitBuffer(reduceBuf_, bytes); pipe_.InitBuffer(accBuf_, bytes); pipe_.InitBuffer(tempBuf_, bytes);
        pipe_.InitBuffer(sumBuf_, 32); pipe_.InitBuffer(outputQue_, 1, bytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        if (core >= coreNum_) return;
        const uint32_t begin = core * queriesPerCore_, total = batch_ * queryHeads_ * queryLen_;
        uint32_t end = begin + queriesPerCore_; if (end > total) end = total;
        for (uint32_t index = begin; index < end; ++index) { CopyInQuery(index); Compute(index); CopyOut(index); }
    }

private:
    __aicore__ inline void CopyInQuery(uint32_t index)
    {
        const uint32_t qBase = index * headDim_;
        LocalTensor<float> query = queryQue_.AllocTensor<float>();
        DataCopy(query, queryGm_[qBase], headDim_);
        queryQue_.EnQue(query);
    }

    __aicore__ inline void Compute(uint32_t index)
    {
        const uint32_t qPos = index % queryLen_, qHead = (index / queryLen_) % queryHeads_, batch = index / (queryLen_ * queryHeads_);
        const uint32_t kvHead = qHead / (queryHeads_ / kvHeads_), kvBase = (batch * kvHeads_ + kvHead) * keyLen_ * headDim_;
        uint32_t validKeys = keyLen_;
        if (causal_ != 0) {
            const int32_t visible = static_cast<int32_t>(qPos) + static_cast<int32_t>(keyLen_) - static_cast<int32_t>(queryLen_) + 1;
            validKeys = visible <= 0 ? 0 : (static_cast<uint32_t>(visible) < keyLen_ ? static_cast<uint32_t>(visible) : keyLen_);
        }
        LocalTensor<float> query = queryQue_.DeQue<float>();
        LocalTensor<float> product = productBuf_.Get<float>(), reduce = reduceBuf_.Get<float>(), acc = accBuf_.Get<float>(), temp = tempBuf_.Get<float>(), sum = sumBuf_.Get<float>();
        Duplicate(acc, 0.0f, static_cast<int32_t>(headDim_));
        float maxScore = -3.402823e+38f, normalizer = 0.0f;
        // Online softmax fuses score, normalization and V accumulation: no SxS matrix is materialized.
        for (uint32_t keyIndex = 0; keyIndex < validKeys; ++keyIndex) {
            const uint32_t base = kvBase + keyIndex * headDim_;
            LocalTensor<float> keyIn = keyQue_.AllocTensor<float>();
            LocalTensor<float> valueIn = valueQue_.AllocTensor<float>();
            DataCopy(keyIn, keyGm_[base], headDim_);
            DataCopy(valueIn, valueGm_[base], headDim_);
            keyQue_.EnQue(keyIn);
            valueQue_.EnQue(valueIn);
            LocalTensor<float> key = keyQue_.DeQue<float>();
            LocalTensor<float> value = valueQue_.DeQue<float>();
            Mul(product, query, key, static_cast<int32_t>(headDim_));
            ReduceSum(sum, product, reduce, static_cast<int32_t>(headDim_));
            const float score = sum.GetValue(0) * scale_, nextMax = score > maxScore ? score : maxScore;
            const float oldFactor = GqaOptimizedExp(maxScore - nextMax), newFactor = GqaOptimizedExp(score - nextMax);
            normalizer = normalizer * oldFactor + newFactor;
            Muls(acc, acc, oldFactor, static_cast<int32_t>(headDim_));
            Muls(temp, value, newFactor, static_cast<int32_t>(headDim_));
            Add(acc, acc, temp, static_cast<int32_t>(headDim_));
            maxScore = nextMax;
            keyQue_.FreeTensor(key);
            valueQue_.FreeTensor(value);
        }
        LocalTensor<float> out = outputQue_.AllocTensor<float>();
        Muls(out, acc, normalizer > 0.0f ? 1.0f / normalizer : 0.0f, static_cast<int32_t>(headDim_));
        outputQue_.EnQue(out);
        queryQue_.FreeTensor(query);
    }

    __aicore__ inline void CopyOut(uint32_t index)
    {
        LocalTensor<float> out = outputQue_.DeQue<float>();
        DataCopy(outputGm_[index * headDim_], out, headDim_);
        outputQue_.FreeTensor(out);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> queryQue_, keyQue_, valueQue_;
    TBuf<TPosition::VECCALC> productBuf_, reduceBuf_, accBuf_, tempBuf_, sumBuf_;
    TQue<TPosition::VECOUT, 1> outputQue_;
    GlobalTensor<float> queryGm_, keyGm_, valueGm_, outputGm_;
    uint32_t batch_ = 0, queryHeads_ = 0, kvHeads_ = 0, queryLen_ = 0, keyLen_ = 0, headDim_ = 0, coreNum_ = 1, queriesPerCore_ = 1, causal_ = 1;
    float scale_ = 1.0f;
};
#endif
