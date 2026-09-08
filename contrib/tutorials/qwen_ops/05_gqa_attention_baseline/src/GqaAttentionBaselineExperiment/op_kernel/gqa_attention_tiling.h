#ifndef GQA_ATTENTION_BASELINE_TILING_H
#define GQA_ATTENTION_BASELINE_TILING_H

#include <cstdint>

#pragma pack(push, 1)
struct GqaAttentionBaselineTiling {
    uint32_t batch = 0;
    uint32_t queryHeads = 0;
    uint32_t kvHeads = 0;
    uint32_t queryLen = 0;
    uint32_t keyLen = 0;
    uint32_t headDim = 0;
    uint32_t totalQueries = 0;
    uint32_t coreNum = 1;
    uint32_t queriesPerCore = 1;
    uint32_t causal = 1;
    float scale = 1.0f;
};
#pragma pack(pop)

static_assert(sizeof(GqaAttentionBaselineTiling) == 44, "Unexpected GQA tiling size");
#endif
