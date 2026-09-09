#ifndef SUFFIX_EVAL_LITE_TILING_H
#define SUFFIX_EVAL_LITE_TILING_H
#include <cstdint>

struct SuffixEvalLiteTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tokenCount;
};

#endif
