#ifndef INFIX_TO_POSTFIX_LITE_TILING_H
#define INFIX_TO_POSTFIX_LITE_TILING_H
#include <cstdint>

struct InfixToPostfixLiteTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t exprLength;
};

#endif
