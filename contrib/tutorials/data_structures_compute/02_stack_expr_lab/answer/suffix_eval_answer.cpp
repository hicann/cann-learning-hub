/*
 * SuffixEvalLite — 后缀表达式求值算子（参考答案）
 *
 * 核心思想：操作数入栈，运算符出栈计算后结果再入栈
 *   - 遇到操作数 → Push入栈
 *   - 遇到运算符 → Pop两个操作数，计算后Push结果
 *
 * Token编码：
 *   - 非负整数 → 操作数值
 *   - -1 = '+', -2 = '-', -3 = '*', -4 = '/'
 */
#include "suffix_eval_lite_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

#define MAX_OPND_STACK 128

class KernelSuffixEvalLite {
public:
    __aicore__ inline KernelSuffixEvalLite() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t tokenCount) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tokenCount = tokenCount;
        xGm.SetGlobalBuffer((__gm__ int32_t *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength / blockLength);
    }

    __aicore__ inline void Process() {
        uint32_t blockId = GetBlockIdx();
        uint32_t start = blockId * blockLength;

        uint32_t top = 0;

        for (uint32_t i = 0; i < tokenCount; i++) {
            int32_t token = xGm.GetValue(start + i);

            if (token >= 0) {
                if (top >= MAX_OPND_STACK) break;  // 栈满，提前终止
                ubOpndStack[top++] = (float)token;
            } else {
                if (top < 2) break;  // 操作数不足，提前终止
                float b = ubOpndStack[--top];
                float a = ubOpndStack[--top];
                float result = 0.0f;

                if (token == -1) result = a + b;
                else if (token == -2) result = a - b;
                else if (token == -3) result = a * b;
                else if (token == -4) {
                    if (b != 0.0f) result = a / b;
                    else result = 0.0f;
                }

                ubOpndStack[top++] = result;
            }
        }

        float finalResult = (top > 0) ? ubOpndStack[0] : 0.0f;
        yGm.SetValue(blockId, finalResult);
    }

private:
    GlobalTensor<int32_t> xGm;
    GlobalTensor<float> yGm;
    uint32_t totalLength, blockLength, tokenCount;
    float ubOpndStack[MAX_OPND_STACK];  // Local Memory 上的操作数栈
};

__aicore__ inline void RunSuffixEvalLite(GM_ADDR x, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelSuffixEvalLite op;
    op.Init(x, y, workspace, tilingData.totalLength, tilingData.blockLength,
            tilingData.tokenCount);
    op.Process();
}

extern "C" __global__ __aicore__ void suffix_eval_lite(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SuffixEvalLiteTilingData);
    RunSuffixEvalLite(x, y, workspace, tiling);
}
