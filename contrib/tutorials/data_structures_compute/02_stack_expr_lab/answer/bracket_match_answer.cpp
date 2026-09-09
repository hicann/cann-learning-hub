/*
 * BracketMatchLite — 括号匹配算子（参考答案）
 *
 * 核心思想：用 Kernel 类成员数组在 Local Memory 上模拟顺序栈
 *   - 左括号 → Push: stack[top++] = ch
 *   - 右括号 → Pop配对: expected = stack[--top]
 *   - 三种失配：①右括号时栈空 ②类型不匹配 ③扫描结束栈不空
 */
#include "bracket_match_lite_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

#define MAX_STACK_SIZE 256

class KernelBracketMatchLite {
public:
    __aicore__ inline KernelBracketMatchLite() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t exprLength) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->exprLength = exprLength;
        xGm.SetGlobalBuffer((__gm__ int8_t *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ int32_t *)y, totalLength / blockLength);
    }

    __aicore__ inline void Process() {
        uint32_t blockId = GetBlockIdx();
        uint32_t start = blockId * blockLength;

        uint32_t top = 0;
        uint32_t status = 0;

        for (uint32_t i = 0; i < exprLength; i++) {
            char ch = (char)xGm.GetValue(start + i);

            if (ch == '(' || ch == '[' || ch == '{') {
                if (top >= MAX_STACK_SIZE) {
                    status = 4;
                    break;
                }
                ubStack[top++] = ch;
            } else if (ch == ')' || ch == ']' || ch == '}') {
                if (top == 0) {
                    status = 1;
                    break;
                }
                char expected = ubStack[--top];
                if ((ch == ')' && expected != '(') ||
                    (ch == ']' && expected != '[') ||
                    (ch == '}' && expected != '{')) {
                    status = 2;
                    break;
                }
            }
        }

        if (status == 0 && top != 0) {
            status = 3;
        }

        yGm.SetValue(blockId, (int32_t)status);
    }

private:
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int32_t> yGm;
    uint32_t totalLength, blockLength, exprLength;
    char ubStack[MAX_STACK_SIZE];  // Local Memory 上的栈空间
};

__aicore__ inline void RunBracketMatchLite(GM_ADDR x, GM_ADDR y,
                                            GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelBracketMatchLite op;
    op.Init(x, y, workspace, tilingData.totalLength, tilingData.blockLength,
            tilingData.exprLength);
    op.Process();
}

extern "C" __global__ __aicore__ void bracket_match_lite(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BracketMatchLiteTilingData);
    RunBracketMatchLite(x, y, workspace, tiling);
}
