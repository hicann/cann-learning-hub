#include "infix_to_postfix_lite_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

#define MAX_OPTR_STACK 128
#define MAX_OUTPUT 256

class KernelInfixToPostfixLite {
public:
    __aicore__ inline KernelInfixToPostfixLite() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t exprLength) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->exprLength = exprLength;
        xGm.SetGlobalBuffer((__gm__ int8_t *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y, totalLength);
    }

    __aicore__ inline int32_t Priority(char op) {
        if (op == '+' || op == '-') return 1;
        if (op == '*' || op == '/') return 2;
        if (op == '(') return 0;
        return -1;
    }

    __aicore__ inline bool IsOperator(char ch) {
        return (ch == '+' || ch == '-' || ch == '*' || ch == '/' ||
                ch == '(' || ch == ')');
    }

    __aicore__ inline bool IsOperand(char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') ||
               (ch >= 'A' && ch <= 'Z');
    }

    __aicore__ inline void Process() {
        uint32_t blockId = GetBlockIdx();
        uint32_t start = blockId * blockLength;

        uint32_t optrTop = 0;
        ubOptrStack[optrTop++] = '#';

        uint32_t outIdx = 0;

        for (uint32_t i = 0; i < exprLength; i++) {
            char ch = (char)xGm.GetValue(start + i);

            if (ch == '#') break;

            if (IsOperand(ch)) {
                if (outIdx >= MAX_OUTPUT - 1) break;  // 输出缓冲区满，提前终止
                ubOutput[outIdx++] = ch;
            } else if (ch == '(') {
                if (optrTop >= MAX_OPTR_STACK - 1) break;  // 运算符栈满，提前终止
                ubOptrStack[optrTop++] = ch;
            } else if (ch == ')') {
                while (optrTop > 0 && ubOptrStack[optrTop - 1] != '(' &&
                       outIdx < MAX_OUTPUT - 1) {
                    ubOutput[outIdx++] = ubOptrStack[--optrTop];
                }
                if (optrTop > 0) optrTop--;
            } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/') {
                while (optrTop > 0 &&
                       Priority(ubOptrStack[optrTop - 1]) >= Priority(ch) &&
                       ubOptrStack[optrTop - 1] != '(' &&
                       outIdx < MAX_OUTPUT - 1) {
                    ubOutput[outIdx++] = ubOptrStack[--optrTop];
                }
                if (optrTop >= MAX_OPTR_STACK - 1) break;  // 运算符栈满，提前终止
                ubOptrStack[optrTop++] = ch;
            }
        }

        while (optrTop > 0 && ubOptrStack[optrTop - 1] != '#' &&
               outIdx < MAX_OUTPUT - 1) {
            ubOutput[outIdx++] = ubOptrStack[--optrTop];
        }

        for (uint32_t i = 0; i < outIdx; i++) {
            yGm.SetValue(start + i, (int8_t)ubOutput[i]);
        }
        yGm.SetValue(start + outIdx, (int8_t)'#');
    }

private:
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> yGm;
    uint32_t totalLength, blockLength, exprLength;
    char ubOptrStack[MAX_OPTR_STACK];
    char ubOutput[MAX_OUTPUT];
};

__aicore__ inline void RunInfixToPostfixLite(GM_ADDR x, GM_ADDR y,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelInfixToPostfixLite op;
    op.Init(x, y, workspace, tilingData.totalLength, tilingData.blockLength,
            tilingData.exprLength);
    op.Process();
}

extern "C" __global__ __aicore__ void infix_to_postfix_lite(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(InfixToPostfixLiteTilingData);
    RunInfixToPostfixLite(x, y, workspace, tiling);
}
