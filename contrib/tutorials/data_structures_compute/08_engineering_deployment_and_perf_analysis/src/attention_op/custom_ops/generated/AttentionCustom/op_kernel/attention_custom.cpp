#include "kernel_operator.h"
#include "attention_custom_tiling.h"

// ---------------------------------------------------------------------------
// 朴素三步注意力算子（纯标量实现，多核 AIV 按行切分，无 workspace）
//
//   1) scores = Q x K^T   （标量点积）
//   2) P = softmax(scores * scale)     （逐行，max/exp/sum）
//   3) o = P x V          （标量点积）
//
// 实现说明：本环境（CANN 9.0.0 + 驱动 25.5.0）对 Ascend C 高级特性存在缺陷
// （MIX 任务崩溃、workspace 不可写、TQue/printf 异常、UB GetValue 与大循环
// 组合被错误编译等），因此本实现刻意只用"标量 GM 访问 + UB 普通数组"，
// 即最朴素、最稳定的 API 子集（与已验证可用的 StackExprOps 同构）。
//
// 数据流：
//   P 行留在 UB 普通数组（pRow），不落 GM；
//   scores 行在行内两遍计算（先算行并求 max，再 exp/sum 归一），不落 GM。
// ---------------------------------------------------------------------------
// 快速指数近似（softmax 用，输入范围约 [-20, 0]）
__aicore__ inline float FastExp(float x)
{
    if (x < -20.0f) x = -20.0f;
    if (x > 0.0f) x = 0.0f;
    float y = x * 1.4426950408889634f;      // log2(e)
    int yi = (int)y;
    float yf = y - static_cast<float>(yi);
    // 2^yf 多项式（yf ∈ [0,1)）
    float p = 1.0f + yf * (0.69314718056f + yf * (0.240226507f + yf * (0.05550410866f + yf * 0.009618129107f)));
    // 2^yi 位操作
    union { uint32_t u; float f; } bits;
    bits.u = static_cast<uint32_t>(yi + 127) << 23;
    return p * bits.f;
}

class AttentionKernel {
public:
    __aicore__ inline AttentionKernel() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kt, GM_ADDR v, GM_ADDR o,
                                const AttentionCustomTilingData& tilingData);
    __aicore__ inline void Process();

    AscendC::GlobalTensor<half> qGlobal;        // Q  [S, D]（阶段 A 后兼作 P 批暂存）
    AscendC::GlobalTensor<half> ktGlobal;       // Kᵀ [D, S]（预转置）
    AscendC::GlobalTensor<half> vGlobal;        // V  [S, D]
    AscendC::GlobalTensor<half> oGlobal;        // o  [S, D]

    uint32_t seqLen;
    uint32_t dim;
    float scale;

private:
    // 教学约束：seq_len ≤ 4096（栈帧限制：4096×4B + 64×4B ≈ 16.5KB < 32KB）
    static constexpr uint32_t kMaxSeqLen = 4096;
    float scoresRow[kMaxSeqLen];    // 一行 scores，softmax 后原地变 P（UB 普通数组）
    float oRow[64];                 // 一行 o（UB 普通数组，dim ≤ 64）
};

__aicore__ inline void AttentionKernel::Init(GM_ADDR q, GM_ADDR kt, GM_ADDR v, GM_ADDR o,
                                             const AttentionCustomTilingData& tilingData)
{
    this->seqLen = tilingData.seqLen;
    this->dim = tilingData.dim;
    this->scale = tilingData.scale;

    qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), seqLen * dim);
    ktGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(kt), seqLen * dim);
    vGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), seqLen * dim);
    oGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o), seqLen * dim);
}

__aicore__ inline void AttentionKernel::Process()
{
    // 行独立计算：每行完成 P = softmax(Q·Kᵀ·scale) 后立即算 o = P·V。
    // 行间无共享状态，多核按行切分（blockIdx 为核号，步进 blockDim）。
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockDim = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t row = blockIdx; row < seqLen; row += blockDim) {
        // ---------- 阶段 A：P = softmax(Q[row] · Kᵀ · scale) ----------
        // 两遍计算：scores 行原地计算并求 max → exp/sum 归一（原地变 P）
        float rowMax = -1.0e30f;
        for (uint32_t j = 0; j < seqLen; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < dim; ++k) {
                s += static_cast<float>(qGlobal.GetValue(row * dim + k))
                   * static_cast<float>(ktGlobal.GetValue(k * seqLen + j));
            }
            s *= scale;
            scoresRow[j] = s;
            if (s > rowMax) rowMax = s;
        }
        float rowSum = 0.0f;
        for (uint32_t j = 0; j < seqLen; ++j) {
            float e = FastExp(scoresRow[j] - rowMax);
            scoresRow[j] = e;
            rowSum += e;
        }
        float inv = 1.0f / rowSum;
        for (uint32_t j = 0; j < seqLen; ++j) {
            scoresRow[j] *= inv;
        }

        // ---------- 阶段 B：o = P x V ----------
        for (uint32_t j = 0; j < dim; ++j) {
            float s = 0.0f;
            for (uint32_t k = 0; k < seqLen; ++k) {
                s += scoresRow[k] * static_cast<float>(vGlobal.GetValue(k * dim + j));
            }
            oRow[j] = s;
        }
        for (uint32_t j = 0; j < dim; ++j) {
            oGlobal.SetValue(row * dim + j, static_cast<half>(oRow[j]));
        }
    }
}

extern "C" __global__ __aicore__ void attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AttentionCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AttentionKernel op;
    op.Init(q, k, v, o, tilingData);
    op.Process();
}
