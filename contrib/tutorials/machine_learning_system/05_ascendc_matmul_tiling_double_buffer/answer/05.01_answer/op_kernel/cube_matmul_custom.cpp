// ============================== 参考答案 ==============================
//   本文件是 op_kernel/cube_matmul_custom.cpp 的完整实现，五处 TODO(2–6) 已补全。
//   建议先独立完成后再对照阅读。
//   除这些 TODO 区块外，本文件与 Notebook 第 7 节写入的版本完全一致。
// ====================================================================

#include "kernel_operator.h"
#include "experiment_config.h"

using namespace AscendC;

template <int32_t BUFFER_NUM>
class KernelMmadTiled {
private:
    // ---------------- 答案 TODO 5：队列深度取编译期 Buffer 模式 ----------------
    // Single Buffer 用 1 个槽位，Double Buffer 用 2 个；A1/B1 必须同深度，
    // 保证一组 A/B Tile 成对搬运与消费。
    static constexpr int32_t A1B1_QUEUE_DEPTH = BUFFER_NUM;

public:
    __aicore__ inline KernelMmadTiled() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K,
                                uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t mTiles,
                                uint32_t nTiles, uint32_t kTiles, uint32_t usedCoreNum)
    {
        M_ = M;
        N_ = N;
        K_ = K;

        baseM_ = baseM;
        baseN_ = baseN;
        baseK_ = baseK;

        mTiles_ = mTiles;
        nTiles_ = nTiles;
        kTiles_ = kTiles;
        usedCoreNum_ = usedCoreNum;

        mBlocks_ = baseM_ / C0;
        kBlocks_ = baseK_ / C0;
        nFragments_ = baseN_ / FRAGMENT_N;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a), M_ * K_);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), K_ * N_);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), M_ * N_);

        const uint32_t aTileElements = baseM_ * baseK_;
        const uint32_t bTileElements = baseK_ * baseN_;
        const uint32_t bFragmentElements = baseK_ * FRAGMENT_N;
        const uint32_t cFragmentElements = baseM_ * FRAGMENT_N;

        // 答案 TODO 5（续）：A1/B1 队列按 A1B1_QUEUE_DEPTH 申请，与 Buffer 模式一致。
        pipe_.InitBuffer(a1Queue_, A1B1_QUEUE_DEPTH, aTileElements * sizeof(half));
        pipe_.InitBuffer(b1Queue_, A1B1_QUEUE_DEPTH, bTileElements * sizeof(half));

        pipe_.InitBuffer(a2Queue_, 1, aTileElements * sizeof(half));
        pipe_.InitBuffer(b2Queue_, 1, bFragmentElements * sizeof(half));
        pipe_.InitBuffer(c1Queue_, 1, cFragmentElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t outputTileCount = mTiles_ * nTiles_;

        for (uint32_t outputTile = GetBlockIdx(); outputTile < outputTileCount;
             outputTile += usedCoreNum_) {
            // ---------------- 答案 TODO 2：一维 outputTile 映射为 (mTile, nTile) ----------------
            // 行主序展开：M 为行、N 为列。每个 Core 从 GetBlockIdx() 起步，
            // 循环内 outputTile += usedCoreNum_，故输出 Tile 多于 Core 时仍能全覆盖。
            const uint32_t mTile = outputTile / nTiles_;
            const uint32_t nTile = outputTile % nTiles_;

            ComputeOutputTile(mTile, nTile);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t mTile, uint32_t nTile, uint32_t kTile)
    {
        LocalTensor<half> a1 = a1Queue_.template AllocTensor<half>();
        LocalTensor<half> b1 = b1Queue_.template AllocTensor<half>();

        const uint32_t mStart = mTile * baseM_;
        const uint32_t nStart = nTile * baseN_;
        const uint32_t kStart = kTile * baseK_;

        // ---------------- 答案 TODO 3（A/B）：行主序 GM(ND) 偏移 ----------------
        // 元素 (row, col) 的一维偏移为 row * rowStride + col。
        // A 的行宽是 K_，B 的行宽是 N_。
        const uint32_t aOffset = mStart * K_ + kStart;
        const uint32_t bOffset = kStart * N_ + nStart;

        Nd2NzParams aParams{};
        aParams.ndNum = 1;
        aParams.nValue = baseM_;
        aParams.dValue = baseK_;
        aParams.srcNdMatrixStride = 0;
        aParams.srcDValue = K_;
        aParams.dstNzC0Stride = baseM_;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;

        Nd2NzParams bParams{};
        bParams.ndNum = 1;
        bParams.nValue = baseK_;
        bParams.dValue = baseN_;
        bParams.srcNdMatrixStride = 0;
        bParams.srcDValue = N_;
        bParams.dstNzC0Stride = baseK_;
        bParams.dstNzNStride = 1;
        bParams.dstNzMatrixStride = 0;

        DataCopy(a1, aGm_[aOffset], aParams);
        DataCopy(b1, bGm_[bOffset], bParams);

        a1Queue_.EnQue(a1);
        b1Queue_.EnQue(b1);
    }

    __aicore__ inline void SplitA(const LocalTensor<half>& a1)
    {
        LocalTensor<half> a2 = a2Queue_.template AllocTensor<half>();

        uint32_t srcOffset = 0;
        uint32_t dstOffset = 0;

        for (uint16_t mBlock = 0; mBlock < mBlocks_; ++mBlock) {
            LoadData2DParams params{};
            params.repeatTimes = kBlocks_;
            params.srcStride = mBlocks_;
            params.ifTranspose = false;

            LoadData(a2[dstOffset], a1[srcOffset], params);

            srcOffset += C0 * C0;
            dstOffset += baseK_ * C0;
        }

        a2Queue_.EnQue(a2);
    }

    __aicore__ inline void SplitB(const LocalTensor<half>& b1, uint16_t fragment)
    {
        LocalTensor<half> b2 = b2Queue_.template AllocTensor<half>();

        LoadData2DParams params{};
        params.repeatTimes = kBlocks_;
        params.srcStride = 1;
        params.ifTranspose = true;

        const uint32_t srcOffset = static_cast<uint32_t>(fragment) * baseK_ * FRAGMENT_N;

        LoadData(b2, b1[srcOffset], params);

        b2Queue_.EnQue(b2);
    }

    __aicore__ inline void MmadKTile(const LocalTensor<float>& c1, const LocalTensor<half>& a2,
                                     const LocalTensor<half>& b2, uint32_t kTile)
    {
        MmadParams params{};
        params.m = baseM_;
        params.n = FRAGMENT_N;
        params.k = baseK_;
        params.cmatrixSource = false;

        // ---------------- 答案 TODO 4：CO1 中的 K Tile 累加 ----------------
        // 第一块 K Tile（kTile == 0）从零初始化 CO1；
        // 后续 K Tile 以已有 CO1 为初值继续累加。
        params.cmatrixInitVal = (kTile == 0);

        Mmad(c1, a2, b2, params);

        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void CopyOut(const LocalTensor<float>& c1, uint32_t mTile, uint32_t nTile,
                                   uint16_t fragment)
    {
        const uint32_t mStart = mTile * baseM_;
        const uint32_t nStart = nTile * baseN_;

        // ---------------- 答案 TODO 3（C）：行主序 C 偏移（含 16 列 fragment）----------------
        // C 的行宽是 N_；再加上本 fragment 在 N 方向的 16 列偏移。
        const uint32_t cOffset = mStart * N_ + nStart +
                                 static_cast<uint32_t>(fragment) * FRAGMENT_N;

        FixpipeParamsV220 params{};
        params.nSize = FRAGMENT_N;
        params.mSize = baseM_;
        params.srcStride = baseM_;
        params.dstStride = N_;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        params.quantPre = QuantMode_t::NoQuant;

        Fixpipe(cGm_[cOffset], c1, params);
    }

    __aicore__ inline void ComputeFragment(uint32_t mTile, uint32_t nTile, uint16_t fragment)
    {
        LocalTensor<float> c1 = c1Queue_.template AllocTensor<float>();

        CopyIn(mTile, nTile, 0);

        if (BUFFER_NUM == 2 && kTiles_ > 1) {
            // ---------------- 答案 TODO 6（初始填充）：进入 K 循环前预取第二个 Tile ----------------
            // 上面已 CopyIn(Tile 0)，这里再填 Tile 1，凑满两个队列槽位。
            CopyIn(mTile, nTile, 1);
        }

        for (uint32_t kTile = 0; kTile < kTiles_; ++kTile) {
            LocalTensor<half> a1 = a1Queue_.template DeQue<half>();
            LocalTensor<half> b1 = b1Queue_.template DeQue<half>();

            SplitA(a1);
            SplitB(b1, fragment);

            a1Queue_.FreeTensor(a1);
            b1Queue_.FreeTensor(b1);

            // ---------------- 答案 TODO 6（稳态预取）：补入未来的 kTile + 2 ----------------
            // 当前消费 kTile，kTile + 1 已在第二个槽位；释放当前槽位后预取 kTile + 2，
            // 始终保持两个槽位满。再取当前 Tile 或 kTile + 1 会造成重复数据、破坏流水。
            if (BUFFER_NUM == 2) {
                if (kTile + 2 < kTiles_) {
                    CopyIn(mTile, nTile, kTile + 2);
                }
            }

            LocalTensor<half> a2 = a2Queue_.template DeQue<half>();
            LocalTensor<half> b2 = b2Queue_.template DeQue<half>();

            MmadKTile(c1, a2, b2, kTile);

            a2Queue_.FreeTensor(a2);
            b2Queue_.FreeTensor(b2);

            if (BUFFER_NUM == 1 && kTile + 1 < kTiles_) {
                CopyIn(mTile, nTile, kTile + 1);
            }
        }

        c1Queue_.EnQue(c1);
        LocalTensor<float> c1Ready = c1Queue_.template DeQue<float>();

        CopyOut(c1Ready, mTile, nTile, fragment);

        c1Queue_.FreeTensor(c1Ready);
    }

    __aicore__ inline void ComputeOutputTile(uint32_t mTile, uint32_t nTile)
    {
        for (uint16_t fragment = 0; fragment < nFragments_; ++fragment) {
            ComputeFragment(mTile, nTile, fragment);
        }
    }

private:
    static constexpr uint16_t C0 = 16;
    static constexpr uint16_t FRAGMENT_N = 16;

    TPipe pipe_;

    TQue<TPosition::A1, A1B1_QUEUE_DEPTH> a1Queue_;
    TQue<TPosition::B1, A1B1_QUEUE_DEPTH> b1Queue_;
    TQue<TPosition::A2, 1> a2Queue_;
    TQue<TPosition::B2, 1> b2Queue_;
    TQue<TPosition::CO1, 1> c1Queue_;

    GlobalTensor<half> aGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<float> cGm_;

    uint32_t M_ = 0;
    uint32_t N_ = 0;
    uint32_t K_ = 0;

    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
    uint32_t baseK_ = 0;

    uint32_t mTiles_ = 0;
    uint32_t nTiles_ = 0;
    uint32_t kTiles_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint16_t mBlocks_ = 0;
    uint16_t kBlocks_ = 0;
    uint16_t nFragments_ = 0;
};

extern "C" __global__ __aicore__ void cube_matmul_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K,
    uint32_t mTiles, uint32_t nTiles, uint32_t kTiles, uint32_t usedCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    if (M == 0 || N == 0 || K == 0 || mTiles == 0 || nTiles == 0 || kTiles == 0 ||
        usedCoreNum == 0 || M % LAB_BASE_M != 0 || N % LAB_BASE_N != 0 ||
        K % LAB_BASE_K != 0 || mTiles != M / LAB_BASE_M || nTiles != N / LAB_BASE_N ||
        kTiles != K / LAB_BASE_K) {
        return;
    }

    KernelMmadTiled<LAB_BUFFER_NUM> kernel;

    kernel.Init(a, b, c, M, N, K, LAB_BASE_M, LAB_BASE_N, LAB_BASE_K, mTiles, nTiles, kTiles,
                usedCoreNum);

    kernel.Process();
}
