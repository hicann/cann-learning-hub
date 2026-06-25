#ifndef SIMPLE_BLOCK_SCHEDULER_H
#define SIMPLE_BLOCK_SCHEDULER_H

#include "kernel_operator.h"
#include "block_scheduler_utils.h"

namespace MatmulAllGatherImpl {

using BlockCoord = AscendC::Coord<int64_t, int64_t>;

class SimpleBlockScheduler {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline SimpleBlockScheduler(int64_t baseM, int64_t baseN) : baseM_(baseM), baseN_(baseN)
    {}

    __aicore__ inline void UpdateNextProblem(int64_t m, int64_t n)
    {
        m_ = m;
        n_ = n;
        mCnt_ = CeilDiv(m_, baseM_);
        nCnt_ = CeilDiv(n_, baseN_);
        mBaseTail_ = m_ - (mCnt_ - 1) * baseM_;
        nBaseTail_ = n_ - (nCnt_ - 1) * baseN_;
        totalCnt_ = mCnt_ * nCnt_;

        mainMWindow_ = WINDOW_LEN < mCnt_ ? WINDOW_LEN : mCnt_;
        mainRow_ = mCnt_ / mainMWindow_ - 1;
        tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;

        roundIdx_ = 0;
        round_ = CeilDiv(totalCnt_, static_cast<int64_t>(blockNum_));
        startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
        endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;

        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::printf(
                "UpdateNextProblem: m=%ld, n=%ld, mCnt=%ld, nCnt=%ld, totalCnt=%ld, round=%ld\n", m_, n_, mCnt_, nCnt_,
                totalCnt_, round_);
        }
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (round_ == 0 || roundIdx_ > round_ - 1) {
            return false;
        }

        int64_t index = blockIdx_ + roundIdx_ * blockNum_;

        if (blockIdx_ < startBlockIdx_) {
            index += blockNum_ - startBlockIdx_;
        } else {
            index -= startBlockIdx_;
        }

        if (index >= totalCnt_) {
            roundIdx_++;
            return false;
        }

        int64_t rowIdx = index / nCnt_ / mainMWindow_;
        if (rowIdx < mainRow_) {
            Get<MNK_M>(blockCoord) = rowIdx * mainMWindow_ + index % mainMWindow_;
            Get<MNK_N>(blockCoord) = (index / mainMWindow_) % nCnt_;
        } else {
            int64_t tailIndex = index - mainRow_ * mainMWindow_ * nCnt_;
            Get<MNK_M>(blockCoord) = mainRow_ * mainMWindow_ + tailIndex % tailWindow_;
            Get<MNK_N>(blockCoord) = (tailIndex / tailWindow_) % nCnt_;
        }

        if (rowIdx & 1) {
            Get<MNK_N>(blockCoord) = nCnt_ - 1 - Get<MNK_N>(blockCoord);
        }

        roundIdx_++;
        return true;
    }

    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord)
    {
        int64_t mIdx = Get<MNK_M>(blockCoord);
        int64_t nIdx = Get<MNK_N>(blockCoord);
        int64_t singleCoreM = mIdx != (mCnt_ - 1) ? baseM_ : mBaseTail_;
        int64_t singleCoreN = nIdx != (nCnt_ - 1) ? baseN_ : nBaseTail_;
        return {singleCoreM, singleCoreN, 0, 0};
    }

private:
    int64_t m_{0};
    int64_t n_{0};
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t mBaseTail_{0};
    int64_t nBaseTail_{0};
    int64_t mainMWindow_{0};
    int64_t tailWindow_{0};
    int64_t mainRow_{0};
    int64_t round_{0};
    int64_t roundIdx_{0};
    int64_t baseM_{128};
    int64_t baseN_{128};
    uint32_t blockNum_ = AscendC::GetBlockNum();
    uint32_t blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{blockNum_ - 1};
};

} // namespace MatmulAllGatherImpl

#endif