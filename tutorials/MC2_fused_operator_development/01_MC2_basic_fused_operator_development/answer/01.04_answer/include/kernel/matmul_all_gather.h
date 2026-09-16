#ifndef MATMUL_ALL_GATHER_KERNEL_H
#define MATMUL_ALL_GATHER_KERNEL_H

#include "utils.h"
#include "mte_comm.h"
#include "include/tensor.h"
#include "block/simple_block_scheduler.h"
#include "block/simple_block_mmad.h"

namespace MatmulAllGatherImpl {

using namespace AscendC;
using namespace asc::te;

template<typename DataType>
class MatmulAllGatherOp {
public:
    using AType = DataType;
    using BType = DataType;
    using CType = DataType;
    using BlockMmadType = SimpleBlockMmad<AType, BType, CType>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t>;

    __aicore__ inline MatmulAllGatherOp() {};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    uint64_t mDim_{0};
    uint64_t kDim_{0};
    uint64_t nDim_{0};
    uint64_t totalWinSize_{0};
    int64_t baseM_{128};
    int64_t baseK_{128};
    int64_t baseN_{128};

    GM_ADDR aGm_;
    GM_ADDR bGm_;
    GM_ADDR cGm_;

    __gm__ MatmulAllGatherTilingData* tilingDataOut_;
    MTECommunication<DataType> mteComm_;

    BlockMmadType mmadOp_;
};

template<typename DataType>
__aicore__ inline void MatmulAllGatherOp<DataType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *tPipe)
{
    mteComm_.InitHcclContextByAddr(mc2Context);
    tilingDataOut_ = (__gm__ MatmulAllGatherTilingData*)tilingGM;

    auto&& tInfo = tilingDataOut_->tilingInfo;
    mDim_ = tInfo.mDim;
    kDim_ = tInfo.kDim;
    nDim_ = tInfo.nDim;
    totalWinSize_ = tInfo.totalWinSize;
    baseM_ = tInfo.baseM;
    baseK_ = tInfo.baseK;
    baseN_ = tInfo.baseN;

    aGm_ = a;
    bGm_ = b;
    cGm_ = y;
    
    mteComm_.InitBuffer(tPipe);
    mteComm_.InitGMTensor(y, totalWinSize_);
    
    typename BlockMmadType::L1Params l1Params{static_cast<uint64_t>(baseK_), static_cast<uint64_t>(baseK_)};
    mmadOp_.Init(baseM_, baseN_, baseK_, l1Params);
    mmadOp_.UpdateK(kDim_);
}

template<typename DataType>
__aicore__ inline void MatmulAllGatherOp<DataType>::Process()
{
    int64_t m = static_cast<int64_t>(mDim_);
    int64_t n = static_cast<int64_t>(nDim_);
    int64_t k = static_cast<int64_t>(kDim_);
    
    if ASCEND_IS_AIC {
        auto layoutA = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<AType>>(m, k);
        auto layoutB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<BType>>(k, n);
        auto layoutC = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<CType>>(m, n);
        
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ AType*)aGm_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ BType*)bGm_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ CType*)mteComm_.localWinDataAddr_), layoutC);
        
        SimpleBlockScheduler bs(baseM_, baseN_);
        bs.UpdateNextProblem(m, n);
        
        BlockCoord tileCoord;
        while (bs.GetTileIdx(tileCoord)) {
            SimpleBlockScheduler::BlockShape singleShape = bs.GetBlockShape(tileCoord);
            
            int64_t tileM = Get<MNK_M>(singleShape);
            int64_t tileN = Get<MNK_N>(singleShape);
            
            if (tileM <= 0 || tileN <= 0) {
                continue;
            }
            
            int64_t mOffset = Get<MNK_M>(tileCoord) * baseM_;
            int64_t nOffset = Get<MNK_N>(tileCoord) * baseN_;
            
            auto gmBlockA = gmA(asc::te::make_coord(mOffset, 0), asc::te::make_shape(tileM, k));
            auto gmBlockB = gmB(asc::te::make_coord(0, nOffset), asc::te::make_shape(k, tileN));
            auto gmBlockC = gmC(asc::te::make_coord(mOffset, nOffset), asc::te::make_shape(tileM, tileN));
            
            typename BlockMmadType::BlockShape mmadShape{tileM, tileN, k};
            mmadOp_(gmBlockA, gmBlockB, gmBlockC, mmadShape);
        }
        return;
    }

    PipeBarrier<PIPE_ALL>();

    mteComm_.WriteStatusToWin();
    mteComm_.ReadStatus();
    mteComm_.CopyDataFromWin(mDim_, nDim_);
}

}

#endif
