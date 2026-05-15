#ifndef SIMPLE_BLOCK_MMAD_H
#define SIMPLE_BLOCK_MMAD_H

#include "kernel_operator.h"
#include "include/tensor.h"
#include "block/block_scheduler_utils.h"
#include "utils.h"

namespace MatmulAllGatherImpl {

constexpr uint64_t L1_BUFFER_NUM = 2UL;
constexpr uint8_t FINAL_ACCUMULATION = 3U;
constexpr uint8_t NON_FINAL_ACCUMULATION = 2U;
constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = 2U;

constexpr AscendC::Te::MmadTrait MMAD_TRAIT{0, false, false, true, AscendC::Te::MmadType::NORMAL};

struct MmadTraitConfig {
    using TraitType = AscendC::Te::MmadTrait;
    static constexpr const TraitType value = MMAD_TRAIT;
};

template <typename AType_, typename BType_, typename CType_>
class SimpleBlockMmad {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL1 = AscendC::Te::ZnLayoutFormat<BType>;

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / L1_BUFFER_NUM / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = L0C_SIZE / L1_BUFFER_NUM / sizeof(float);

    struct L1Params {
        uint64_t kAL1;
        uint64_t kBL1;
    };

    __aicore__ inline SimpleBlockMmad()
    {
#pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~SimpleBlockMmad()
    {
#pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(int64_t baseM, int64_t baseN, int64_t baseK, const L1Params& l1Params)
    {
        baseM_ = baseM;
        baseN_ = baseN;
        baseK_ = baseK;
        kAL1_ = l1Params.kAL1;
        kBL1_ = l1Params.kBL1;
        orderAL1BL1_ = kAL1_ >= kBL1_;
        
        AscendC::printf("Init: baseM=%ld, baseN=%ld, baseK=%ld, kAL1=%lu, kBL1=%lu, orderAL1BL1=%d\n",
            baseM_, baseN_, baseK_, kAL1_, kBL1_, orderAL1BL1_);
        
        aL1OneBuffer_ = baseM_ * kAL1_;
        bL1OneBuffer_ = baseN_ * kBL1_;
        
        AscendC::printf("aL1OneBuffer=%lu (elements), bL1OneBuffer=%lu (elements)\n", aL1OneBuffer_, bL1OneBuffer_);
        
        for (int32_t bufId = 0; bufId < L1_BUFFER_NUM; bufId++) {
            uint64_t l1Offset = (L1_SIZE >> 1) * (bufId & 1);
            l1BufferAOffset_[bufId] = l1Offset + aL1OneBuffer_ * sizeof(AType) * (bufId >> 1);
            l1BufferBOffset_[bufId] = l1Offset + aL1OneBuffer_ * sizeof(AType) * (L1_BUFFER_NUM >> 1) + bL1OneBuffer_ * sizeof(BType) * (bufId >> 1);
            AscendC::printf("bufId=%d: l1Offset=%lu, l1BufferAOffset=%lu, l1BufferBOffset=%lu\n",
                bufId, l1Offset, l1BufferAOffset_[bufId], l1BufferBOffset_[bufId]);
        }
    }

    __aicore__ inline void UpdateK(int64_t k)
    {
        k_ = k;
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmB, TensorC gmC, const BlockShape& singleShape)
    {
        Run(gmA, gmB, gmC, singleShape);
    }

private:
    template <typename TensorA>
    __aicore__ inline void CopyAInL1(
        TensorA gmA, uint64_t curM, uint64_t curGmAKL1, uint64_t aL1BufId, uint64_t kL1Offset)
    {
        AscendC::printf("CopyAInL1: curM=%lu, curGmAKL1=%lu, aL1BufId=%lu, kL1Offset=%lu, l1BufferAOffset=%lu\n",
            curM, curGmAKL1, aL1BufId, kL1Offset, l1BufferAOffset_[aL1BufId]);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto layoutAL1 = MakeLayoutAL1{}(curM, curGmAKL1);
        auto tensorAL1 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[aL1BufId]), layoutAL1);
        auto gmBlockA = gmA(AscendC::Te::MakeCoord(0, kL1Offset), AscendC::Te::MakeShape(curM, curGmAKL1));
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    }

    template <typename TensorB>
    __aicore__ inline void CopyBInL1(
        TensorB gmB, uint64_t curN, uint64_t curGmBKL1, uint64_t bL1BufId, uint64_t kL1Offset)
    {
        AscendC::printf("CopyBInL1: curN=%lu, curGmBKL1=%lu, bL1BufId=%lu, kL1Offset=%lu, l1BufferBOffset=%lu\n",
            curN, curGmBKL1, bL1BufId, kL1Offset, l1BufferBOffset_[bL1BufId]);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
        auto tensorBL1 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);
        auto gmBlockB = gmB(AscendC::Te::MakeCoord(kL1Offset, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1>
    __aicore__ inline void IterateK(
        TensorL0C tensorL0C, TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curM, uint64_t curN, uint64_t curGmAKL1,
        uint64_t curGmBKL1, uint64_t aL1BufId, uint64_t bL1BufId, uint64_t absKStart)
    {
        AscendC::printf("IterateK: curM=%lu, curN=%lu, curGmAKL1=%lu, curGmBKL1=%lu, absKStart=%lu, k_=%ld\n",
            curM, curN, curGmAKL1, curGmBKL1, absKStart, k_);
        
        auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});

        uint64_t minGmKL1 = Min(curGmAKL1, curGmBKL1);

        for (uint64_t kL0Offset = 0; kL0Offset < minGmKL1; kL0Offset += baseK_) {
            uint64_t curKL0 = (kL0Offset + baseK_ > minGmKL1) ? (minGmKL1 - kL0Offset) : baseK_;
            
            AscendC::printf("IterateK loop: kL0Offset=%lu, curKL0=%lu, l0PingPong=%lu\n", kL0Offset, curKL0, l0PingPong_);
            
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);

            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL0AmemPtr<AType>(l0Offset * sizeof(AType)), AscendC::Te::MakeNzLayout<AType>(curM, curKL0));
            auto tensorBlockAL1 = tensorAL1(
                AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curM, curKL0));
            AscendC::Te::Copy(copyL12L0, tensorAL0, tensorBlockAL1);

            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL0BmemPtr<BType>(l0Offset * sizeof(BType)), AscendC::Te::MakeZnLayout<BType>(curKL0, curN));
            auto tensorBlockBL1 = tensorBL1(
                AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, curN));
            AscendC::Te::Copy(copyL12L0, tensorBL0, tensorBlockBL1);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            uint8_t mmadUnitFlag = (absKStart + kL0Offset + baseK_ >= k_) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            bool mmadCmatrixInitVal = (absKStart + kL0Offset == 0);
            
            AscendC::printf("MMAD: curM=%lu, curN=%lu, curKL0=%lu, unitFlag=%u, initC=%d\n",
                curM, curN, curKL0, mmadUnitFlag, mmadCmatrixInitVal);
            
            AscendC::Te::MmadParams mmadParams(
                static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curKL0), mmadUnitFlag,
                mmadCmatrixInitVal);

            AscendC::Te::Mad(
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, MmadTraitConfig>>{},
                tensorL0C, tensorAL0, tensorBL0, mmadParams);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void Run(TensorA gmA, TensorB gmB, TensorC gmC, const BlockShape& singleShape)
    {
        uint64_t curM = Get<MNK_M>(singleShape);
        uint64_t curN = Get<MNK_N>(singleShape);
        
        AscendC::printf("Run: curM=%lu, curN=%lu, k_=%ld, orderAL1BL1=%d\n", curM, curN, k_, orderAL1BL1_);
        
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL0CmemPtr<float>(l0cOffset * sizeof(float)), AscendC::Te::MakeL0CLayout(curM, curN));
        
        if (orderAL1BL1_) {
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
                uint64_t aL1BufId = aL1LoopCnt_ & 1;
                uint64_t curGmAKL1 = (kOuter + kAL1_ > k_) ? (k_ - kOuter) : kAL1_;

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kOuter);

                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                    uint64_t bL1BufId = bL1LoopCnt_ & 1;
                    uint64_t curGmBKL1 = (kInner + kBL1_ > k_) ? (k_ - kInner) : kBL1_;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
                    CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kInner);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    
                    AscendC::printf("Run orderAL1BL1: kInner=%lu, aL1BufId=%lu, bL1BufId=%lu\n", kInner, aL1BufId, bL1BufId);
                    
                    auto layoutAL1 = MakeLayoutAL1{}(curM, curGmAKL1);
                    auto tensorAL1 = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[aL1BufId]), layoutAL1);
                    auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
                    auto tensorBL1 = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);

                    IterateK(
                        tensorL0C, tensorAL1, tensorBL1, curM, curN, curGmAKL1, curGmBKL1, aL1BufId, bL1BufId, kInner);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
                    bL1LoopCnt_++;
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                aL1LoopCnt_++;
            }
        } else {
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
                uint64_t bL1BufId = bL1LoopCnt_ & 1;
                uint64_t curGmBKL1 = (kOuter + kBL1_ > k_) ? (k_ - kOuter) : kBL1_;

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
                CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kOuter);

                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                    uint64_t aL1BufId = aL1LoopCnt_ & 1;
                    uint64_t curGmAKL1 = (kInner + kAL1_ > k_) ? (k_ - kInner) : kAL1_;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                    CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kInner);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    
                    AscendC::printf("Run !orderAL1BL1: kInner=%lu, aL1BufId=%lu, bL1BufId=%lu\n", kInner, aL1BufId, bL1BufId);
                    
                    auto layoutAL1 = MakeLayoutAL1{}(curM, curGmAKL1);
                    auto tensorAL1 = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[aL1BufId]), layoutAL1);
                    auto layoutBL1 = MakeLayoutBL1{}(curGmBKL1, curN);
                    auto tensorBL1 = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeL1memPtr<BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);

                    IterateK(
                        tensorL0C, tensorAL1, tensorBL1, curM, curN, curGmAKL1, curGmBKL1, aL1BufId, bL1BufId, kInner);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                    aL1LoopCnt_++;
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
                bL1LoopCnt_++;
            }
        }
        
        AscendC::printf("Fixpipe: l0cPingPong=%lu\n", l0cPingPong_);
        
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(copyL0C2GM, gmC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});

        l0cPingPong_++;
    }

private:
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};

    int64_t baseM_{16};
    int64_t baseN_{16};
    int64_t baseK_{16};
    int64_t k_{0};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};

    uint64_t aL1LoopCnt_{0};
    uint64_t bL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};

    bool orderAL1BL1_{false};
};

} // namespace MatmulAllGatherImpl

#endif