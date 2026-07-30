#ifndef TENSOR_MMAD_STEPK_ANSWER_KERNEL_H
#define TENSOR_MMAD_STEPK_ANSWER_KERNEL_H

#include "basic_api/kernel_operator_block_sync_intf.h"
#include "tensor_api/tensor.h"

template <
    int32_t M, int32_t N, int32_t K,
    int32_t singleM, int32_t singleN, int32_t singleK,
    int32_t baseM, int32_t baseN, int32_t baseK,
    int32_t stepK>
__cube__ __global__ void TensorMmadStepKAnswerKernel(__gm__ half *x, __gm__ half *y, __gm__ half *z)
{
    using namespace AscendC::Te;
    static_assert(M % singleM == 0 && N % singleN == 0 && K % singleK == 0);
    static_assert(singleM % baseM == 0 && singleN % baseN == 0 && singleK % baseK == 0);
    static_assert(singleK % (baseK * stepK) == 0);

    constexpr uint32_t mCoreLoop = M / singleM;
    constexpr uint32_t nCoreLoop = N / singleN;
    constexpr uint32_t mLoop = singleM / baseM;
    constexpr uint32_t nLoop = singleN / baseN;
    constexpr uint32_t kLoop = singleK / baseK;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t mCoreIdx = blockIdx % mCoreLoop;
    uint32_t nCoreIdx = blockIdx / mCoreLoop;

    auto gmATensor = MakeTensor(MakeMemPtr(x), MakeFrameLayout<NDLayoutPtn>(M, K));
    auto gmBTensor = MakeTensor(MakeMemPtr(y), MakeFrameLayout<DNLayoutPtn>(K, N));
    auto gmCTensor = MakeTensor(MakeMemPtr(z), MakeFrameLayout<NDLayoutPtn>(M, N));

    auto gmASingle = gmATensor.Slice(MakeCoord(mCoreIdx * singleM, 0), MakeShape(singleM, singleK));
    auto gmBSingle = gmBTensor.Slice(MakeCoord(0, nCoreIdx * singleN), MakeShape(singleK, singleN));
    auto gmCSingle = gmCTensor.Slice(MakeCoord(mCoreIdx * singleM, nCoreIdx * singleN), MakeShape(singleM, singleN));

    __cbuf__ half l1ABuf[stepK * baseM * baseK];
    __cbuf__ half l1BBuf[stepK * baseK * baseN];
    __ca__ half l0ABuf[baseM * baseK];
    __cb__ half l0BBuf[baseK * baseN];
    __cc__ float l0CBuf[baseM * baseN];

    auto copyGM2L1Atom = MakeCopy(CopyGM2L1{}, CopyGM2L1TraitDefault{});
    auto copyL12L0AAtom = MakeCopy(CopyL12L0A{}, CopyL12L0ATraitDefault{});
    auto copyL12L0BAtom = MakeCopy(CopyL12L0B{}, CopyL12L0BTraitDefault{});
    auto copyL0C2GMAtom = MakeCopy(CopyL0C2GM{}, CopyL0C2GMTraitDefault{});
    auto mMadAtom = MakeMmad(MmadOperation{}, MmadTraitDefault{});

    auto l1ATensor = MakeTensor(MakeMemPtr(l1ABuf), MakeFrameLayout<NZLayoutPtn, half>(baseM, stepK * baseK));
    auto l1BTensor = MakeTensor(MakeMemPtr(l1BBuf), MakeFrameLayout<ZNLayoutPtn, half>(stepK * baseK, baseN));
    auto l0ATensor = MakeTensor(MakeMemPtr(l0ABuf), MakeFrameLayout<NZLayoutPtn, half>(baseM, baseK));
    auto l0BTensor = MakeTensor(MakeMemPtr(l0BBuf), MakeFrameLayout<ZNLayoutPtn, half>(baseK, baseN));
    auto l0CTensor = MakeTensor(MakeMemPtr(l0CBuf), MakeFrameLayout<NZLayoutPtn>(baseM, baseN));

    constexpr uint32_t L1_EVENT_ID = 0;
    constexpr uint32_t L0_EVENT_ID = 1;
    constexpr uint32_t L0C_EVENT_ID = 2;
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);

    for (uint32_t mi = 0; mi < mLoop; ++mi) {
        for (uint32_t ni = 0; ni < nLoop; ++ni) {
            for (uint32_t ki = 0; ki < kLoop; ++ki) {
                if (ki % stepK == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
                    Copy(copyGM2L1Atom, l1ATensor,
                        gmASingle.Slice(MakeCoord(mi * baseM, ki * baseK), MakeShape(baseM, stepK * baseK)));
                    Copy(copyGM2L1Atom, l1BTensor,
                        gmBSingle.Slice(MakeCoord(ki * baseK, ni * baseN), MakeShape(stepK * baseK, baseN)));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_EVENT_ID);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_EVENT_ID);
                }

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
                Copy(copyL12L0AAtom, l0ATensor,
                    l1ATensor.Slice(MakeCoord(0, (ki % stepK) * baseK), MakeShape(baseM, baseK)));
                Copy(copyL12L0BAtom, l0BTensor,
                    l1BTensor.Slice(MakeCoord((ki % stepK) * baseK, 0), MakeShape(baseK, baseN)));
                if ((ki + 1) % stepK == 0 || ki + 1 == kLoop) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_EVENT_ID);

                if (ki == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
                }
                MmadParams params{baseM, baseN, baseK, 0, (ki == 0)};
                Mmad(mMadAtom.with(params), l0CTensor, l0ATensor, l0BTensor);
                if (ki + 1 == kLoop) {
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_EVENT_ID);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
            }

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_EVENT_ID);
            Copy(copyL0C2GMAtom, gmCSingle.Slice(MakeCoord(mi * baseM, ni * baseN), MakeShape(baseM, baseN)), l0CTensor);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
}

#endif
