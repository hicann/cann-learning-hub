# 04.05 Tensor API 矩阵算子实践参考答案

## 课后练习

### 练习 1：多核切分与 Cube Core 数量

多核版本中 `singleM=1024`、`singleN=2048`，核心任务数为：

```text
(M / singleM) * (N / singleN) = (8192 / 1024) * (8192 / 2048) = 8 * 4 = 32
```

该任务数与 Ascend 950PR/DT 的 32 个 Cube Core 对齐，每个 Cube Core 负责一个 `1024 x 2048` 输出 tile。

### 练习 2：stepK 大包搬运参数与 Buffer 占用

在实践配置 `baseM=128`、`baseN=256`、`baseK=64`、`stepK=4` 时，单个 Cube Core 内部循环次数为：

```text
mLoop = singleM / baseM = 1024 / 128 = 8
nLoop = singleN / baseN = 2048 / 256 = 8
kLoop = singleK / baseK = 8192 / 64 = 128
大包搬运次数 = kLoop / stepK = 128 / 4 = 32
```

L1 缓冲区需容纳 stepK 个基本块，占用如下：

```text
L1A 缓冲：stepK * baseM * baseK = 4 * 128 * 64 = 32768 half = 64 KB
L1B 缓冲：stepK * baseK * baseN = 4 * 64 * 256 = 65536 half = 128 KB
L1 输入缓冲合计：64 KB + 128 KB = 192 KB
```

相比不带 stepK 的多核版本（L1A=16KB、L1B=32KB、合计 48KB），L1 占用增加到 4 倍，但 GM→L1 搬运指令数从 128 次减少到 32 次，有利于降低 GM 访问开销。

## 章节实践：stepK 大包搬运 kernel 参考答案

以下先给出 `tensor_mmad_stepK_kernel.h` 中三处 TODO 的参考填法；完整可运行参考实现对应课程工程中的 `tensor_mmad_stepK_answer_kernel.h`。

### TODO(1)：L1 缓冲区声明

```cpp
__cbuf__ half l1ABuf[stepK * baseM * baseK];
__cbuf__ half l1BBuf[stepK * baseK * baseN];
```

L1 缓冲区需要容纳 `stepK` 个基本块，因此大小是单块的 `stepK` 倍。

### TODO(2)：GM→L1 大包搬运

```cpp
if (ki % stepK == 0) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
    Copy(copyGM2L1Atom, l1ATensor,
         gmASingle.Slice(MakeCoord(mi * baseM, ki * baseK), MakeShape(baseM, stepK * baseK)));
    Copy(copyGM2L1Atom, l1BTensor,
         gmBSingle.Slice(MakeCoord(ki * baseK, ni * baseN), MakeShape(stepK * baseK, baseN)));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_EVENT_ID);
}
```

每 `stepK` 次 K 循环执行一次 GM→L1 搬运，一次搬入 `stepK` 个基本块。GM 切片的 K 方向大小为 `stepK * baseK`，起始偏移为 `ki * baseK`。L1 缓冲区会被后续 `stepK` 次 L1→L0 搬运复用，因此只在发起新一轮 GM→L1 前等待上一轮 L1 数据消费完成。

### TODO(3)：L1→L0 切片搬运

```cpp
Copy(copyL12L0AAtom, l0ATensor,
     l1ATensor.Slice(MakeCoord(0, (ki % stepK) * baseK), MakeShape(baseM, baseK)));
Copy(copyL12L0BAtom, l0BTensor,
     l1BTensor.Slice(MakeCoord((ki % stepK) * baseK, 0), MakeShape(baseK, baseN)));
if ((ki + 1) % stepK == 0 || ki + 1 == kLoop) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
}
```

每次 K 循环从 L1 大包中切出当前 `baseK` 块。当前块在 L1 中的 K 方向偏移为 `(ki % stepK) * baseK`，通过 `Slice` 切取后搬入 L0A/L0B。当一个 stepK 大包消费完成，或者到达最后一个 K 分块时，再释放 L1 缓冲区给下一轮 GM→L1 搬运使用。

### 完整参考实现代码

```cpp
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
    auto mmadAtom = MakeMmad(MmadOperation{}, MmadTraitDefault{});

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
                Mmad(mmadAtom.with(params), l0CTensor, l0ATensor, l0BTensor);
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
```

运行实践样例：

```bash
bash run.sh --npu-arch=dav-3510 --case=practice
```

`tensor_mmad_practice.asc` 默认引用 `tensor_mmad_stepK_answer_kernel.h`，当 `verify_result.py` 输出校验通过时，说明实践版本结果正确。stepK 大包搬运将 GM→L1 搬运指令数从 `kLoop`（128 次）减少到 `kLoop / stepK`（32 次），有助于降低 GM 访问开销。若需要统计性能数据，可在相同输入和相同设备状态下使用 profiling 工具采集 `tensor_mmad_practice` 的 kernel 耗时，并与 `single_core`、`multi_core`、`double_buffer` 三个版本的耗时进行对比。
