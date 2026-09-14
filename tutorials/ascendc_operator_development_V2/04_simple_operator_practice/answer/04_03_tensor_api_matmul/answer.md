# 04.03 Tensor API 矩阵算子实践参考答案

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
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, L1_EVENT_ID);
    copy(copyGM2L1Atom, l1ATensor,
         gmASingle.slice(make_coord(mi * baseM, ki * baseK), make_shape(baseM, stepK * baseK)));
    copy(copyGM2L1Atom, l1BTensor,
         gmBSingle.slice(make_coord(ki * baseK, ni * baseN), make_shape(stepK * baseK, baseN)));
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, L1_EVENT_ID);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, L1_EVENT_ID);
}
```

每 `stepK` 次 K 循环执行一次 GM→L1 搬运，一次搬入 `stepK` 个基本块。GM 切片的 K 方向大小为 `stepK * baseK`，起始偏移为 `ki * baseK`。L1 缓冲区会被后续 `stepK` 次 L1→L0 搬运复用，因此只在发起新一轮 GM→L1 前等待上一轮 L1 数据消费完成。

### TODO(3)：L1→L0 切片搬运

```cpp
copy(copyL12L0AAtom, l0ATensor,
     l1ATensor.slice(make_coord(0, (ki % stepK) * baseK), make_shape(baseM, baseK)));
copy(copyL12L0BAtom, l0BTensor,
     l1BTensor.slice(make_coord((ki % stepK) * baseK, 0), make_shape(baseK, baseN)));
if ((ki + 1) % stepK == 0 || ki + 1 == kLoop) {
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, L1_EVENT_ID);
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
    using namespace asc::te;
    static_assert(M % singleM == 0 && N % singleN == 0 && K % singleK == 0);
    static_assert(singleM % baseM == 0 && singleN % baseN == 0 && singleK % baseK == 0);
    static_assert(singleK % (baseK * stepK) == 0);

    constexpr uint32_t mCoreLoop = M / singleM;
    constexpr uint32_t nCoreLoop = N / singleN;
    constexpr uint32_t mLoop = singleM / baseM;
    constexpr uint32_t nLoop = singleN / baseN;
    constexpr uint32_t kLoop = singleK / baseK;

    uint32_t blockIdx = block_idx;
    uint32_t mCoreIdx = blockIdx % mCoreLoop;
    uint32_t nCoreIdx = blockIdx / mCoreLoop;

    auto gmATensor = make_tensor(make_mem_ptr(x), make_frame_layout<nd_layout_ptn>(M, K));
    auto gmBTensor = make_tensor(make_mem_ptr(y), make_frame_layout<dn_layout_ptn>(K, N));
    auto gmCTensor = make_tensor(make_mem_ptr(z), make_frame_layout<nd_layout_ptn>(M, N));

    auto gmASingle = gmATensor.slice(make_coord(mCoreIdx * singleM, 0), make_shape(singleM, singleK));
    auto gmBSingle = gmBTensor.slice(make_coord(0, nCoreIdx * singleN), make_shape(singleK, singleN));
    auto gmCSingle = gmCTensor.slice(make_coord(mCoreIdx * singleM, nCoreIdx * singleN), make_shape(singleM, singleN));

    __cbuf__ half l1ABuf[stepK * baseM * baseK];
    __cbuf__ half l1BBuf[stepK * baseK * baseN];
    __ca__ half l0ABuf[baseM * baseK];
    __cb__ half l0BBuf[baseK * baseN];
    __cc__ float l0CBuf[baseM * baseN];

    auto copyGM2L1Atom = make_copy(copy_gm_to_l1{}, gm_to_l1_trait_default{});
    auto copyL12L0AAtom = make_copy(copy_l1_to_l0a{}, l1_to_l0a_trait_default{});
    auto copyL12L0BAtom = make_copy(copy_l1_to_l0b{}, l1_to_l0b_trait_default{});
    auto copyL0C2GMAtom = make_copy(copy_l0c_to_gm{}, l0c_to_gm_trait_default{});
    auto mmadAtom = make_mmad(mmad_operation{}, mmad_trait_default{});

    auto l1ATensor = make_tensor(make_mem_ptr(l1ABuf), make_frame_layout<nz_layout_ptn, half>(baseM, stepK * baseK));
    auto l1BTensor = make_tensor(make_mem_ptr(l1BBuf), make_frame_layout<zn_layout_ptn, half>(stepK * baseK, baseN));
    auto l0ATensor = make_tensor(make_mem_ptr(l0ABuf), make_frame_layout<nz_layout_ptn, half>(baseM, baseK));
    auto l0BTensor = make_tensor(make_mem_ptr(l0BBuf), make_frame_layout<zn_layout_ptn, half>(baseK, baseN));
    auto l0CTensor = make_tensor(make_mem_ptr(l0CBuf), make_frame_layout<nz_layout_ptn>(baseM, baseN));

    constexpr event_t L1_EVENT_ID = EVENT_ID0;
    constexpr event_t L0_EVENT_ID = EVENT_ID1;
    constexpr event_t L0C_EVENT_ID = EVENT_ID2;
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, L1_EVENT_ID);
    asc_sync_notify(PIPE_M, PIPE_MTE1, L0_EVENT_ID);
    asc_sync_notify(PIPE_FIX, PIPE_M, L0C_EVENT_ID);

    for (uint32_t mi = 0; mi < mLoop; ++mi) {
        for (uint32_t ni = 0; ni < nLoop; ++ni) {
            for (uint32_t ki = 0; ki < kLoop; ++ki) {
                if (ki % stepK == 0) {
                    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, L1_EVENT_ID);
                    copy(copyGM2L1Atom, l1ATensor,
                         gmASingle.slice(make_coord(mi * baseM, ki * baseK), make_shape(baseM, stepK * baseK)));
                    copy(copyGM2L1Atom, l1BTensor,
                         gmBSingle.slice(make_coord(ki * baseK, ni * baseN), make_shape(stepK * baseK, baseN)));
                    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, L1_EVENT_ID);
                    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, L1_EVENT_ID);
                }

                asc_sync_wait(PIPE_M, PIPE_MTE1, L0_EVENT_ID);

                copy(copyL12L0AAtom, l0ATensor,
                     l1ATensor.slice(make_coord(0, (ki % stepK) * baseK), make_shape(baseM, baseK)));
                copy(copyL12L0BAtom, l0BTensor,
                     l1BTensor.slice(make_coord((ki % stepK) * baseK, 0), make_shape(baseK, baseN)));

                if ((ki + 1) % stepK == 0 || ki + 1 == kLoop) {
                    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, L1_EVENT_ID);
                }
                asc_sync_notify(PIPE_MTE1, PIPE_M, L0_EVENT_ID);
                asc_sync_wait(PIPE_MTE1, PIPE_M, L0_EVENT_ID);

                if (ki == 0) {
                    asc_sync_wait(PIPE_FIX, PIPE_M, L0C_EVENT_ID);
                }
                mmad_params params{baseM, baseN, baseK, unit_flag_mode::disable, (ki == 0)};
                mmad(mmadAtom.with(params), l0CTensor, l0ATensor, l0BTensor);
                if (ki + 1 == kLoop) {
                    asc_sync_notify(PIPE_M, PIPE_FIX, L0C_EVENT_ID);
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, L0_EVENT_ID);
            }

            asc_sync_wait(PIPE_M, PIPE_FIX, L0C_EVENT_ID);
            copy(copyL0C2GMAtom, gmCSingle.slice(make_coord(mi * baseM, ni * baseN), make_shape(baseM, baseN)), l0CTensor);
            asc_sync_notify(PIPE_FIX, PIPE_M, L0C_EVENT_ID);
        }
    }

    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, L1_EVENT_ID);
    asc_sync_wait(PIPE_M, PIPE_MTE1, L0_EVENT_ID);
    asc_sync_wait(PIPE_FIX, PIPE_M, L0C_EVENT_ID);
}

#endif
```

运行实践样例：

```bash
bash run.sh --npu-arch=dav-3510 --case=practice
```

`tensor_mmad_practice.asc` 默认引用 `tensor_mmad_stepK_answer_kernel.h`，当 `verify_result.py` 输出校验通过时，说明实践版本结果正确。stepK 大包搬运将 GM→L1 搬运指令数从 `kLoop`（128 次）减少到 `kLoop / stepK`（32 次），有助于降低 GM 访问开销。若需要统计性能数据，可在相同输入和相同设备状态下使用 profiling 工具采集 `tensor_mmad_practice` 的 kernel 耗时，并与 `single_core`、`multi_core`、`double_buffer` 三个版本的耗时进行对比。
