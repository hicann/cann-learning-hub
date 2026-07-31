/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 1024;                             // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                    // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;          // length computed of each core
constexpr int32_t TILE_NUM = 8;                                        // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                      // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;  // separate to 2 parts, due to double buffer

class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    xGm.SetGlobalBuffer((__gm__ float *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__ float *)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ float *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = TILE_NUM * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
    AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);

    outQueueZ.EnQue<float>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> yGm;
  AscendC::GlobalTensor<float> zGm;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelAdd op;
  op.Init(x, y, z);
  op.Process();
}
