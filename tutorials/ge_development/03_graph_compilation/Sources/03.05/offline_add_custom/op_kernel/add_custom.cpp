/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "add_custom_tiling.h"

namespace {
constexpr int32_t kBufferNum = 1;
}

class KernelAddCustom {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t total_length, uint32_t tile_num) {
    block_length_ = total_length / AscendC::GetBlockNum();
    tile_length_ = block_length_ / tile_num;
    x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + block_length_ * AscendC::GetBlockIdx(),
                          block_length_);
    y_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + block_length_ * AscendC::GetBlockIdx(),
                          block_length_);
    z_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z) + block_length_ * AscendC::GetBlockIdx(),
                          block_length_);
    pipe_.InitBuffer(input_x_, kBufferNum, tile_length_ * sizeof(float));
    pipe_.InitBuffer(input_y_, kBufferNum, tile_length_ * sizeof(float));
    pipe_.InitBuffer(output_z_, kBufferNum, tile_length_ * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t tile_num) {
    for (uint32_t i = 0U; i < tile_num; ++i) {
      auto x_local = input_x_.AllocTensor<float>();
      auto y_local = input_y_.AllocTensor<float>();
      AscendC::DataCopy(x_local, x_gm_[i * tile_length_], tile_length_);
      AscendC::DataCopy(y_local, y_gm_[i * tile_length_], tile_length_);
      input_x_.EnQue(x_local);
      input_y_.EnQue(y_local);

      x_local = input_x_.DeQue<float>();
      y_local = input_y_.DeQue<float>();
      auto z_local = output_z_.AllocTensor<float>();
      AscendC::Add(z_local, x_local, y_local, tile_length_);
      output_z_.EnQue<float>(z_local);
      input_x_.FreeTensor(x_local);
      input_y_.FreeTensor(y_local);

      z_local = output_z_.DeQue<float>();
      AscendC::DataCopy(z_gm_[i * tile_length_], z_local, tile_length_);
      output_z_.FreeTensor(z_local);
    }
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> input_x_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> input_y_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> output_z_;
  AscendC::GlobalTensor<float> x_gm_;
  AscendC::GlobalTensor<float> y_gm_;
  AscendC::GlobalTensor<float> z_gm_;
  uint32_t block_length_ = 0U;
  uint32_t tile_length_ = 0U;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  REGISTER_TILING_DEFAULT(AddCustomTilingData);
  GET_TILING_DATA_WITH_STRUCT(AddCustomTilingData, tiling_data, tiling);
  KernelAddCustom op;
  op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
  op.Process(tiling_data.tileNum);
}
