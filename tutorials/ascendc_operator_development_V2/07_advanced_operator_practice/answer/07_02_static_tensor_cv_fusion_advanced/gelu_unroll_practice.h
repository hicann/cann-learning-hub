/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_07_02_GELU_UNROLL_PRACTICE_H
#define ASCENDC_07_02_GELU_UNROLL_PRACTICE_H

// 双路展开使用两组 RegTensor 交错执行 GELU 算术链。
__simd_vf__ inline void GeluVf(__ubuf__ float* xAddr, __ubuf__ float* yAddr, uint32_t n)
{
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    uint32_t loopNum = DivCeil(n, oneRepeatSize);
    AscendC::Reg::MaskReg mask0, mask1;
    AscendC::Reg::RegTensor<float> xReg0, yReg0, xReg1, yReg1;

    uint16_t i = 0;
    for (; i + 1 < loopNum; i += 2) {
        mask0 = AscendC::Reg::UpdateMask<float>(n);
        AscendC::Reg::LoadAlign(xReg0, xAddr + i * oneRepeatSize);
        mask1 = AscendC::Reg::UpdateMask<float>(n);
        AscendC::Reg::LoadAlign(xReg1, xAddr + (i + 1) * oneRepeatSize);

        AscendC::Reg::Mul(yReg0, xReg0, xReg0, mask0);
        AscendC::Reg::Mul(yReg1, xReg1, xReg1, mask1);
        AscendC::Reg::Mul(yReg0, yReg0, xReg0, mask0);
        AscendC::Reg::Mul(yReg1, yReg1, xReg1, mask1);
        AscendC::Reg::Muls(yReg0, yReg0, GELU_COEFF_A, mask0);
        AscendC::Reg::Muls(yReg1, yReg1, GELU_COEFF_A, mask1);
        AscendC::Reg::Add(yReg0, xReg0, yReg0, mask0);
        AscendC::Reg::Add(yReg1, xReg1, yReg1, mask1);
        AscendC::Reg::Muls(yReg0, yReg0, GELU_COEFF_B, mask0);
        AscendC::Reg::Muls(yReg1, yReg1, GELU_COEFF_B, mask1);
        AscendC::Reg::Exp(yReg0, yReg0, mask0);
        AscendC::Reg::Exp(yReg1, yReg1, mask1);
        AscendC::Reg::Adds(yReg0, yReg0, 1.0f, mask0);
        AscendC::Reg::Adds(yReg1, yReg1, 1.0f, mask1);
        AscendC::Reg::Div(yReg0, xReg0, yReg0, mask0);
        AscendC::Reg::Div(yReg1, xReg1, yReg1, mask1);

        AscendC::Reg::StoreAlign(yAddr + i * oneRepeatSize, yReg0, mask0);
        AscendC::Reg::StoreAlign(yAddr + (i + 1) * oneRepeatSize, yReg1, mask1);
    }

    if (i < loopNum) {
        mask0 = AscendC::Reg::UpdateMask<float>(n);
        AscendC::Reg::LoadAlign(xReg0, xAddr + i * oneRepeatSize);
        AscendC::Reg::Mul(yReg0, xReg0, xReg0, mask0);
        AscendC::Reg::Mul(yReg0, yReg0, xReg0, mask0);
        AscendC::Reg::Muls(yReg0, yReg0, GELU_COEFF_A, mask0);
        AscendC::Reg::Add(yReg0, xReg0, yReg0, mask0);
        AscendC::Reg::Muls(yReg0, yReg0, GELU_COEFF_B, mask0);
        AscendC::Reg::Exp(yReg0, yReg0, mask0);
        AscendC::Reg::Adds(yReg0, yReg0, 1.0f, mask0);
        AscendC::Reg::Div(yReg0, xReg0, yReg0, mask0);
        AscendC::Reg::StoreAlign(yAddr + i * oneRepeatSize, yReg0, mask0);
    }
}

#endif // ASCENDC_07_02_GELU_UNROLL_PRACTICE_H
