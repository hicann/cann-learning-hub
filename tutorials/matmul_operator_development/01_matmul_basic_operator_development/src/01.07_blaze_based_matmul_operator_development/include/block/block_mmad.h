/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad.h
 * \brief BlockMmad 基类声明与 MX SWAT 特化入口（教程精简版）。
 */

#pragma once

#include "kernel_utils/common_utils.h"

namespace Block {
template <
    class DispatchPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,
    class Enable = void>
class BlockMmad {
    static_assert(!AscendC::Std::is_same_v<DispatchPolicy_, DispatchPolicy_>, "Should not be here!");
};
} // namespace Block

#include "quant_matmul_mx_block_mmad_swat.h"
