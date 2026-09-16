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
 * \file copy_scale_l1_to_l0a.h
 * \brief Tile helper that copies MX scaleA data from L1 to L0A.
 */

#pragma once

#include "include/tensor_api/tensor.h"
#include "kernel_utils/common_utils.h"
#include "../utils/constant.h"

namespace Tile {
struct CopyL12L0MxScaleA3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        using srcType = typename U::element_type;
        using dstType = typename T::element_type;
        static_assert(
            AscendC::Std::is_one_of_v<
                AscendC::Std::tuple<dstType, srcType>, AscendC::Std::tuple<__ca__ fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
        // `coord` is expressed in the original M/K element space; the helper
        // converts it to the packed MX scale coordinates expected by the L0A
        // scale layout and issues one hardware MX load.
        // (m1, k/64, m0, 2)
        // shape ((m0, m1), (2, k/64))
        // stride ((2, k/64*m0*2), (1, m0*2))
        // Zz -> Zz
        uint16_t mStartPosition = CeilDiv(AscendC::Std::get<0>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = CeilDiv(AscendC::Std::get<1>(coord), MXFP_DIVISOR_SIZE);
        auto mStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.layout().shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.layout().shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(src.layout().stride())) >> 5;
        auto dstStride = kStep;
        // The intrinsic takes a 16-byte unit address, hence the right shift.
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.data().get())) >> 4;
        asc_copy_l12l0a_mx(
            mxDstAddr, src.data().get(), mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
    }
};

// Expose this helper through TE's generic copy-trait interface.
} // namespace Tile

template <>
struct asc::te::copy_traits<::Tile::CopyL12L0MxScaleA3510>
    : public copy_traits<
          ::Tile::CopyL12L0MxScaleA3510, CopyL12L0ATraitDefault, ::Tile::CopyL12L0MxScaleA3510,
          CopyL12L0ATraitDefault> {};
