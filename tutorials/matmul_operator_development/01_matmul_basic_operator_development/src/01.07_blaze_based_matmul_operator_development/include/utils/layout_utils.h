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
 * \file layout_utils.h
 * \brief Layout pattern helpers for GEMM-style transpose / weight-NZ traits (device-side).
 */

#pragma once

#include "include/tensor_api/tensor.h"
#include "kernel_utils/common_utils.h"
#include "constant.h"

namespace MatmulRecipe {

template <typename T>
struct Weight4BitNzLayout {
    __aicore__ inline decltype(auto) operator()(int64_t kSize, int64_t nSize)
    {
        // C0 follows the converted compute type; MXFP8FP4 passes FP8 here, so C0 is 32.
        static constexpr int64_t C0 = AscendC::AuxGetC0Size<T>();
        static constexpr int64_t Block = CUBE_BLOCK;
        int64_t k1 = CeilDiv(kSize, C0);
        int64_t n1 = CeilDiv(nSize, Block);

        auto shape = asc::te::make_shape(
            asc::te::make_shape(AscendC::Std::Int<C0>{}, k1),
            asc::te::make_shape(AscendC::Std::Int<Block>{}, n1));
        auto stride = asc::te::make_stride(
            asc::te::make_stride(AscendC::Std::Int<1>{}, n1 * AscendC::Std::Int<Block>{} * AscendC::Std::Int<C0>{}),
            asc::te::make_stride(AscendC::Std::Int<C0>{}, AscendC::Std::Int<Block>{} * AscendC::Std::Int<C0>{}));
        return asc::te::make_layout(shape, stride);
    }
};

/// If \p T is \c asc::te::frame_layout_format<P, Trait>, yields \p P; otherwise yields \p T.
template <typename T>
struct LayoutPatternOf {
    using type = T;
};

template <typename P, typename Trait>
struct LayoutPatternOf<asc::te::frame_layout_format<P, Trait>> {
    using type = P;
};

/// Strips top-level cv on \p T so \c const frame_layout_format<...> unwraps the same as non-const.
template <typename T>
using LayoutPatternOf_t = typename LayoutPatternOf<AscendC::Std::remove_cv_t<T>>::type;

// IsTrans
template <typename Layout>
constexpr bool GetTransValue()
{
    using LayoutPattern = LayoutPatternOf_t<Layout>;
    constexpr bool isNonTrans =
        AscendC::Std::is_one_of_v<LayoutPattern, asc::te::nd_ext_layout_ptn, asc::te::nz_layout_ptn>;
    constexpr bool isTrans =
        AscendC::Std::is_one_of_v<LayoutPattern, asc::te::dn_ext_layout_ptn, asc::te::zn_layout_ptn>;

    constexpr bool isKnown = isNonTrans || isTrans;
    static_assert(isKnown, "IsTrans is not implemented for this layout pattern");

    return !isNonTrans && isTrans;
}

template <typename Layout>
struct IsTrans {
    static constexpr bool value = GetTransValue<Layout>();
};

// IsWeightNz
template <typename Layout>
constexpr bool GetWeightNzValue()
{
    using LayoutPattern = LayoutPatternOf_t<Layout>;
    constexpr bool isNonWeightNz =
        AscendC::Std::is_one_of_v<LayoutPattern, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>;
    constexpr bool isWeightNz =
        AscendC::Std::is_one_of_v<LayoutPattern, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>;

    constexpr bool isKnown = isNonWeightNz || isWeightNz;
    static_assert(isKnown, "IsWeightNz is not implemented for this layout");

    return !isNonWeightNz && isWeightNz;
}

template <typename Layout>
struct IsWeightNz {
    static constexpr bool value = GetWeightNzValue<Layout>();
};

} // namespace MatmulRecipe
