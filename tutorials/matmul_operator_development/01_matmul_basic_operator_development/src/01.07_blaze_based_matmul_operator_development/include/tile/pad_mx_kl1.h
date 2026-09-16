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
 * \file pad_mx_kl1.h
 * \brief Zero-pad A/B L1 buffers along K when GM slices are shorter than the L1-aligned layout.
 */
#pragma once

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
using asc::te::c0_element;

namespace Tile {
struct PadMxKL1Base {
    template <typename T>
    __aicore__ inline static void PadZero(const T& tensorL1, uint64_t repeatTimes, uint64_t blockNum, uint64_t dstGap)
    {
        create_cbuf_matrix((__cbuf__ half*)tensorL1.data().get(), (blockNum << 16) | (dstGap << 32) | repeatTimes, 0);
    }

    template <typename T>
    __aicore__ inline static constexpr bool IsMxFp4()
    {
        using type = typename T::element_type;
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp4x2_e1m2_t, __cbuf__ fp4x2_e2m1_t>;
    }

    template <typename T>
    __aicore__ inline static constexpr bool IsMxFp8()
    {
        using type = typename T::element_type;
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp8_e5m2_t, __cbuf__ fp8_e4m3fn_t>;
    }
};

struct PadMxKAL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        static_assert(IsMxFp4<T>() || IsMxFp8<T>(), "Only supports MXFP4/MXFP8 L1 tensors.");
        auto layoutL1 = tensorL1.layout();
        auto layoutGm = tensorGm.layout();
        auto kAxis = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(layoutGm);
        auto kAxisL1Align =
            asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 0>(layoutL1) *
            asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(layoutL1);

        if constexpr (asc::te::is_satisfied_ptn_format_v<U, asc::te::nd_ext_layout_ptn>) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < c0_element<T>) {
                return;
            }
            auto mAlign =
                asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 0>(layoutL1) *
                asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, c0_element<T>);
            auto sliceTensor = tensorL1.slice(
                asc::te::make_coord(0, kAxisND2NZAlign),
                asc::te::make_shape(mAlign, kAxisL1Align - kAxisND2NZAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
        } else if constexpr (asc::te::is_satisfied_ptn_format_v<U, asc::te::dn_ext_layout_ptn>) {
            if (kAxis == kAxisL1Align) {
                return;
            }

            // DN2NZ can only zero-pad the innermost m0 axis. Clear the K-axis
            // tail across each outer m1 slice of the A-side NZ layout.
            auto m1 = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(layoutL1);
            auto m0 = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 0>(layoutL1);
            auto dstRowStride =
                asc::te::get_element<asc::te::attr_info::stride, asc::te::attr_info::row, 1>(layoutL1);
            auto dstGap = (dstRowStride / c0_element<T>)-kAxisL1Align + kAxis;
            auto sliceTensor =
                tensorL1.slice(asc::te::make_coord(0, kAxis), asc::te::make_shape(m1 * m0, kAxisL1Align - kAxis));
            PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, dstGap);
        }
    }
};

struct PadMxKBL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        static_assert(IsMxFp4<T>() || IsMxFp8<T>(), "Only supports MXFP4/MXFP8 L1 tensors.");
        auto layoutL1 = tensorL1.layout();
        auto layoutGm = tensorGm.layout();

        auto kAxis = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(layoutGm);
        auto kAxisL1Align =
            asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 0>(layoutL1) *
            asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(layoutL1);

        if constexpr (asc::te::is_satisfied_ptn_format_v<U, asc::te::nd_ext_layout_ptn>) {
            if (kAxis == kAxisL1Align) {
                return;
            }
            // tail across each outer n1 slice of the B-side NZ layout.
            auto n1 = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(layoutL1);
            auto n0 = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 0>(layoutL1);
            auto sliceTensor =
                tensorL1.slice(asc::te::make_coord(kAxis, 0), asc::te::make_shape(kAxisL1Align - kAxis, n1 * n0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        } else if constexpr (asc::te::is_satisfied_ptn_format_v<U, asc::te::dn_ext_layout_ptn>) {
            if constexpr (IsMxFp4<T>()) {
                return;
            }

            if (kAxisL1Align - kAxis < c0_element<T>) {
                return;
            }

            // For FP8 DN input, clear any full-C0 outer K tail from the
            auto nAlign =
                asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 0>(layoutL1) *
                asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(layoutL1);
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, c0_element<T>);
            auto sliceTensor = tensorL1.slice(
                asc::te::make_coord(kAxisND2NZAlign, 0),
                asc::te::make_shape(kAxisL1Align - kAxisND2NZAlign, nAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, nAlign, 0);
        } else if constexpr (asc::te::is_satisfied_ptn_format_v<U, asc::te::nz_layout_ptn>) {
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, AscendC::BLOCK_CUBE);
            if (kAxisND2NZAlign == kAxisL1Align) {
                return;
            }

            // NZ GM slices already expose blocked K coordinates. Clear the
            // remaining K-axis tail across each outer n1 slice.
            auto n1 = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(layoutL1);
            auto n0 = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 0>(layoutL1);
            auto sliceTensor =
                tensorL1.slice(asc::te::make_coord(kAxis, 0), asc::te::make_shape(kAxisL1Align - kAxis, n1 * n0));
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        }
    }
};
} // namespace Tile
