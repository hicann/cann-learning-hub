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
 * \file tile_mmad_mx.h
 * \brief Tile-level MMAD traits used by the SWAT MX kernels.
 */

#pragma once

#include "include/tensor_api/tensor.h"

namespace asc {
namespace te {

constexpr mmad_trait MX_MMAD_TRAIT = mmad_trait{0, false, false, true, mmad_type::mx};
struct MmadTraitMX {
    using TraitType = mmad_trait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};

template <>
struct mmad_traits<mmad_operation, MmadTraitMX>
    : public mmad_traits<mmad_operation, mmad_trait_default, mmad_op_with, MmadTraitMX> {};

} // namespace te
} // namespace asc
