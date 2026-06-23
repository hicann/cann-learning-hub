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
 * \file moe_distribute_comm.h
 * \brief Common utilities/constants shared by moe_distribute_* headers.
 */

#ifndef MOE_DISTRIBUTE_COMM_H
#define MOE_DISTRIBUTE_COMM_H

#include <cstdint>

namespace AscendC {
constexpr uint32_t NEED_ONE_HUNDRED_AND_TWENTY_SEVEN = 127;
constexpr uint32_t RIGHT_SHIFT_BIT_SEVEN = 7;
constexpr uint32_t NEED_THIRTY_FIRST = 31;
constexpr uint32_t ALIGN_UP_TO_2_MASK = 1;
constexpr uint32_t ALIGN_UP_TO_32_MASK = 31;
constexpr uint32_t ALIGN_UP_TO_64_MASK = 63;
constexpr uint32_t ALIGN_UP_TO_128_MASK = 127;
constexpr uint32_t ALIGN_UP_TO_256_MASK = 255;
constexpr uint32_t ALIGN_UP_TO_512_MASK = 511;
constexpr uint32_t RIGHT_SHIFT_BIT_FIVE = 5;
constexpr uint32_t FIVE_HUNDRED_AND_ELEVEN = 511;
constexpr uint32_t RIGHT_SHIFT_BIT_NINE = 9;

template <typename T1, typename T2>
__aicore__ inline T2 Ceil(T1 x, T1 y)
{
    return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline T Ceil32(T x)
{
    return (x + NEED_THIRTY_FIRST) >> RIGHT_SHIFT_BIT_FIVE;
}

template <typename T>
__aicore__ inline T Ceil128(T x)
{
    return (x + NEED_ONE_HUNDRED_AND_TWENTY_SEVEN) >> RIGHT_SHIFT_BIT_SEVEN;
}

template <typename T>
__aicore__ inline T Ceil512(T x)
{
    return (x + FIVE_HUNDRED_AND_ELEVEN) >> RIGHT_SHIFT_BIT_NINE;
}

template <typename T1, typename T2>
__aicore__ inline T2 Align(T1 x, T1 y)
{
    return Ceil<T1, T2>(x, y) * y;
}

template <typename T>
__aicore__ inline T Align2(T x)
{
    return (x + ALIGN_UP_TO_2_MASK) & (~ALIGN_UP_TO_2_MASK);
}

template <typename T>
__aicore__ inline T Align32(T x)
{
    return (x + ALIGN_UP_TO_32_MASK) & (~ALIGN_UP_TO_32_MASK);
}

template <typename T>
__aicore__ inline T Align64(T x)
{
    return (x + ALIGN_UP_TO_64_MASK) & (~ALIGN_UP_TO_64_MASK);
}

template <typename T>
__aicore__ inline T Align128(T x)
{
    return (x + ALIGN_UP_TO_128_MASK) & (~ALIGN_UP_TO_128_MASK);
}

template <typename T>
__aicore__ inline T Align256(T x)
{
    return (x + ALIGN_UP_TO_256_MASK) & (~ALIGN_UP_TO_256_MASK);
}

template <typename T>
__aicore__ inline T Align512(T x)
{
    return (x + ALIGN_UP_TO_512_MASK) & (~ALIGN_UP_TO_512_MASK);
}
} // namespace AscendC

namespace Mc2Kernel {
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t EXPAND_IDX_INFO = 3U;  // expand_idx triple: rank_id, token_id, topk_id
constexpr uint8_t BUFFER_NUM = 2;         // double-buffering
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t SFFVALUE_SIZE = 64U;
constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;  // UB_ALIGN / sizeof(float) == 8
constexpr uint32_t ZERONE_STATE_POS = 0U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 16U;
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint64_t STATUS_REGION_OFFSET = 1022UL * 1024UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
} // namespace Mc2Kernel

#endif // MOE_DISTRIBUTE_COMM_H
