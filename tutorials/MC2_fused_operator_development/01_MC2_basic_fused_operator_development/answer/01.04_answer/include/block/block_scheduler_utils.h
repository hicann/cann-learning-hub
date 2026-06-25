#ifndef BLOCK_SCHEDULER_UTILS_H
#define BLOCK_SCHEDULER_UTILS_H

#include "kernel_operator.h"

namespace MatmulAllGatherImpl {

constexpr int64_t MNK_M = 0;
constexpr int64_t MNK_N = 1;
constexpr int64_t MNK_K = 2;
constexpr int64_t MNK_M_SPLIT = 3;
constexpr int64_t MNK_N_SPLIT = 4;

constexpr int64_t WINDOW_LEN = 4LL;
constexpr int64_t INNER_AXIS_MIN_SPLIT_VAL = 128LL;

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline T Align(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <size_t I, typename Tp>
__aicore__ constexpr inline decltype(auto) Get(Tp&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<Tp>(t));
}

}

#endif