#ifndef UTILS_H
#define UTILS_H

#include "tiling_data.h"
#include "kernel_operator.h"

namespace MatmulAllGatherImpl {

using namespace AscendC;

constexpr static int64_t L0A_SIZE = 64 * 1024;
constexpr static int64_t L0B_SIZE = 64 * 1024;
constexpr static int64_t L0C_SIZE = 256 * 1024;
constexpr static int64_t L1_SIZE = 512 * 1024;

constexpr static uint32_t CUBE_BASE_M = 16;
constexpr static uint32_t CUBE_BASE_K = 16;
constexpr static uint32_t CUBE_BASE_N = 16;

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t FloorAlign(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a / b) * b;
}

__aicore__ inline uint64_t Min(uint64_t a, uint64_t b)
{
    return (a < b) ? a : b;
}

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID<event>();
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

}

#endif