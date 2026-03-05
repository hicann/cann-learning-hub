#ifndef MATMUL_ABS_TILING_H
#define MATMUL_ABS_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MatmulAbsTilingData {
    AscendC::tiling::TCubeTiling cubeTilingData;
};

#endif
