#ifndef MATMUL_CUSTOM_TILING_H
#define MATMUL_CUSTOM_TILING_H

#include <cstdint>
#include <iostream>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

inline bool GenMatmulTiling(platform_ascendc::PlatformAscendC* ascendcPlatform, TCubeTiling &tiling, int32_t m,
    int32_t n, int32_t k, int32_t blockDim)
{
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
    cubeTiling.SetDim(blockDim);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(m, n, k);
    cubeTiling.SetOrgShape(m, n, k);
    cubeTiling.SetFixSplit(128, 128, -1);
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    if (cubeTiling.GetTiling(tiling) == -1) {
        std::cerr << "GenMatmulTiling failed." << std::endl;
        return false;
    }
    return true;
}

#endif
