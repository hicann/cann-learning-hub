#ifndef FAST_GELU_TILING_ARCH35_H_
#define FAST_GELU_TILING_ARCH35_H_

#include "register/tilingdata_base.h"

namespace ge {

struct FastGeluTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

}  // namespace ge

#endif  // FAST_GELU_TILING_ARCH35_H_
