/*
 * 1989 Free Software Foundation Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include "../op_kernel/qmm_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cstring>

using namespace matmul_tiling;

namespace optiling {
constexpr size_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);

    int32_t M = x1_shape->GetStorageShape().GetDim(0);
    int32_t K = x1_shape->GetStorageShape().GetDim(1);
    int32_t N = x2_shape->GetStorageShape().GetDim(1);

    auto outputDesc = context->GetOutputDesc(0);
    if (outputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    bool hasPertoken = (outputDesc->GetDataType() == ge::DT_BF16);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    MatmulApiTiling mmTiling(platform);
    mmTiling.SetOrgShape(M, N, K);
    mmTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8);
    mmTiling.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT8);
    mmTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    mmTiling.SetBias(false);

    AscendC::tiling::TCubeTiling cubeTiling;
    if (mmTiling.GetTiling(cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }

    QmmCustomTilingData tiling;
    memset(&tiling, 0, sizeof(tiling));
    tiling.cubeTilingData = cubeTiling;

    tiling.isPertoken = hasPertoken ? 1 : 0;
    tiling.M = M;
    tiling.N = N;
    tiling.K = K;
    tiling.singleCoreM = cubeTiling.singleCoreM;
    tiling.singleCoreN = cubeTiling.singleCoreN;
    tiling.workspaceSize = 0;

    auto* rawData = context->GetRawTilingData();
    auto* buffer = rawData->GetData();
    auto capacity = rawData->GetCapacity();
    auto dataSize = sizeof(QmmCustomTilingData);
    if (dataSize <= capacity) {
        memcpy(buffer, &tiling, dataSize);
    }
    rawData->SetDataSize(dataSize);
    context->SetBlockDim(tiling.cubeTilingData.usedCoreNum);

    size_t userWorkspaceSize = static_cast<size_t>(tiling.workspaceSize);
    size_t systemWorkspaceSize =
        std::max(static_cast<size_t>(platform.GetLibApiWorkSpaceSize()), RESERVED_WORKSPACE_SIZE);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);

    int64_t M = x1_shape->GetDim(0);
    int64_t N = x2_shape->GetDim(1);

    y_shape->SetDimNum(2);
    y_shape->SetDim(0, M);
    y_shape->SetDim(1, N);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto pertokenDataType = context->GetInputDataType(4);
    if (pertokenDataType != ge::DT_UNDEFINED) {
        context->SetOutputDataType(0, ge::DT_BF16);
    } else {
        context->SetOutputDataType(0, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class QmmCustom : public OpDef {
public:
    explicit QmmCustom(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("tmp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("pertoken_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(QmmCustom);
}
