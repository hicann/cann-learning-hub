// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "../op_kernel/as_strided_tiling.h"

#include "../op_kernel/tiling_key_as_strided.h"

namespace optiling {
    static inline uint32_t AlignUp(uint32_t value, uint32_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        // 2. 获取算子输入信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int dtype_size = ge::GetSizeByDataType(dtype_input_x);  // 字长: 2/4/4
        uint32_t inputTotalElements = tensor_input_x->GetShapeSize();

        // 3. 获取属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const gert::TypedContinuousVector<int64_t> *attr_size = attrs->GetListInt(0);
        const gert::TypedContinuousVector<int64_t> *attr_stride = attrs->GetListInt(1);
        const int64_t *attr_storage_offset = attrs->GetInt(2);

        uint32_t ndim = static_cast<uint32_t>(attr_size->GetSize());
        int32_t storageOffset = static_cast<int32_t>(*attr_storage_offset);

        // 4. 计算输出总元素数和 dimStride
        uint32_t totalOutputElements = 1;
        uint32_t sizeArr[4] = {1, 1, 1, 1};
        int32_t strideArr[4] = {0, 0, 0, 0};
        uint32_t dimStrideArr[4] = {1, 1, 1, 1};

        const int64_t *sizeData = attr_size->GetData();
        const int64_t *strideData = attr_stride->GetData();
        for (uint32_t d = 0; d < ndim; d++) {
            sizeArr[d] = static_cast<uint32_t>(sizeData[d]);
            strideArr[d] = static_cast<int32_t>(strideData[d]);
            totalOutputElements *= sizeArr[d];
        }

        // dimStride[d] = size[d+1] * size[d+2] * ... * size[ndim-1]
        dimStrideArr[ndim - 1] = 1;
        for (int d = static_cast<int>(ndim) - 2; d >= 0; d--) {
            dimStrideArr[d] = dimStrideArr[d + 1] * sizeArr[d + 1];
        }

        // 5. 路径判断和 tileSize 计算
        constexpr uint32_t UB_SIZE = 196608;  // 192KB for ascend910b (DAV_2201)
        constexpr uint32_t RESERVED = 8192;   // 预留 8KB
        uint32_t ubBudget = UB_SIZE - RESERVED;  // 188416

        uint32_t inputTotalBytes = inputTotalElements * static_cast<uint32_t>(dtype_size);
        uint32_t inputBytesAligned = AlignUp(inputTotalBytes, 32);

        uint32_t tileSize = 0;
        uint32_t pathFlag = 0;  // 0=PathA, 1=PathB

        // Path A: 输入全量在 UB
        if (inputBytesAligned < ubBudget) {
            uint32_t remainBudget = ubBudget - inputBytesAligned;
            // offset 表: tileSize * 4, 输出: tileSize * dtypeSize * 2 (Double Buffer)
            uint32_t perTileBytes = 4 + static_cast<uint32_t>(dtype_size) * 2;
            tileSize = (remainBudget / perTileBytes) & ~0x7u;  // 向下对齐到 8 元素
            tileSize = std::max(tileSize, 8u);

            // 验证 Path A 可行
            uint32_t offsetTileBytes = AlignUp(tileSize * 4, 32);
            uint32_t outputTileBytes = AlignUp(tileSize * static_cast<uint32_t>(dtype_size), 32);
            if (inputBytesAligned + offsetTileBytes + outputTileBytes * 2 <= ubBudget) {
                pathFlag = 0;  // Path A
            } else {
                pathFlag = 1;  // Path B
            }
        } else {
            pathFlag = 1;  // Path B
        }

        if (pathFlag == 1) {
            // Path B: 无输入全量，tileSize 受 srcIdx 表 + 输出 tile 约束
            uint32_t remainBudget = ubBudget - 32;  // 减去单元素临时 buffer
            uint32_t perTileBytes = 4 + static_cast<uint32_t>(dtype_size);
            tileSize = (remainBudget / perTileBytes) & ~0x7u;
            tileSize = std::max(tileSize, 8u);
        }

        // 6. 多核切分
        uint32_t blockDim = std::min(static_cast<uint32_t>(num_cores_aiv), totalOutputElements);
        if (blockDim == 0) blockDim = 1;

        // 7. 预计算偏移表（Host侧一次性计算，消除 Kernel 内除法/取模和 SetValue）
        std::vector<uint32_t> offsetTable(totalOutputElements);
        for (uint32_t f = 0; f < totalOutputElements; f++) {
            // 从平坦索引反推多维索引
            int32_t srcIdx = storageOffset;
            uint32_t remaining = f;
            for (uint32_t d = 0; d < ndim; d++) {
                uint32_t i_d = remaining / dimStrideArr[d];
                remaining = remaining % dimStrideArr[d];
                srcIdx += static_cast<int32_t>(i_d) * strideArr[d];
            }
            // 防御性 clamp：防止负 stride 或越界参数导致 Gather 偏移越界
            if (srcIdx < 0 || static_cast<uint32_t>(srcIdx) >= inputTotalElements) {
                srcIdx = 0;
            }
            if (pathFlag == 0) {
                // Path A: byte offset for Gather
                offsetTable[f] = static_cast<uint32_t>(srcIdx) * static_cast<uint32_t>(dtype_size);
            } else {
                // Path B: element index for DataCopyPad
                offsetTable[f] = static_cast<uint32_t>(srcIdx);
            }
        }

        // 8. 填充 TilingData 并尝试将偏移表追加到 tiling data
        AsStridedTilingData *tiling = context->GetTilingData<AsStridedTilingData>();
        tiling->totalOutputElements = totalOutputElements;
        tiling->tileSize = tileSize;
        tiling->inputTotalElements = inputTotalElements;
        tiling->ndim = ndim;
        for (uint32_t d = 0; d < 4; d++) {
            tiling->size[d] = sizeArr[d];
            tiling->stride[d] = strideArr[d];
            tiling->dimStride[d] = dimStrideArr[d];
        }
        tiling->storageOffset = storageOffset;
        tiling->pathFlag = pathFlag;
        tiling->blockDim = blockDim;

        // 尝试将偏移表追加到 tiling data（避免 Kernel 内 SetValue 循环）
        uint32_t offsetTableInTiling = 0;
        auto rawTiling = context->GetRawTilingData();
        if (rawTiling != nullptr) {
            size_t tableBytes = totalOutputElements * sizeof(uint32_t);
            size_t currentSize = rawTiling->GetDataSize();
            size_t capacity = rawTiling->GetCapacity();
            if (currentSize + tableBytes <= capacity) {
                // 容量足够，追加偏移表到 tiling data
                rawTiling->Append(offsetTable.data(), totalOutputElements);
                offsetTableInTiling = 1;
            }
        }
        tiling->offsetTableInTiling = offsetTableInTiling;

        // 9. 配置 tiling key
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 10. 配置启动核数
        context->SetBlockDim(blockDim);

        // 11. 配置 workspace 大小（偏移表已存入 tiling data，无需 workspace）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const gert::TypedContinuousVector<int64_t> *attr_size = attrs->GetListInt(0);

        gert::Shape *output_shape = context->GetOutputShape(0);
        const int64_t *sizeData = attr_size->GetData();
        for (size_t i = 0; i < attr_size->GetSize(); i++) {
            output_shape->AppendDim(sizeData[i]);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class AsStrided : public OpDef {
    public:
        explicit AsStrided(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("input_size").AttrType(REQUIRED).ListInt();
            this->Attr("input_stride").AttrType(REQUIRED).ListInt();
            this->Attr("input_storage_offset").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(AsStrided);
}  // namespace ops
