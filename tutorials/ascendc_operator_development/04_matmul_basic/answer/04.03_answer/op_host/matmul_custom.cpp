
// 包含自定义Tiling数据结构头文件，定义了MatmulCustomTilingData
#include "matmul_custom_tiling.h"
// 昇腾算子注册接口，用于定义算子属性、输入输出等
#include "register/op_def_registry.h"
// 昇腾平台信息获取接口，用于查询芯片版本、内存大小等
#include "tiling/platform/platform_ascendc.h"
// 矩阵乘Tiling核心API，包含MultiCoreMatmulTiling等
#include "tiling/tiling_api.h"
// 使用matmul_tiling命名空间，简化类型名
using namespace matmul_tiling;

namespace optiling {

// 昇腾图形编译器在构图阶段调用此函数，动态生成矩阵乘法所需的切分参数。
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // -------------------- 步骤1：创建多核Tiling对象 --------------------
    // 1.1 获取当前运行环境的平台信息（芯片型号等）
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    
    // 1.2 设置矩阵乘法输入shape形状
    int32_t M = 1024;
    int32_t N = 2048;      // B的列数
    int32_t K = 512;      // A的列数 / B的行数
    // M = 1024, N = 2048, K = 512
    // 1.3 指定基础分块大小（Cube单次指令执行的计算量），这里固定为128x128
    int32_t baseM = 128;
    int32_t baseN = 128;
    
    // 1.4 实例化多核Tiling对象，传入平台信息
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    
    // -------------------- 步骤2：配置矩阵属性与切分策略 --------------------
    // 2.1 设置参与多核并行计算的AI Core数量（必须与后续SetBlockDim一致）
    cubeTiling.SetDim(2);   // 使用2个核并行计算
    
    // 2.2 配置A矩阵：存储位置GM、数据格式ND、数据类型FP16
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    // 2.3 配置B矩阵：同样为GM、ND、FP16
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    // 2.4 配置C矩阵：输出为GM、ND、FP32
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // 2.5 配置Bias：GM、ND、FP32
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    
    // 2.6 设置矩阵实际形状（用于计算）
    cubeTiling.SetShape(M, N, K);
    // 2.7 设置原始形状（用于尾块处理，此处与实际形状相同）
    cubeTiling.SetOrgShape(M, N, K);
    
    // 2.8 固定基础分块尺寸：baseM=128，baseN=128，baseK=-1表示由库自动选择
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    
    // 2.9 启用Bias计算
    cubeTiling.SetBias(true);
    
    // 2.10 设置L1缓冲区分配比例，-1表示由Tiling库自动分配
    cubeTiling.SetBufferSpace(-1, -1, -1);
    
    // -------------------- 步骤3：生成Tiling参数 --------------------
    // 3.1 实例化自定义Tiling数据结构，用于接收生成的参数
    MatmulCustomTilingData tiling;
    
    // 3.2 调用GetTiling接口，将计算好的切分参数填充到tiling.cubeTilingData中
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;   // Tiling生成失败
    }
    
    // 3.3 查询当前平台每个AI Core的Unified Buffer（UB）大小（字节）
    uint64_t localMemSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    tiling.set_localMemSize(localMemSize);   // 存入Tiling数据，供核函数分配工作空间
    
    // -------------------- 平台差异化配置 --------------------
    // 根据芯片版本设置BlockDim和TilingKey，核函数根据TilingKey选择执行路径
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        // Ascend310P平台：使用2个Cube核，TilingKey=2（对应Process<true>分支）
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        /* 
         * 对于分离架构芯片（如Ascend910B），AIC:AIV=1:2，即1个Cube核心对应2个Vector核心。
         * SetDim(2)设置的是矢量核数量，而SetBlockDim需要设置为Cube核数量（此处为1）。
         * 此时TilingKey=1（对应Process<false>分支），由核函数根据平台自行管理UB空间。
         */
        context->SetBlockDim(1);
        context->SetTilingKey(1);
    }
    
    // -------------------- 将Tiling数据写入上下文缓冲区 --------------------
    // 序列化Tiling对象，存入context提供的缓冲区，供核函数通过GET_TILING_DATA宏解析
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // -------------------- 设置工作空间大小 --------------------
    // 用户自定义工作空间（本算子无需，设为0）
    size_t userWorkspaceSize = 0;
    // 系统工作空间：Matmul高阶API内部需要的临时内存，通过平台接口获取
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // 向框架申请一块工作空间（大小为1的数组），用于存放总工作空间大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);            
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);   // 设置Tiling函数
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulCustom);

} // namespace ops
