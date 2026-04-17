#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"
#include "aclnn_matmul_sinh.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
        }                                                                                               \
    } while (0)

int32_t main(int32_t argc, char** argv)
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclnnInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    // 定义MatmulSinh的张量形状
    int64_t M = 1024;
    int64_t N = 2048;
    int64_t K = 512;
    const std::vector<int64_t> shape_a = {M, K};    // 输入a形状
    const std::vector<int64_t> shape_b = {K, N};     // 输入b形状
    const std::vector<int64_t> shape_bias = {N};       // 输入bias形状
    const std::vector<int64_t> shape_output = {M, N};// 输出形状

    // 计算各张量元素数量和内存大小
    const int64_t elementCount_a = shape_a[0] * shape_a[1];
    const int64_t elementCount_b = shape_b[0] * shape_b[1];
    const int64_t elementCount_bias = shape_bias[0];
    const int64_t elementCount_output = shape_output[0] * shape_output[1];
    
    const size_t bufferSize_a = elementCount_a * sizeof(float);
    const size_t bufferSize_b = elementCount_b * sizeof(float);
    const size_t bufferSize_bias = elementCount_bias * sizeof(float);
    const size_t bufferSize_output = elementCount_output * sizeof(float);

    // 分配输入a的设备内存并创建张量
    void* inputADeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&inputADeviceMem, bufferSize_a, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* inputA = aclCreateTensor(shape_a.data(), shape_a.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                        shape_a.data(), shape_a.size(), inputADeviceMem);

    // 分配输入b的设备内存并创建张量
    void* inputBDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&inputBDeviceMem, bufferSize_b, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* inputB = aclCreateTensor(shape_b.data(), shape_b.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                        shape_b.data(), shape_b.size(), inputBDeviceMem);

    // 分配输入bias的设备内存并创建张量
    void* inputBiasDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&inputBiasDeviceMem, bufferSize_bias, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* inputBias = aclCreateTensor(shape_bias.data(), shape_bias.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                           shape_bias.data(), shape_bias.size(), inputBiasDeviceMem);

    // 分配输出的设备内存并创建张量
    void* outputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&outputDeviceMem, bufferSize_output, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* output = aclCreateTensor(shape_output.data(), shape_output.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                        shape_output.data(), shape_output.size(), outputDeviceMem);

    // 准备主机测试数据
    std::vector<float> inputAHostData(elementCount_a, float(0.125));
    std::vector<float> inputBHostData(elementCount_b, float(0.0625));
    std::vector<float> inputBiasHostData(elementCount_bias, float(0.5));
    std::vector<float> outputHostData(elementCount_output, float(0.0));
    
    // 计算预期结果：Sinh(0.125*0.0625*512 + 0.5) = 45.003011151991785
    std::vector<float> goldenData(elementCount_output, float(45.003011151991785));

    // 主机数据拷贝到设备
    CHECK_ACL(aclrtMemcpy(inputADeviceMem, bufferSize_a, inputAHostData.data(),
                          bufferSize_a, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(inputBDeviceMem, bufferSize_b, inputBHostData.data(),
                          bufferSize_b, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(inputBiasDeviceMem, bufferSize_bias, inputBiasHostData.data(),
                          bufferSize_bias, ACL_MEMCPY_HOST_TO_DEVICE));

    // 获取MatmulSinh算子工作空间大小和执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnMatmulSinhGetWorkspaceSize(inputA, inputB, inputBias, output, &workspaceSize, &executor));

    // 分配工作空间内存
    void* workspaceDeviceMem = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceDeviceMem, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 执行MatmulSinh算子
    CHECK_ACL(aclnnMatmulSinh(workspaceDeviceMem, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 设备结果拷贝回主机
    CHECK_ACL(aclrtMemcpy(outputHostData.data(), bufferSize_output, outputDeviceMem,
                          bufferSize_output, ACL_MEMCPY_DEVICE_TO_HOST));

    // 打印结果并验证
    printf("result is:\n");
    const int64_t previewCount = std::min<int64_t>(elementCount_output, 10);
    for (int64_t i = 0; i < previewCount; i++) { 
        printf("%.1f ", outputHostData[i]); 
    }
    printf("\ntest %s\n", std::equal(outputHostData.begin(), outputHostData.end(), goldenData.begin()) ? "pass" : "failed");

    // 释放资源
    aclDestroyTensor(inputA);
    aclDestroyTensor(inputB);
    aclDestroyTensor(inputBias);
    aclDestroyTensor(output);
    
    CHECK_ACL(aclrtFree(inputADeviceMem));
    CHECK_ACL(aclrtFree(inputBDeviceMem));
    CHECK_ACL(aclrtFree(inputBiasDeviceMem));
    CHECK_ACL(aclrtFree(outputDeviceMem));
    
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtFree(workspaceDeviceMem));
    }
    
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclnnFinalize());
    
    return 0;
}