/**
 * @file main.cpp
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * 实验5.1 msSanitizer 标准算子工程版：ACLNN Host 测试入口。
 * 课程适配：以模板 add_custom_template/test/main.cpp 为基础，仅把自定义算子
 * 的 ACLNN 接口名替换为 AddCustomMsanitizer（aclnnAddCustomMsanitizer*）。
 * 本测试只证明算子能被 ACLNN 单算子调用；故障模式是否命中由对应的
 * mssanitizer --tool=<mode> 诊断判定，不依赖本程序的退出码。
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_add_custom_msanitizer.h"

#define SUCCESS 0
#define FAILED 1

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Fixed code, acl initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // Call aclrtMemcpy to copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // Call aclCreateTensor to create a aclTensor object
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    // 检查 aclCreateTensor 是否成功；失败时 *deviceAddr 已由调用方持有，
    // 调用方 cleanup 仍会释放已分配的设备内存。
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed\n");
        return FAILED;
    }
    return SUCCESS;
}

struct TestResources {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *outputZDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *outputZ = nullptr;

    void Cleanup()
    {
        if (inputX != nullptr) aclDestroyTensor(inputX);
        if (inputY != nullptr) aclDestroyTensor(inputY);
        if (outputZ != nullptr) aclDestroyTensor(outputZ);
        if (inputXDeviceAddr != nullptr) aclrtFree(inputXDeviceAddr);
        if (inputYDeviceAddr != nullptr) aclrtFree(inputYDeviceAddr);
        if (outputZDeviceAddr != nullptr) aclrtFree(outputZDeviceAddr);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

int CreateInputs(const std::vector<int64_t> &shape, TestResources *resources)
{
    const std::vector<float> inputXHostData(GetShapeSize(shape), 1.0f);
    const std::vector<float> inputYHostData(GetShapeSize(shape), 2.0f);
    const std::vector<float> outputZHostData(GetShapeSize(shape), 0.0f);
    auto ret = CreateAclTensor(inputXHostData, shape, &resources->inputXDeviceAddr,
                               aclDataType::ACL_FLOAT, &resources->inputX);
    if (ret != ACL_SUCCESS) return FAILED;
    ret = CreateAclTensor(inputYHostData, shape, &resources->inputYDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources->inputY);
    if (ret != ACL_SUCCESS) return FAILED;
    return CreateAclTensor(outputZHostData, shape, &resources->outputZDeviceAddr,
                           aclDataType::ACL_FLOAT, &resources->outputZ);
}

int RunOperator(TestResources *resources)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnAddCustomMsanitizerGetWorkspaceSize(
        resources->inputX, resources->inputY, resources->outputZ,
        &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnAddCustomMsanitizerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return FAILED;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&resources->workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return FAILED;
        }
    }
    ret = aclnnAddCustomMsanitizer(resources->workspaceAddr, workspaceSize,
                                   executor, resources->stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnAddCustomMsanitizer failed. ERROR: %d\n", ret);
        return FAILED;
    }
    ret = aclrtSynchronizeStream(resources->stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return FAILED;
    }
    return SUCCESS;
}

int CopyResult(const std::vector<int64_t> &shape, const TestResources &resources,
               std::vector<float> *result)
{
    const auto size = GetShapeSize(shape);
    result->assign(static_cast<std::size_t>(size), 0.0f);
    const auto ret = aclrtMemcpy(result->data(), result->size() * sizeof((*result)[0]),
                                 resources.outputZDeviceAddr, size * sizeof(float),
                                 ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) return SUCCESS;
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return FAILED;
}

int CheckResult(const std::vector<float> &result)
{
    LOG_PRINT("result is:\n");
    for (int64_t i = 0; i < 10; ++i) LOG_PRINT("%.1f ", result[i]);
    LOG_PRINT("\n");
    const std::vector<float> goldenData(result.size(), 3.0f);
    if (std::equal(result.begin(), result.end(), goldenData.begin())) {
        LOG_PRINT("test pass\n");
        return SUCCESS;
    }
    LOG_PRINT("test failed\n");
    return FAILED;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    TestResources resources;
    auto ret = Init(resources.deviceId, &resources.stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);
    const std::vector<int64_t> shape = {64, 64};
    ret = CreateInputs(shape, &resources);
    if (ret == SUCCESS) ret = RunOperator(&resources);
    std::vector<float> result;
    if (ret == SUCCESS) ret = CopyResult(shape, resources, &result);
    resources.Cleanup();
    return ret == SUCCESS ? CheckResult(result) : FAILED;
}
