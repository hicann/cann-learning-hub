#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sigmoid.h"

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
    return SUCCESS;
}

void DestroyResources(std::vector<void *> tensors, std::vector<void *> deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
{
    // Release aclTensor and device
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));
        }
        if (deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    // Destroy stream and reset device
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    // 1. (Fixed code) Initialize device / stream, refer to the list of external interfaces of acl
    // Update deviceId to your own device id
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. Create input and output, need to customize according to the interface of the API
    std::vector<int64_t> inputXShape = {8, 2048};
    
    std::vector<int64_t> outputYShape = {8, 2048};
    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;
    std::vector<float> inputXHostData(inputXShape[0] * inputXShape[1]);
    std::vector<float> outputYHostData(outputYShape[0] * outputYShape[1]);
    for (int i = 0; i < inputXShape[0] * inputXShape[1]; ++i) {
        inputXHostData[i] = 1.0;
        outputYHostData[i] = 0.0;
    }
    std::vector<void *> tensors = {inputX, outputY};
    std::vector<void *> deviceAddrs = {inputXDeviceAddr, outputYDeviceAddr};
    // Create inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    // Create outputY aclTensor
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    // 3. Call the API of the custom operator library
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Calculate the workspace size and allocate memory for it
    ret = aclnnSigmoidGetWorkspaceSize(inputX, outputY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSigmoidGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }
    // Execute the custom operator
    ret = aclnnSigmoid(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSigmoid failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 4. (Fixed code) Synchronize and wait for the task to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 5. Get the output value, copy the result from device memory to host memory, need to modify according to the
    // interface of the API
    auto size = GetShapeSize(outputYShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputYDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 6. Detroy resources, need to modify according to the interface of the API
    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    // print the output result
    std::vector<float> goldenData(size, 0.7310585786300049);

    LOG_PRINT("result is:\n");
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("%.1f ", resultData[i]);
    }
    LOG_PRINT("\n");
    if (std::equal(resultData.begin(), resultData.end(), goldenData.begin())) {
        LOG_PRINT("test pass\n");
    } else {
        LOG_PRINT("test failed\n");
        return FAILED;
    }
    return SUCCESS;
}