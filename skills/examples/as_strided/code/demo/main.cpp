/**
 * Demo executable for as_strided operator profiling.
 * Uses aclnn API to directly invoke the custom Ascend C kernel.
 * Usage: msprof op --warm-up=10 --output=./msprof_output ./demo_as_strided
 */
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_as_strided.h"

#define ACL_CHECK(call) do { \
    aclError ret = (call); \
    if (ret != ACL_SUCCESS) { \
        std::cerr << "ACL error at " << __FILE__ << ":" << __LINE__ \
                  << " ret=" << ret << std::endl; \
        return -1; \
    } \
} while(0)

#define ACLNN_CHECK(call) do { \
    aclnnStatus ret = (call); \
    if (ret != 0) { \
        std::cerr << "ACLNN error at " << __FILE__ << ":" << __LINE__ \
                  << " ret=" << ret << std::endl; \
        return -1; \
    } \
} while(0)

int main() {
    // Initialize ACL
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = 0;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    // ============================================================
    // Test case: T9 - large shape, Path A
    // input_shape=[50000], size=[10000], stride=[3], storage_offset=0, dtype=float32
    // ============================================================
    int64_t inputTotal = 50000;
    int64_t outputTotal = 10000;
    std::vector<int64_t> inputShape = {inputTotal};
    std::vector<int64_t> outputShape = {outputTotal};
    std::vector<int64_t> sizeList = {10000};
    std::vector<int64_t> strideList = {3};
    int64_t storageOffset = 0;

    size_t inputBytes = inputTotal * sizeof(float);
    size_t outputBytes = outputTotal * sizeof(float);

    // Allocate device memory
    void* inputDev = nullptr;
    void* outputDev = nullptr;
    ACL_CHECK(aclrtMalloc(&inputDev, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&outputDev, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // Initialize input data on host and copy to device
    std::vector<float> inputHost(inputTotal);
    for (int64_t i = 0; i < inputTotal; i++) {
        inputHost[i] = static_cast<float>(i);
    }
    ACL_CHECK(aclrtMemcpy(inputDev, inputBytes, inputHost.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // Create aclTensors
    // aclCreateTensor(viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum, tensorData)
    int64_t inputStride[1] = {1};
    int64_t outputStride[1] = {1};
    aclTensor* inputTensor = aclCreateTensor(
        inputShape.data(), 1, ACL_FLOAT, inputStride, 0, ACL_FORMAT_ND,
        inputShape.data(), 1, inputDev);
    aclTensor* outputTensor = aclCreateTensor(
        outputShape.data(), 1, ACL_FLOAT, outputStride, 0, ACL_FORMAT_ND,
        outputShape.data(), 1, outputDev);

    // Create aclIntArray for size and stride
    aclIntArray* sizeArray = aclCreateIntArray(sizeList.data(), sizeList.size());
    aclIntArray* strideArray = aclCreateIntArray(strideList.data(), strideList.size());

    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACLNN_CHECK(aclnnAsStridedGetWorkspaceSize(
        inputTensor, sizeArray, strideArray, storageOffset,
        outputTensor, &workspaceSize, &executor));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Warm-up runs
    for (int i = 0; i < 5; i++) {
        ACLNN_CHECK(aclnnAsStrided(workspace, workspaceSize, executor, stream));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Measured runs
    const int numIters = 20;
    for (int i = 0; i < numIters; i++) {
        // Re-create executor for each call
        executor = nullptr;
        workspaceSize = 0;
        ACLNN_CHECK(aclnnAsStridedGetWorkspaceSize(
            inputTensor, sizeArray, strideArray, storageOffset,
            outputTensor, &workspaceSize, &executor));
        if (workspaceSize > 0 && workspace == nullptr) {
            ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        ACLNN_CHECK(aclnnAsStrided(workspace, workspaceSize, executor, stream));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::cout << "Profiling workload completed: " << numIters << " iterations" << std::endl;
    std::cout << "Input shape: [50000], Output shape: [10000], stride=[3], offset=0, dtype=float32" << std::endl;

    // Cleanup
    if (workspace) aclrtFree(workspace);
    aclDestroyIntArray(sizeArray);
    aclDestroyIntArray(strideArray);
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclrtFree(inputDev);
    aclrtFree(outputDev);
    aclrtDestroyStream(stream);
    ACL_CHECK(aclrtResetDevice(deviceId));
    aclFinalize();

    return 0;
}
