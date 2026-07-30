#ifndef TENSOR_MMAD_HOST_H
#define TENSOR_MMAD_HOST_H

#include "acl/acl.h"
#include "data_utils.h"

#include <cstdint>
#include <cstdio>
#include <string>

#define CHECK_ACL_RET(expr)                                      \
    do {                                                         \
        aclError ret = (expr);                                   \
        if (ret != ACL_SUCCESS) {                                \
            ERROR_LOG("%s failed, ret = %d", #expr, ret);        \
            return 1;                                            \
        }                                                        \
    } while (0)

template <typename LaunchFunc>
int RunMatmulHost(size_t inputSize, size_t outputSize, const char *outputPath, LaunchFunc launch)
{
    CHECK_ACL_RET(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL_RET(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));

    uint8_t *xHost = nullptr;
    uint8_t *yHost = nullptr;
    uint8_t *zHost = nullptr;
    half *xDevice = nullptr;
    half *yDevice = nullptr;
    half *zDevice = nullptr;

    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&xHost), inputSize));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&yHost), inputSize));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&zHost), outputSize));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&xDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&yDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&zDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t fileSize = inputSize;
    if (!ReadFile("./input/input_x.bin", fileSize, xHost, inputSize)) {
        return 1;
    }
    fileSize = inputSize;
    if (!ReadFile("./input/input_y.bin", fileSize, yHost, inputSize)) {
        return 1;
    }

    CHECK_ACL_RET(aclrtMemcpy(xDevice, inputSize, xHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(yDevice, inputSize, yHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    launch(xDevice, yDevice, zDevice, stream);
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));

    CHECK_ACL_RET(aclrtMemcpy(zHost, outputSize, zDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile(outputPath, zHost, outputSize)) {
        return 1;
    }

    (void)aclrtFree(xDevice);
    (void)aclrtFree(yDevice);
    (void)aclrtFree(zDevice);
    (void)aclrtFreeHost(xHost);
    (void)aclrtFreeHost(yHost);
    (void)aclrtFreeHost(zHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return 0;
}

#endif
