#ifndef BATCH_MATMUL_HOST_H
#define BATCH_MATMUL_HOST_H

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

template <typename T, typename LaunchFunc>
int run_batch_matmul_host(
    size_t a_file_size, size_t b_file_size, size_t c_file_size, size_t bias_file_size,
    const char *output_path, LaunchFunc launch)
{
    CHECK_ACL_RET(aclInit(nullptr));
    int32_t device_id = 0;
    CHECK_ACL_RET(aclrtSetDevice(device_id));

    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));

    uint8_t *a_host = nullptr;
    uint8_t *b_host = nullptr;
    uint8_t *c_host = nullptr;
    uint8_t *bias_host = nullptr;
    T *a_device = nullptr;
    T *b_device = nullptr;
    T *c_device = nullptr;
    T *bias_device = nullptr;

    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&a_host), a_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&b_host), b_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&c_host), c_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&bias_host), bias_file_size));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&a_device), a_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&b_device), b_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&c_device), c_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&bias_device), bias_file_size, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t file_size = a_file_size;
    if (!ReadFile("./input/x1_gm.bin", file_size, a_host, a_file_size)) { return 1; }
    file_size = b_file_size;
    if (!ReadFile("./input/x2_gm.bin", file_size, b_host, b_file_size)) { return 1; }
    file_size = bias_file_size;
    if (!ReadFile("./input/bias.bin", file_size, bias_host, bias_file_size)) { return 1; }

    CHECK_ACL_RET(aclrtMemcpy(a_device, a_file_size, a_host, a_file_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(b_device, b_file_size, b_host, b_file_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(bias_device, bias_file_size, bias_host, bias_file_size, ACL_MEMCPY_HOST_TO_DEVICE));

    launch(a_device, b_device, c_device, bias_device, stream);
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));

    CHECK_ACL_RET(aclrtMemcpy(c_host, c_file_size, c_device, c_file_size, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile(output_path, c_host, c_file_size)) { return 1; }

    (void)aclrtFree(a_device);
    (void)aclrtFree(b_device);
    (void)aclrtFree(c_device);
    (void)aclrtFree(bias_device);
    (void)aclrtFreeHost(a_host);
    (void)aclrtFreeHost(b_host);
    (void)aclrtFreeHost(c_host);
    (void)aclrtFreeHost(bias_host);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
    return 0;
}

#endif
