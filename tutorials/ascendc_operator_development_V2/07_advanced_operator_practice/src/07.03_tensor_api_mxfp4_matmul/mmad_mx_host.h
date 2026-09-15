#ifndef MMAD_MX_HOST_H
#define MMAD_MX_HOST_H

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
int run_mmadmx_host(
    size_t a_file_size, size_t b_file_size, size_t as_file_size, size_t bs_file_size, size_t c_file_size,
    const char *output_path, LaunchFunc launch)
{
    CHECK_ACL_RET(aclInit(nullptr));
    int32_t device_id = 0;
    CHECK_ACL_RET(aclrtSetDevice(device_id));

    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));

    uint8_t *a_host = nullptr, *b_host = nullptr, *as_host = nullptr, *bs_host = nullptr, *c_host = nullptr;
    uint8_t *a_device = nullptr, *b_device = nullptr, *as_device = nullptr, *bs_device = nullptr, *c_device = nullptr;

    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&a_host), a_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&b_host), b_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&as_host), as_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&bs_host), bs_file_size));
    CHECK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&c_host), c_file_size));

    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&a_device), a_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&b_device), b_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&as_device), as_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&bs_device), bs_file_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMalloc(reinterpret_cast<void **>(&c_device), c_file_size, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t file_size = a_file_size;
    if (!ReadFile("./input/x1_gm.bin", file_size, a_host, a_file_size)) { return 1; }
    file_size = b_file_size;
    if (!ReadFile("./input/x2_gm.bin", file_size, b_host, b_file_size)) { return 1; }
    file_size = as_file_size;
    if (!ReadFile("./input/x1_scale_gm.bin", file_size, as_host, as_file_size)) { return 1; }
    file_size = bs_file_size;
    if (!ReadFile("./input/x2_scale_gm.bin", file_size, bs_host, bs_file_size)) { return 1; }

    CHECK_ACL_RET(aclrtMemcpy(a_device, a_file_size, a_host, a_file_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(b_device, b_file_size, b_host, b_file_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(as_device, as_file_size, as_host, as_file_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(bs_device, bs_file_size, bs_host, bs_file_size, ACL_MEMCPY_HOST_TO_DEVICE));

    launch(a_device, b_device, as_device, bs_device, c_device, stream);
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));

    CHECK_ACL_RET(aclrtMemcpy(c_host, c_file_size, c_device, c_file_size, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile(output_path, c_host, c_file_size)) { return 1; }

    (void)aclrtFree(a_device); (void)aclrtFree(b_device);
    (void)aclrtFree(as_device); (void)aclrtFree(bs_device);
    (void)aclrtFree(c_device);
    (void)aclrtFreeHost(a_host); (void)aclrtFreeHost(b_host);
    (void)aclrtFreeHost(as_host); (void)aclrtFreeHost(bs_host);
    (void)aclrtFreeHost(c_host);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
    return 0;
}

#endif
