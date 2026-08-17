/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UTILS_H
#define UTILS_H

#include <cstring>
#include <algorithm>
#include <acl/acl.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/file.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include "shmem.h"

#define INFO_LOG(fmt, args...) do { \
    fprintf(stdout, "[INFO] %s:%d " fmt "\n", __FILE__, __LINE__, ##args); \
    fflush(stdout); \
} while(0)
#define WARNING_LOG(fmt, args...) do { \
    fprintf(stdout, "[WARNING] " fmt "\n", ##args); \
    fflush(stdout); \
} while(0)
#define ERROR_LOG(fmt, args...) do { \
    fprintf(stdout, "[ERROR] " fmt "\n", ##args); \
    fflush(stdout); \
} while(0)

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

// ACL错误检查宏
#define ACL_CHECK_WITH_RET(cond, log_func, return_expr)                                                  \
    do {                                                                                                 \
        if ((cond) != ACL_ERROR_NONE) {                                                                  \
            log_func;                                                                                    \
            return_expr;                                                                                 \
        }                                                                                                \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    // lock
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // move ptr to specified offset
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // write data
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << strerror(errno) << std::endl;
    }

    // unlock
    flock(fd, LOCK_UN);

    close(fd);
    return true;
}

inline int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port, aclshmemx_uniqueid_t default_flag_uid,
                       aclshmemx_init_attr_t *attributes)
{
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;

    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }
    attributes->ip_port[ip_len] = '\0';

    attributes->local_mem_size = local_mem_size;

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};

    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    return ACLSHMEM_SUCCESS;
}

// -----------------------------
// Demo host-side common helpers
// -----------------------------

inline void InitData(uint8_t **hostPtr, uint8_t **devicePtr, size_t aSize, const std::string &path = "")
{
    std::cout << path << std::endl;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(devicePtr), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(hostPtr), aSize));
    if (path.empty()) {
        return;
    }
    ReadFile(path, *hostPtr, aSize);
    ACL_CHECK(aclrtMemcpy(*devicePtr, aSize, *hostPtr, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

inline void FinalizeData(uint8_t *hostPtr, uint8_t *devicePtr, size_t aSize = 0, const std::string &path = "")
{
    std::cout << path << std::endl;
    if (!path.empty() && aSize > 0) {
        ACL_CHECK(aclrtMemcpy(hostPtr, aSize, devicePtr, aSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(path, hostPtr, aSize);
    }
    ACL_CHECK(aclrtFreeHost(reinterpret_cast<void *>(hostPtr)));
    ACL_CHECK(aclrtFree(reinterpret_cast<void *>(devicePtr)));
}

inline std::string GetInputFilePath(const std::string &tensorName, int rankId)
{
    return "./input/chip_" + std::to_string(rankId) + "/" + tensorName + "_" + std::to_string(rankId) + ".bin";
}

inline std::string GetOutputFilePath(const std::string &tensorName, int rankId)
{
    return "./output/chip_" + std::to_string(rankId) + "/" + tensorName + "_" + std::to_string(rankId) + ".bin";
}

using MoeRankWorkerFunc = int (*)(int rankNum, int rankId, int bs, uint64_t shmemSpaceSize);

inline int MoeDemoForkMain(int argc, char *argv[], MoeRankWorkerFunc worker, uint64_t defaultShmemBytes)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rankNum> <bs> [shmemSizeInBytes]" << std::endl;
        return -1;
    }
    const int rankNum = std::atoi(argv[1]);
    const int bs = std::atoi(argv[2]);
    uint64_t shmemSpaceSize = defaultShmemBytes;
    if (argc >= 4) {
        shmemSpaceSize = std::stoull(argv[3]);
    }

    INFO_LOG("Master (PID=%d) will fork %d processes", getpid(), rankNum);

    std::vector<pid_t> pids(static_cast<size_t>(rankNum));
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        const pid_t pid = fork();
        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            return -1;
        }
        if (pid == 0) {
            const int ret = worker(rankNum, rankId, bs, shmemSpaceSize);
            std::exit(ret);
        }
        pids[static_cast<size_t>(rankId)] = pid;
        INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
    }

    bool allSuccess = true;
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        int status = 0;
        const pid_t pid = pids[static_cast<size_t>(rankId)];
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            allSuccess = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (allSuccess ? "SUCCESS" : "FAILURE") << std::endl;
    return allSuccess ? 0 : -1;
}

#endif // UTILS_H