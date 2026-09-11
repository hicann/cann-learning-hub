/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAMPLE_ACL_COMMON_UTILS_H_
#define SAMPLE_ACL_COMMON_UTILS_H_

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <unistd.h>
#include <cstring>
#include <dirent.h>
#include <regex>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "acl/acl.h"

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(expr) (static_cast<bool>(__builtin_expect(static_cast<bool>(expr), 1)))
#define UNLIKELY(expr) (static_cast<bool>(__builtin_expect(static_cast<bool>(expr), 0)))
#else
#define LIKELY(expr) (expr)
#define UNLIKELY(expr) (expr)
#endif

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#define CHECK(status)                                                                                    \
  do {                                                                                                   \
    if ((status) != ACL_SUCCESS) {                                                                       \
      std::cerr << "Ascendcl failure at " << __FILE__ << ":" << __LINE__ << ": " << status << std::endl; \
      exit(EXIT_FAILURE);                                                                                \
    }                                                                                                    \
  } while (0)

#define CHECK_NOT_NULL(val)                                                              \
  do {                                                                                   \
    if (UNLIKELY((val) == nullptr)) {                                                    \
      std::cerr << "Ascendcl failure, param must not be null: " << #val << std::endl; \
      exit(EXIT_FAILURE);                                                                \
    }                                                                                    \
  } while (0)

enum Result { SUCCESS = 0, FAILED = 1 };

class Utils {
 public:
  static Result ReadBinFile(const std::string &fileName, void *&inputBuff, uint32_t &fileSize, bool isDevice);
  static void *MemcpyToDeviceBuffer(const void *data, uint32_t size, aclrtRunMode runMode);
  static void GetAllFiles(const std::string &pathList, std::vector<std::string> &fileVec);
  static void SplitPath(const std::string &path, std::vector<std::string> &pathVec);
  static void GetPathFiles(const std::string &path, std::vector<std::string> &fileVec);
  static bool IsDirectory(const std::string &path);
  static bool IsPathExist(const std::string &path);
  static Result CheckPathIsFile(const std::string &fileName);
  static Result MemcpyFileToDeviceBuffer(const std::string &fileName, void *&picDevBuffer, size_t inputBuffSize,
                                         bool isDevice);
  static Result MemcpyFilesToDeviceBuffer(const std::vector<std::string> &fileNames, void **picDevBuffer,
                                          size_t &inputBuffSize, uint64_t batchSize, bool isDevice);
};

#endif  // SAMPLE_ACL_COMMON_UTILS_H_
