/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include "utils.h"
#include "acl/acl.h"
using namespace std;
namespace {
const std::string g_imagePathSeparator = ",";
const int STAT_SUCCESS = 0;
const std::string g_pathSeparator = "/";
}  // namespace
Result Utils::ReadBinFile(const std::string &fileName, void *&inputBuff, uint32_t &fileSize, bool isDevice) {
  if (CheckPathIsFile(fileName) == FAILED) {
    ERROR_LOG("%s is not a file", fileName.c_str());
    return FAILED;
  }

  std::ifstream binFile(fileName, std::ifstream::binary);
  if (binFile.is_open() == false) {
    ERROR_LOG("open file %s failed", fileName.c_str());
    return FAILED;
  }

  binFile.seekg(0, binFile.end);
  const std::streamoff fileLength = binFile.tellg();
  if (fileLength <= 0 || static_cast<uint64_t>(fileLength) > UINT32_MAX) {
    ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
    binFile.close();
    return FAILED;
  }
  uint32_t binFileBufferLen = static_cast<uint32_t>(fileLength);
  binFile.seekg(0, binFile.beg);

  aclError ret = ACL_SUCCESS;
  if (!isDevice) {  // app is running in host
    ret = aclrtMallocHost(&inputBuff, binFileBufferLen);
    if (ret != ACL_SUCCESS || inputBuff == nullptr) {
      ERROR_LOG("malloc binFileBufferData failed, binFileBufferLen is %u, errorCode is %d", binFileBufferLen,
                static_cast<int32_t>(ret));
      binFile.close();
      return FAILED;
    }
  } else {  // app is running in device
    ret = aclrtMalloc(&inputBuff, binFileBufferLen, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      ERROR_LOG("malloc device buffer failed. size is %u, errorCode is %d", binFileBufferLen,
                static_cast<int32_t>(ret));
      binFile.close();
      return FAILED;
    }
  }
  binFile.read(static_cast<char *>(inputBuff), binFileBufferLen);
  if (!binFile) {
    ERROR_LOG("read file %s failed", fileName.c_str());
    if (!isDevice) (void)aclrtFreeHost(inputBuff);
    else (void)aclrtFree(inputBuff);
    return FAILED;
  }
  binFile.close();
  fileSize = binFileBufferLen;
  return SUCCESS;
}

void *Utils::MemcpyToDeviceBuffer(const void *data, uint32_t size, aclrtRunMode runMode) {
  if ((data == nullptr) || (size == 0)) {
    ERROR_LOG("Copy data args invalid, data %p, size %d", data, size);
    return nullptr;
  }
  aclrtMemcpyKind policy = ACL_MEMCPY_HOST_TO_DEVICE;
  if (runMode == ACL_DEVICE) {
    policy = ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
  void *buffer = nullptr;
  aclError aclRet = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if ((aclRet != ACL_SUCCESS) || (buffer == nullptr)) {
    ERROR_LOG("Malloc memory failed, errorno:%d", aclRet);
    return nullptr;
  }
  aclRet = aclrtMemcpy(buffer, size, data, size, policy);
  if (aclRet != ACL_SUCCESS) {
    ERROR_LOG("Copy data to device failed, aclRet is %d", aclRet);
    (void)aclrtFree(buffer);
    return nullptr;
  }
  return buffer;
}

void Utils::GetAllFiles(const std::string &pathList, std::vector<string> &fileVec) {
  vector<string> pathVec;
  SplitPath(pathList, pathVec);

  for (string everyPath : pathVec) {
    // check path exist or not
    if (!IsPathExist(everyPath)) {
      ERROR_LOG("Failed to deal path=%s. Reason: not exist or cannot access.", everyPath.c_str());
      continue;
    }
    // get files in path and sub-path
    GetPathFiles(everyPath, fileVec);
  }
}

bool Utils::IsPathExist(const std::string &path) {
  ifstream file(path);
  if (!file) {
    return false;
  }
  return true;
}

void Utils::SplitPath(const std::string &path, std::vector<std::string> &pathVec) {
  size_t begin = 0;
  while (begin <= path.size()) {
    size_t end = path.find(g_imagePathSeparator, begin);
    if (end == std::string::npos) end = path.size();
    if (end > begin) pathVec.emplace_back(path.substr(begin, end - begin));
    if (end == path.size()) break;
    begin = end + g_imagePathSeparator.size();
  }
}

void Utils::GetPathFiles(const std::string &path, std::vector<std::string> &fileVec) {
  struct dirent *direntPtr = nullptr;
  DIR *dir = nullptr;
  if (IsDirectory(path)) {
    dir = opendir(path.c_str());
    if (dir == nullptr) return;
    while ((direntPtr = readdir(dir)) != nullptr) {
      // skip . and ..
      if (direntPtr->d_name[0] == '.') {
        continue;
      }

      // file path
      string fullPath = path + g_pathSeparator + direntPtr->d_name;
      // directory need recursion
      if (IsDirectory(fullPath)) {
        GetPathFiles(fullPath, fileVec);
      } else {
        // put file
        fileVec.emplace_back(fullPath);
      }
    }
    closedir(dir);
  } else {
    fileVec.emplace_back(path);
  }
}

bool Utils::IsDirectory(const std::string &path) {
  // get path stat
  struct stat buf;
  if (stat(path.c_str(), &buf) != STAT_SUCCESS) {
    return false;
  }
  // check
  return S_ISDIR(buf.st_mode);
}

Result Utils::CheckPathIsFile(const std::string &fileName) {
  struct stat sBuf;
  int fileStatus = stat(fileName.data(), &sBuf);
  if (fileStatus == -1) {
    ERROR_LOG("failed to get file");
    return FAILED;
  }
  if (S_ISREG(sBuf.st_mode) == 0) {
    ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Result Utils::MemcpyFileToDeviceBuffer(const std::string &fileName, void *&picDevBuffer, size_t inputBuffSize,
                                       bool isDevice) {
  void *inputBuff = nullptr;
  uint32_t fileSize = 0;
  auto ret = Utils::ReadBinFile(fileName, inputBuff, fileSize, isDevice);
  if (ret != SUCCESS) {
    ERROR_LOG("read bin file failed, file name is %s", fileName.c_str());
    return FAILED;
  }
  if (inputBuffSize != static_cast<size_t>(fileSize)) {
    ERROR_LOG("input image size[%u] is not equal to model input size[%zu]", fileSize, inputBuffSize);
    if (!isDevice) {
      (void)aclrtFreeHost(inputBuff);
    } else {
      (void)aclrtFree(inputBuff);
    }
    return FAILED;
  }
  if (!isDevice) {
    // if app is running in host, need copy data from host to device
    aclError aclRet = aclrtMemcpy(picDevBuffer, inputBuffSize, inputBuff, inputBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
      ERROR_LOG("memcpy failed. buffer size is %zu, errorCode is %d", inputBuffSize, static_cast<int32_t>(aclRet));
      (void)aclrtFreeHost(inputBuff);
      return FAILED;
    }
    (void)aclrtFreeHost(inputBuff);
  } else {  // app is running in device
    aclError aclRet = aclrtMemcpy(picDevBuffer, inputBuffSize, inputBuff, inputBuffSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
      ERROR_LOG("memcpy d2d failed. buffer size is %zu, errorCode is %d", inputBuffSize, static_cast<int32_t>(aclRet));
      (void)aclrtFree(inputBuff);
      return FAILED;
    }
    (void)aclrtFree(inputBuff);
  }
  return SUCCESS;
}

Result Utils::MemcpyFilesToDeviceBuffer(const std::vector<std::string> &fileNames, void **picDevBuffer,
                                        size_t &inputBuffSize, uint64_t batchSize, bool isDevice) {
  if (fileNames.size() < batchSize) {
    ERROR_LOG("input file num less than batch size:%zu", batchSize);
    return FAILED;
  }
  aclrtMemcpyKind memcpyKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
  if (!isDevice) {
    memcpyKind = ACL_MEMCPY_HOST_TO_DEVICE;
  }

  aclError aclRet = ACL_ERROR_NONE;
  for (size_t idx = 0; idx < batchSize; idx++) {
    void *picData = nullptr;
    uint32_t picDataSize = 0U;
    INFO_LOG("start to process file:%s", fileNames[idx].c_str());
    auto ret = Utils::ReadBinFile(fileNames[idx], picData, picDataSize, isDevice);
    if (ret != SUCCESS) {
      ERROR_LOG("read bin file failed, file name is %s", fileNames[idx].c_str());
      return FAILED;
    }

    if (*picDevBuffer == nullptr) {
      // finally: picDataSize * batchSize == inputBuffSize
      aclRet = aclrtMalloc(picDevBuffer, picDataSize * batchSize, ACL_MEM_MALLOC_HUGE_FIRST);
      if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("malloc device memory failed");
        if (!isDevice) (void)aclrtFreeHost(picData);
        else (void)aclrtFree(picData);
        return FAILED;
      }
    }

    aclRet = aclrtMemcpy(static_cast<uint8_t *>(*picDevBuffer) + inputBuffSize, picDataSize, picData, picDataSize,
                         memcpyKind);
    if (aclRet != ACL_SUCCESS) {
      ERROR_LOG("memcpy input data to device failed");
      if (!isDevice) (void)aclrtFreeHost(picData);
      else (void)aclrtFree(picData);
      return FAILED;
    }
    inputBuffSize += picDataSize;
    if (!isDevice) (void)aclrtFreeHost(picData);
    else (void)aclrtFree(picData);
  }

  return SUCCESS;
}
