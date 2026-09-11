/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAMPLE_ACL_COMMON_SAMPLE_DEVICE_H_
#define SAMPLE_ACL_COMMON_SAMPLE_DEVICE_H_

#include "acl/acl.h"
#include "utils.h"

class AclInstance {
 public:
  AclInstance(const char *aclConfigPath) {
    CHECK(aclInit(aclConfigPath));
  }
  ~AclInstance() {
    CHECK(aclFinalize());
  }
};

class AclStream {
 public:
  AclStream() {
    CHECK(aclrtCreateStream(&stream_));
  }
  ~AclStream() {
    CHECK(aclrtDestroyStream(stream_));
    stream_ = nullptr;
  }

 private:
  aclrtStream stream_;
};

class AclContext {
 public:
  AclContext(int32_t deviceId) {
    CHECK(aclrtCreateContext(&context_, deviceId));
  }
  ~AclContext() {
    CHECK(aclrtDestroyContext(context_));
  }

 private:
  aclrtContext context_;
};

class AclDevice {
 public:
  AclDevice(int32_t deviceId) {
    deviceId_ = deviceId;
    CHECK(aclrtSetDevice(deviceId_));
  }
  ~AclDevice() {
    CHECK(aclrtResetDevice(deviceId_));
  }

 private:
  int32_t deviceId_;
};

#endif  // SAMPLE_ACL_COMMON_SAMPLE_DEVICE_H_
