/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAMPLE_ACL_COMMON_SAMPLE_MODEL_H_
#define SAMPLE_ACL_COMMON_SAMPLE_MODEL_H_

#include <cstdlib>
#include "acl/acl.h"
#include "utils.h"

class AclModelWeight {
 public:
  AclModelWeight(size_t modelWeightSize) {
    modelWeightSize_ = modelWeightSize;
    CHECK(aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  ~AclModelWeight() {
    CHECK(aclrtFree(modelWeightPtr_));
    modelWeightPtr_ = nullptr;
    modelWeightSize_ = 0;
  }
  size_t GetModelWeightSize();
  void *GetModelWeightPtr();

 private:
  size_t modelWeightSize_;
  void *modelWeightPtr_;
};

class AclModelWork {
 public:
  AclModelWork(size_t modelWorkSize) {
    modelWorkSize_ = modelWorkSize;
    CHECK(aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  ~AclModelWork() {
    CHECK(aclrtFree(modelWorkPtr_));
    modelWorkPtr_ = nullptr;
    modelWorkSize_ = 0;
  }
  size_t GetModelWorkSize();
  void *GetModelWorkPtr();

 private:
  size_t modelWorkSize_;
  void *modelWorkPtr_;
};

class AclModelDesc {
 public:
  AclModelDesc(uint32_t modelId) {
    modelId_ = modelId;
    modelDesc_ = aclmdlCreateDesc();
    CHECK_NOT_NULL(modelDesc_);
    CHECK(aclmdlGetDesc(modelDesc_, modelId));
  }
  ~AclModelDesc() {
    CHECK(aclmdlUnload(modelId_));
    CHECK(aclmdlDestroyDesc(modelDesc_));
    modelDesc_ = nullptr;
  }
  aclmdlDesc *GetModelDesc();

 private:
  uint32_t modelId_;
  aclmdlDesc *modelDesc_;
};

class AclModelInput {
 public:
  AclModelInput(void *inputDataBuffer, size_t bufferSize, aclmdlDesc *modelDesc);
  ~AclModelInput();
  aclmdlDataset *GetDataSet();

 private:
  aclmdlDataset *dataset_;
};

class AclModelOutput {
 public:
  AclModelOutput(aclmdlDesc *modelDesc);
  ~AclModelOutput();
  aclmdlDataset *GetDataSet();

 private:
  aclmdlDataset *dataset_;
};

#endif  // SAMPLE_ACL_COMMON_SAMPLE_MODEL_H_
