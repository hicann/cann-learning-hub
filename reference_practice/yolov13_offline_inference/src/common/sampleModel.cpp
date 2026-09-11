/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sampleModel.h"
#include "utils.h"

aclmdlDataset *AclModelOutput::GetDataSet() {
  return dataset_;
}

aclmdlDataset *AclModelInput::GetDataSet() {
  return dataset_;
}

size_t AclModelWork::GetModelWorkSize() {
  return modelWorkSize_;
}

void *AclModelWork::GetModelWorkPtr() {
  return modelWorkPtr_;
}

size_t AclModelWeight::GetModelWeightSize() {
  return modelWeightSize_;
}

void *AclModelWeight::GetModelWeightPtr() {
  return modelWeightPtr_;
}

aclmdlDesc *AclModelDesc::GetModelDesc() {
  return modelDesc_;
}

AclModelInput::AclModelInput(void *inputDataBuffer, size_t bufferSize, aclmdlDesc *modelDesc) {
  CHECK_NOT_NULL(modelDesc);
  uint32_t dataNum = aclmdlGetNumInputs(modelDesc);
  dataset_ = aclmdlCreateDataset();
  CHECK_NOT_NULL(dataset_);
  aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
  CHECK_NOT_NULL(inputData);
  CHECK(aclmdlAddDatasetBuffer(dataset_, inputData));

  size_t dynamicIdx = 0;
  auto ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &dynamicIdx);
  if ((ret == ACL_SUCCESS) && (dynamicIdx == (dataNum - 1))) {
    size_t dataLen = aclmdlGetInputSizeByIndex(modelDesc, dynamicIdx);
    void *data = nullptr;
    CHECK(aclrtMalloc(&data, dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    aclDataBuffer *dataBuf = aclCreateDataBuffer(data, dataLen);
    CHECK_NOT_NULL(dataBuf);
    CHECK(aclmdlAddDatasetBuffer(dataset_, dataBuf));
  }
}

AclModelInput::~AclModelInput() {
  CHECK_NOT_NULL(dataset_);

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset_, i);
    CHECK(aclDestroyDataBuffer(dataBuffer));
  }
  CHECK(aclmdlDestroyDataset(dataset_));
  dataset_ = nullptr;
}

AclModelOutput::AclModelOutput(aclmdlDesc *modelDesc) {
  CHECK_NOT_NULL(modelDesc);

  dataset_ = aclmdlCreateDataset();
  CHECK_NOT_NULL(dataset_);

  size_t outputSize = aclmdlGetNumOutputs(modelDesc);
  for (size_t i = 0; i < outputSize; ++i) {
    size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
    void *outputBuffer = nullptr;
    CHECK(aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
    CHECK_NOT_NULL(outputData);
    CHECK(aclmdlAddDatasetBuffer(dataset_, outputData));
  }
}

AclModelOutput::~AclModelOutput() {
  if (dataset_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset_, i);
    void *data = aclGetDataBufferAddr(dataBuffer);
    CHECK(aclrtFree(data));
    CHECK(aclDestroyDataBuffer(dataBuffer));
  }
  CHECK(aclmdlDestroyDataset(dataset_));
  dataset_ = nullptr;
}
