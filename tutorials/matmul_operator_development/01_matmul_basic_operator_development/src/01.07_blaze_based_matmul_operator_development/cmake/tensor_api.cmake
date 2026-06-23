# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(TARGET ops_tensor_api)
    return()
endif()

include(FetchContent)

# Pin ops-tensor to a known-good commit; do not track master.
set(OPS_TENSOR_REPO "https://gitcode.com/cann/ops-tensor.git")
set(OPS_TENSOR_COMMIT "751b32e477e2a71915b3d91ec4a4946403135d3e")
set(OPS_TENSOR_LOCAL_PATH "${PROJECT_SOURCE_DIR}/third_party/ops-tensor")
set(OPS_TENSOR_HEADER_REL "include/tensor_api/include/tensor_api/tensor.h")

# Reuse a local checkout when present (optional offline cache).
if(EXISTS "${OPS_TENSOR_LOCAL_PATH}/${OPS_TENSOR_HEADER_REL}")
    set(OPS_TENSOR_PATH "${OPS_TENSOR_LOCAL_PATH}")
    message(STATUS "Using local ops-tensor at ${OPS_TENSOR_PATH}")
else()
    FetchContent_Declare(
        ops_tensor
        GIT_REPOSITORY ${OPS_TENSOR_REPO}
        GIT_TAG ${OPS_TENSOR_COMMIT}
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(ops_tensor)
    if(NOT ops_tensor_POPULATED)
        message(STATUS "Fetching ops-tensor ${OPS_TENSOR_COMMIT} via FetchContent...")
        FetchContent_Populate(ops_tensor)
    endif()
    set(OPS_TENSOR_PATH ${ops_tensor_SOURCE_DIR})
    message(STATUS "ops-tensor ready at ${OPS_TENSOR_PATH}")
endif()

if(NOT EXISTS "${OPS_TENSOR_PATH}/${OPS_TENSOR_HEADER_REL}")
    message(FATAL_ERROR
        "ops-tensor headers not found at ${OPS_TENSOR_PATH}.\n"
        "Expected: ${OPS_TENSOR_HEADER_REL}\n"
        "Repository: ${OPS_TENSOR_REPO}\n"
        "Pinned commit: ${OPS_TENSOR_COMMIT}")
endif()

add_library(ops_tensor_api INTERFACE)
add_library(ops_tensor::api ALIAS ops_tensor_api)

# ops-tensor headers must precede CANN bundled tensor_api under asc/include.
target_include_directories(ops_tensor_api BEFORE INTERFACE
    "${OPS_TENSOR_PATH}/include"
    "${OPS_TENSOR_PATH}/include/tensor_api"
    "${OPS_TENSOR_PATH}/include/tensor_api/include"
    "${ASCEND_DIR}/asc"
)
