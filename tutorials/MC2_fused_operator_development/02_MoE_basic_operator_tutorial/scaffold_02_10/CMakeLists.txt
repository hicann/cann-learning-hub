# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# ----------------------------------------------------------------------------
#
# 量化 Dispatch 算子练习 - 自包含独立工程
# 不依赖 cann-samples 父项目，所有必需的 cmake 模块已内置到 cmake/ 目录。

cmake_minimum_required(VERSION 3.16)

set(VALID_NPU_ARCHS dav-3510 dav-2201)
if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    message(FATAL_ERROR "NPU_ARCH is required. Example: cmake -S . -B build -DNPU_ARCH=dav-3510")
endif()
if(NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported: dav-3510, dav-2201")
endif()

include(cmake/ascend.cmake)
find_package(ASC)
project(quant_dispatch_practice LANGUAGES C CXX ASC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/shmem.cmake)

add_executable(moe_dispatch_and_combine_story src/dispatch_and_combine_final.asc)
target_compile_definitions(moe_dispatch_and_combine_story PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(moe_dispatch_and_combine_story PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_options(moe_dispatch_and_combine_story PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)
target_link_libraries(moe_dispatch_and_combine_story
    PRIVATE
    m
    dl
    platform
    tiling_api
    cann_samples::shmem
    ascendcl
    runtime
    stdc++
)
