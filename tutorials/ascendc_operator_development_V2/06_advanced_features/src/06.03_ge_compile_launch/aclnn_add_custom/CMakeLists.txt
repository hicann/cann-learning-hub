# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16.0)
project(add_custom_opp)

find_package(ASC REQUIRED)

set(package_name ${vendor_name})
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}/
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
