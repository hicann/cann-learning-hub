# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# 标准算子工程版 msSanitizer 示例。
# LAB05_OPERATOR_FAULT_MODE 选择故障；LAB05_ENABLE_SANITIZER 控制 Kernel 插桩。
# 故障模式通过 configure_file 写入构建目录，源码树保持不变。

cmake_minimum_required(VERSION 3.16.0)
project(operator_project_sanitizer)

# 默认同时生成 910B 与 A3 系列(ascend910_93，Ascend910_9362 对应此标识)的算子产物；
# 如需仅编某一平台，可在配置阶段用 -DASCEND_COMPUTE_UNIT=<单元> 覆盖。
set(ASCEND_COMPUTE_UNIT "ascend910b;ascend910_93" CACHE STRING
    "目标计算单元列表，默认 ascend910b 与 ascend910_93")
find_package(ASC REQUIRED)

# 该案例在 Release 下进行 sanitizer 检测。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(package_name customize)

# 选择一种故障模式。
set(LAB05_OPERATOR_FAULT_MODE "baseline" CACHE STRING
    "Lab05 算子故障模式：baseline / memcheck / racecheck / initcheck / synccheck")
set_property(CACHE LAB05_OPERATOR_FAULT_MODE PROPERTY STRINGS
    baseline memcheck racecheck initcheck synccheck)
set(LAB05_ALLOWED_FAULT_MODES baseline memcheck racecheck initcheck synccheck)
if(NOT LAB05_OPERATOR_FAULT_MODE IN_LIST LAB05_ALLOWED_FAULT_MODES)
    message(FATAL_ERROR
        "LAB05_OPERATOR_FAULT_MODE 只能是 baseline、memcheck、racecheck、initcheck 或 synccheck，"
        "实际为 '${LAB05_OPERATOR_FAULT_MODE}'")
endif()

# 将所选模式转换为模板需要的 0/1 字面量。
foreach(mode IN ITEMS MEMCHECK RACECHECK INITCHECK SYNCCHECK)
    string(TOLOWER "${mode}" mode_lower)
    if(LAB05_OPERATOR_FAULT_MODE STREQUAL "${mode_lower}")
        set(LAB05_FAULT_MODE_${mode}_LITERAL 1)
    else()
        set(LAB05_FAULT_MODE_${mode}_LITERAL 0)
    endif()
endforeach()
message(STATUS
    "LAB05 fault literals: memcheck=${LAB05_FAULT_MODE_MEMCHECK_LITERAL}, "
    "racecheck=${LAB05_FAULT_MODE_RACECHECK_LITERAL}, "
    "initcheck=${LAB05_FAULT_MODE_INITCHECK_LITERAL}, "
    "synccheck=${LAB05_FAULT_MODE_SYNCCHECK_LITERAL} "
    "(mode=${LAB05_OPERATOR_FAULT_MODE})")

# 控制 Kernel 侧 msSanitizer 插桩。
set(LAB05_ENABLE_SANITIZER "ON" CACHE BOOL
    "Lab05：是否启用 Kernel 侧 msSanitizer 插桩(-sanitizer)，由 op_kernel/CMakeLists.txt 的两级兼容分支落实")
if(LAB05_ENABLE_SANITIZER)
    message(STATUS "LAB05_ENABLE_SANITIZER=ON：将在 op_kernel/CMakeLists.txt 中启用 msSanitizer 插桩")
else()
    message(STATUS "LAB05_ENABLE_SANITIZER=OFF：不启用 msSanitizer 插桩")
endif()

# 故障模式必须配合插桩，避免假阴性。
if(NOT LAB05_OPERATOR_FAULT_MODE STREQUAL "baseline" AND NOT LAB05_ENABLE_SANITIZER)
    message(FATAL_ERROR
        "LAB05_OPERATOR_FAULT_MODE=${LAB05_OPERATOR_FAULT_MODE} 必须配合 "
        "LAB05_ENABLE_SANITIZER=ON：无插桩会产生假阴性，配置阶段直接拒绝。")
endif()

message(STATUS "LAB05_OPERATOR_FAULT_MODE=${LAB05_OPERATOR_FAULT_MODE}")
message(STATUS "LAB05_ENABLE_SANITIZER=${LAB05_ENABLE_SANITIZER}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
