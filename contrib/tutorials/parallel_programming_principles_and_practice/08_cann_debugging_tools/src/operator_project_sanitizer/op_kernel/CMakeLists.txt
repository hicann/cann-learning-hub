# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# 标准算子工程不能对普通 target 直接调用 target_compile_options。
# 不同 CANN 版本分别提供 add_ops_compile_options 或 npu_op_kernel_options，
# 两条分支都只把官方 -sanitizer 选项传给 Kernel 编译器。
set(LAB05_KERNEL_COMPILE_OPTIONS)
if(LAB05_ENABLE_SANITIZER)
    if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
        message(FATAL_ERROR "标准算子 msSanitizer 正式课程路径必须使用 Release："
                            "CANN 9.0.0 真机已验证 Debug(-g/-O0) 会产生 memcheck 假阴性。")
    endif()
    list(APPEND LAB05_KERNEL_COMPILE_OPTIONS -sanitizer)
    message(STATUS "LAB05_ENABLE_SANITIZER=ON：启用 Kernel 侧 msSanitizer 插桩")
else()
    message(STATUS "LAB05_ENABLE_SANITIZER=OFF：不启用 Kernel 侧 msSanitizer 插桩")
endif()

if(LAB05_KERNEL_COMPILE_OPTIONS)
    if(COMMAND add_ops_compile_options)
        message(STATUS "找到 add_ops_compile_options：一次性 apply ${LAB05_KERNEL_COMPILE_OPTIONS}")
        add_ops_compile_options(ALL OPTIONS ${LAB05_KERNEL_COMPILE_OPTIONS})
    elseif(COMMAND npu_op_kernel_options)
        message(STATUS "未找到 add_ops_compile_options，使用 npu_op_kernel_options：一次性 apply ${LAB05_KERNEL_COMPILE_OPTIONS}")
        npu_op_kernel_options(ascendc_kernels ALL OPTIONS ${LAB05_KERNEL_COMPILE_OPTIONS})
    elseif(LAB05_ENABLE_SANITIZER)
        message(FATAL_ERROR "LAB05_ENABLE_SANITIZER=ON，但当前 ASC/CANN 构建系统既不提供 "
                            "add_ops_compile_options 也不提供 npu_op_kernel_options，无法完成 "
                            "Kernel 侧 msSanitizer 插桩；请设置 LAB05_ENABLE_SANITIZER=OFF 或 "
                            "更换提供上述辅助函数的 CANN 工具链，禁止静默未插桩。")
    else()
        message(STATUS "当前工具链不提供标准 Kernel 选项辅助函数，跳过可选 Debug 参数")
    endif()
endif()

# CANN 9 不会把这里的自定义 -D 继续传入 Kernel 预处理，因此使用 configure_file
# 在构建目录生成所选故障模式的源码，不改写课程源码。
set(LAB05_KERNEL_GEN_DIR "${CMAKE_BINARY_DIR}/lab05_generated_kernel")
file(MAKE_DIRECTORY "${LAB05_KERNEL_GEN_DIR}")
configure_file(add_custom_msanitizer.cpp.in
    "${LAB05_KERNEL_GEN_DIR}/add_custom_msanitizer.cpp" @ONLY)
configure_file(add_custom_msanitizer_tiling.h
    "${LAB05_KERNEL_GEN_DIR}/add_custom_msanitizer_tiling.h" COPYONLY)
message(STATUS "LAB05_OPERATOR_FAULT_MODE=${LAB05_OPERATOR_FAULT_MODE}："
               "Kernel 源码由模板生成到 ${LAB05_KERNEL_GEN_DIR}")

# Kernel 只读取上一步生成的源码与 tiling header。
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${LAB05_KERNEL_GEN_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
