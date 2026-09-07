# ============================================================================
# CMake 构建配置 - Ascend C Kernel 直调 - rope_optimized 算子
# ============================================================================
#
# 目录结构:
#   op_kernel/     - NPU 计算层（rope_optimized_tiling.h + rope_optimized_kernel.asc）
#   op_host/       - Host 直调层（rope_optimized.asc + data_utils.h）
#   op_extension/  - PyTorch 接入层（rope_optimized_torch.cpp + register.cpp + ops.h）
#   scripts/       - 测试脚本
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(rope_optimized LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ============================================================================
# Target 1: 可执行文件（直调验证）
# ============================================================================

add_executable(rope_optimized op_host/rope_optimized.asc)

target_link_libraries(rope_optimized PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(rope_optimized PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# NPU 架构配置: DAV_2201 (Ascend 910B4)
target_compile_options(rope_optimized PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ============================================================================
# Target 2: TORCH_LIBRARY 模块 (librope_optimized_ops.so)
#   Python: torch.ops.load_library("build/librope_optimized_ops.so")
#           torch.ops.npu.rope_optimized(x, cos, sin)
# ============================================================================

find_package(Python3 COMPONENTS Interpreter Development QUIET)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    ERROR_QUIET
)

if (TORCH_CMAKE_PREFIX_PATH)
    find_package(Torch QUIET HINTS ${TORCH_CMAKE_PREFIX_PATH})
else()
    find_package(Torch QUIET)
endif()

if (Torch_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE TORCH_NPU_PATH
        ERROR_QUIET
    )
    set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
    set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

    add_library(rope_optimized_ops SHARED
        op_kernel/rope_optimized_kernel.asc
        op_extension/rope_optimized_torch.cpp
        op_extension/register.cpp
    )

    target_include_directories(rope_optimized_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
        ${TORCH_INCLUDE_DIRS}
        ${TORCH_NPU_INCLUDE_DIRS}
        ${ACL_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
    )

    target_link_libraries(rope_optimized_ops PRIVATE
        torch_npu
        tiling_api
        register
        platform
        unified_dlog
        dl
        m
        graph_base
        ascendcl
        ascendc_runtime
    )

    target_link_directories(rope_optimized_ops PRIVATE
        ${ACL_LIB_DIR}
        "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
        ${TORCH_NPU_LIBRARIES}
    )

    target_compile_options(rope_optimized_ops PRIVATE
        ${TORCH_CXX_FLAGS}
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    )

    message(STATUS "TORCH_LIBRARY module 'librope_optimized_ops.so' will be built")
else()
    message(STATUS "Torch not found, skip librope_optimized_ops.so (直调验证只需 rope_optimized)")
endif()
