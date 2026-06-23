# ----------------------------------------------------------------------------
# MXFP8 SWAT 2-buffer 量化 MatMul — 教程最小构建配置
# 依赖 CANN 工具链；ops-tensor 由 cmake/tensor_api.cmake 通过 FetchContent 自动拉取（@ 751b32e）
# Blaze 头文件已内置于 include/
# 编译参数: cmake -DNPU_ARCH=dav-3510
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

set(VALID_NPU_ARCHS dav-3510)
if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    set(NPU_ARCH dav-3510 CACHE STRING "NPU architecture (Ascend950 PR/DT = dav-3510)")
endif()
if(NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "This tutorial only supports NPU_ARCH=dav-3510 (Ascend950 PR/DT). Got: ${NPU_ARCH}")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake)
find_package(ASC REQUIRED)
project(quant_matmul_mxfp8_tutorial LANGUAGES C CXX ASC)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensor_api.cmake)

set(MATMUL_ASCENDC_COMPILE_OPTS
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)

add_executable(quant_matmul_mxfp8_tutorial quant_matmul_mxfp8_tutorial.asc)
target_include_directories(quant_matmul_mxfp8_tutorial PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)
target_compile_options(quant_matmul_mxfp8_tutorial PRIVATE ${MATMUL_ASCENDC_COMPILE_OPTS})
target_link_libraries(quant_matmul_mxfp8_tutorial PRIVATE
    m
    dl
    platform
    tiling_api
    ops_tensor::api
)

install(TARGETS quant_matmul_mxfp8_tutorial RUNTIME DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION .
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.sh"
)
