# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510, dav-2201")

find_package(ASC REQUIRED)
project(aclgraph_pybind LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(query_python_path output_var expression)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "${expression}"
        OUTPUT_VARIABLE query_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE query_result
    )
    if(NOT query_result EQUAL 0)
        message(FATAL_ERROR "Failed to query ${output_var} from Python")
    endif()
    string(REPLACE "\r" "\n" query_lines "${query_output}")
    string(REGEX MATCH "[^\n]+$" query_path "${query_lines}")
    set(${output_var} "${query_path}" PARENT_SCOPE)
endfunction()

query_python_path(TORCH_CMAKE_PREFIX_PATH "import torch; print(torch.utils.cmake_prefix_path)")
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

query_python_path(TORCH_NPU_PATH "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))")
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

query_python_path(PYBIND11_CMAKE_PREFIX_PATH "import pybind11; print(pybind11.get_cmake_dir())")
find_package(pybind11 REQUIRED HINTS ${PYBIND11_CMAKE_PREFIX_PATH})

pybind11_add_module(ascendc_ops SHARED
    add_custom.asc
)

target_include_directories(ascendc_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
)

target_link_libraries(ascendc_ops PRIVATE
    torch_npu
)

target_link_directories(ascendc_ops PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(ascendc_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
