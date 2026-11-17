
cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(ASC REQUIRED)
project(kernel_pytorch_call_torch_library LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_QUERY_RESULT
)
if(NOT TORCH_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to query Torch CMake prefix")
endif()
string(REPLACE "\r" "\n" TORCH_CMAKE_PREFIX_PATH_LINES "${TORCH_CMAKE_PREFIX_PATH_RAW}")
string(REGEX MATCH "[^\n]+$" TORCH_CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH_LINES}")
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_QUERY_RESULT
)
if(NOT TORCH_NPU_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to query torch_npu path")
endif()
string(REPLACE "\r" "\n" TORCH_NPU_PATH_LINES "${TORCH_NPU_PATH_RAW}")
string(REGEX MATCH "[^\n]+$" TORCH_NPU_PATH "${TORCH_NPU_PATH_LINES}")
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(ascendc_kernels SHARED
    ${CMAKE_CURRENT_LIST_DIR}/../common/add_custom.asc
)

add_library(ascendc_ops SHARED
    torch_binding.cpp
)

target_include_directories(ascendc_ops SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
)

target_link_directories(ascendc_ops PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

target_link_libraries(ascendc_ops PRIVATE
    ascendc_kernels
    torch_npu
    Python3::Python
)

target_compile_options(ascendc_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(ascendc_ops PRIVATE
    ${TORCH_CXX_FLAGS}
)

set_target_properties(ascendc_ops PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
