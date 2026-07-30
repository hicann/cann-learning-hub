cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "Tensor API only supports dav-3510 in this sample")

if(NOT CMAKE_ASC_ARCHITECTURES STREQUAL "dav-3510")
    message(FATAL_ERROR "tensor_api_matmul only supports CMAKE_ASC_ARCHITECTURES=dav-3510")
endif()

find_package(ASC REQUIRED)

project(tensor_api_matmul LANGUAGES ASC CXX)

set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})

set(ASC_INCLUDE_DIR "${ASCEND_HOME}/asc/")
include_directories(
    "${ASC_INCLUDE_DIR}"
)


foreach(target_name IN ITEMS
    tensor_mmad_single_core
    tensor_mmad_multi_core
    tensor_mmad_double_buffer
    tensor_mmad_practice
)
    add_executable(${target_name} ${target_name}.asc)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )
endforeach()
