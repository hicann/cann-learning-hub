cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(ASC REQUIRED)
project(kernel_pytorch_call_mmad_answer LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(select_probe_line output_var raw_output suffix)
    string(REPLACE "\n" ";" output_lines "${raw_output}")
    foreach(output_line IN LISTS output_lines)
        string(STRIP "${output_line}" output_line)
        if(output_line MATCHES "${suffix}$")
            set(${output_var} "${output_line}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Cannot find path ending with ${suffix} in Python probe output: ${raw_output}")
endfunction()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
select_probe_line(TORCH_CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH_RAW}" "/torch/share/cmake")
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
select_probe_line(TORCH_NPU_PATH "${TORCH_NPU_PATH_RAW}" "/torch_npu")
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(practice_matmul_kernels SHARED
    matmul_custom.asc
)

add_library(practice_matmul_ops SHARED
    torch_binding.cpp
)

target_include_directories(practice_matmul_ops SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
)

target_link_directories(practice_matmul_ops PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

target_link_libraries(practice_matmul_ops PRIVATE
    practice_matmul_kernels
    torch_npu
    Python3::Python
)

target_compile_options(practice_matmul_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(practice_matmul_ops PRIVATE
    ${TORCH_CXX_FLAGS}
)

set_target_properties(practice_matmul_ops PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
