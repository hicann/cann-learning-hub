cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(ASC REQUIRED)
project(custom_launch_sk LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
        ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import importlib.util, pathlib; spec = importlib.util.find_spec('torch_npu'); print(pathlib.Path(spec.origin).parent)"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(add_custom_kernels SHARED
    add_custom_kernel.asc
)

add_library(custom_ops_lib SHARED
    add_custom_pytorch.cpp
)

target_include_directories(custom_ops_lib SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}/include
)

target_link_directories(custom_ops_lib PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

target_link_libraries(custom_ops_lib PRIVATE
    add_custom_kernels
    torch_npu
    Python3::Python
)

target_compile_options(add_custom_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(custom_ops_lib PRIVATE
    ${TORCH_CXX_FLAGS}
)

set_target_properties(custom_ops_lib PROPERTIES
    PREFIX ""
    BUILD_RPATH "$ORIGIN"
)
