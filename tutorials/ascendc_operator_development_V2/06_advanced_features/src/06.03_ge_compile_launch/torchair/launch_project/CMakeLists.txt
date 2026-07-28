cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture, such as dav-3510 or dav-2201")
find_package(ASC REQUIRED)
project(ge_torchair_launch LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")

add_library(ge_torchair_launch_ops SHARED
    launch_add_pytorch.cpp
    launch_add_custom.asc
)
set_target_properties(ge_torchair_launch_ops PROPERTIES LINKER_LANGUAGE ASC)
target_compile_options(ge_torchair_launch_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS_LIST}>
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_include_directories(ge_torchair_launch_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/graph
    ${ASCEND_HOME_PATH}/include/register
    ${ASCEND_HOME_PATH}/include/external
)
target_link_directories(ge_torchair_launch_ops PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(ge_torchair_launch_ops PRIVATE
    -Wl,--no-as-needed
    Python3::Python
    gert
    graph
    ${TORCH_LIBRARIES}
    -Wl,--as-needed
)
