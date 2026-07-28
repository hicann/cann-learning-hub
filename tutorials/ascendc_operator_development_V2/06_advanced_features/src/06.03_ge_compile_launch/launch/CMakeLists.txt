cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture, such as dav-3510 or dav-2201")
find_package(ASC REQUIRED)
project(ge_launch_add LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
set(OPP_ROOT "${CMAKE_BINARY_DIR}/output")
set(OPP_LIB "${OPP_ROOT}/op_graph/lib/linux/${CMAKE_SYSTEM_PROCESSOR}")
file(MAKE_DIRECTORY "${OPP_LIB}")

add_library(launch_add_custom SHARED launch_add_custom.asc)
target_compile_options(launch_add_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_compile_definitions(launch_add_custom PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(launch_add_custom PRIVATE ${CANN_HOME}/include)
target_link_directories(launch_add_custom PRIVATE ${CANN_HOME}/lib64)
target_link_libraries(launch_add_custom PRIVATE gert graph)
set_target_properties(launch_add_custom PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OPP_LIB}")

add_executable(launch_add launch_add.cpp)
target_compile_definitions(launch_add PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(launch_add PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANN_HOME}/include
    ${CANN_HOME}/include/graph
    ${CANN_HOME}/opp/built-in/op_graph/inc
    ${CANN_HOME}/opp/built-in/op_proto/inc)
target_link_directories(launch_add PRIVATE ${CANN_HOME}/lib64)
target_link_libraries(launch_add PRIVATE graph ge_runner graph_base)
