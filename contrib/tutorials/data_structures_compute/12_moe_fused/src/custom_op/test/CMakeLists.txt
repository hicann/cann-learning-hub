cmake_minimum_required(VERSION 3.16)
project(moe_router_fused_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Please source the CANN set_env.sh first.")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

# 算子包构建输出目录（build_out），内含 autogen 头文件与 libcust_opapi.so
get_filename_component(CUSTOM_OP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
set(BUILD_OUT ${CUSTOM_OP_ROOT}/build_out)

include_directories(${ASCEND_HOME_PATH}/include)
include_directories(${ASCEND_HOME_PATH}/include/aclnn)
include_directories(${BUILD_OUT}/autogen)

link_directories(${ASCEND_HOME_PATH}/lib64)
link_directories(${BUILD_OUT}/op_host)

add_executable(execute_moe_router_fused main.cpp)
target_link_libraries(execute_moe_router_fused ascendcl nnopbase cust_opapi)
set_target_properties(execute_moe_router_fused PROPERTIES
    BUILD_RPATH "${BUILD_OUT}/op_host;${ASCEND_HOME_PATH}/lib64"
)
