cmake_minimum_required(VERSION 3.16.0)
project(demo_as_strided)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/home/developer/Ascend/cann-9.0.0")
endif()

# Include directories
include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/acl
    ${ASCEND_HOME_PATH}/include/aclnn
    ${CMAKE_SOURCE_DIR}/../build/autogen
)

# Link directories
link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${CMAKE_SOURCE_DIR}/../build/op_host
)

add_executable(demo_as_strided main.cpp)

target_link_libraries(demo_as_strided
    ascendcl
    nnopbase
    cust_opapi
    cust_opmaster_rt2.0
    dl
)
