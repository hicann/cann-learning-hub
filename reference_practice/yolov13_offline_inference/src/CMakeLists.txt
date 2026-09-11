# Copyright (c) Huawei Technologies Co., Ltd. 2015. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(ACL_YOLOV13)

add_compile_options(-std=c++17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{ASCEND_HOME_PATH})

if (NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Please source the CANN set_env.sh before building.")
else ()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH "${INC_PATH}/devlib")
message(STATUS "set LIB_PATH: ${LIB_PATH}")

include_directories(
    ${INC_PATH}/include/
    ../inc/
)

link_directories(
    ${LIB_PATH}
)

add_executable(yolov13_main
        common/utils.cpp
        common/sampleModel.cpp
        sample_yolov13.cpp)

target_link_libraries(yolov13_main acl_rt acl_mdl stdc++)

install(TARGETS yolov13_main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
