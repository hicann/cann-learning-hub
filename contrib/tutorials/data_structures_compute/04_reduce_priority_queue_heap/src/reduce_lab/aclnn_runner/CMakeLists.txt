cmake_minimum_required(VERSION 3.16)
project(reduce_lite_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
else()
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH})
  set(CUSTOM_OPP_PATH $ENV{ASCEND_CUSTOM_OPP_PATH})
else()
  set(CUSTOM_OPP_PATH ${ASCEND_HOME_PATH}/opp/vendors/customize)
endif()

if(NOT EXISTS ${CUSTOM_OPP_PATH}/op_api/include AND EXISTS ${CUSTOM_OPP_PATH}/vendors/customize/op_api/include)
  set(CUSTOM_OPP_PATH ${CUSTOM_OPP_PATH}/vendors/customize)
endif()

include_directories(${ASCEND_HOME_PATH}/include)
include_directories(${ASCEND_HOME_PATH}/include/aclnn)
include_directories(${ASCEND_HOME_PATH}/include/aclnnop)
include_directories(${CUSTOM_OPP_PATH}/op_api/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/build_out/autogen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/ReduceLiteOps/build_out/autogen)

link_directories(${ASCEND_HOME_PATH}/lib64)
link_directories(${CUSTOM_OPP_PATH}/op_api/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/build_out/autogen)

add_executable(main_reduce_benchmark main_reduce_benchmark.cpp)
target_link_libraries(main_reduce_benchmark ascendcl nnopbase cust_opapi ascend_all_ops)
set_target_properties(main_reduce_benchmark PROPERTIES
  BUILD_RPATH "${CUSTOM_OPP_PATH}/op_api/lib;${ASCEND_HOME_PATH}/lib64;${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/build_out/autogen"
)

message(STATUS "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
message(STATUS "CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}")
