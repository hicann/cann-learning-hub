cmake_minimum_required(VERSION 3.16)
project(tree_queue_lite_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
else()
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(LOCAL_CUSTOM_OPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/local_opp/vendors/customize")
if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH} AND EXISTS "$ENV{ASCEND_CUSTOM_OPP_PATH}/op_api/include")
  set(CUSTOM_OPP_PATH "$ENV{ASCEND_CUSTOM_OPP_PATH}")
else()
  set(CUSTOM_OPP_PATH "${LOCAL_CUSTOM_OPP_PATH}")
endif()

if(NOT EXISTS "${CUSTOM_OPP_PATH}/op_api/include" OR NOT EXISTS "${CUSTOM_OPP_PATH}/op_api/lib")
  message(FATAL_ERROR "Local custom OPP is missing. Run ../scripts/build_ops.sh before building the runner.")
endif()

include_directories(${ASCEND_HOME_PATH}/include)
include_directories(${ASCEND_HOME_PATH}/include/aclnn)
include_directories(${ASCEND_HOME_PATH}/include/aclnnop)
include_directories(${CUSTOM_OPP_PATH}/op_api/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/TreeQueueLiteOps/build_out/autogen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/TreeQueueLiteOps/TreeQueueLiteOps/build_out/autogen)

link_directories(${ASCEND_HOME_PATH}/lib64)
link_directories(${CUSTOM_OPP_PATH}/op_api/lib)

add_executable(main_tree_queue_benchmark main_tree_queue_benchmark.cpp)
target_link_libraries(main_tree_queue_benchmark ascendcl nnopbase cust_opapi)
set_target_properties(main_tree_queue_benchmark PROPERTIES
  BUILD_RPATH "${CUSTOM_OPP_PATH}/op_api/lib;${ASCEND_HOME_PATH}/lib64"
)

message(STATUS "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
message(STATUS "CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}")
