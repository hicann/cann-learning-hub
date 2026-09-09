cmake_minimum_required(VERSION 3.16)
project(reduce_lite_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REDUCE_LAB_BLOCK_DIM 20)
if(DEFINED ENV{REDUCE_LAB_BLOCK_DIM})
  set(REDUCE_LAB_BLOCK_DIM "$ENV{REDUCE_LAB_BLOCK_DIM}")
endif()
if(NOT REDUCE_LAB_BLOCK_DIM MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "REDUCE_LAB_BLOCK_DIM must be a positive integer")
endif()

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/build_out/autogen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/ReduceLiteOps/build_out/autogen)

link_directories(${ASCEND_HOME_PATH}/lib64)
link_directories(${CUSTOM_OPP_PATH}/op_api/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/build_out/autogen)

add_executable(main_reduce_benchmark main_reduce_benchmark.cpp)
target_compile_definitions(main_reduce_benchmark PRIVATE REDUCE_LAB_BLOCK_DIM=${REDUCE_LAB_BLOCK_DIM})
target_link_libraries(main_reduce_benchmark ascendcl nnopbase cust_opapi ascend_all_ops)
set_target_properties(main_reduce_benchmark PROPERTIES
  BUILD_RPATH "${CUSTOM_OPP_PATH}/op_api/lib;${ASCEND_HOME_PATH}/lib64;${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/ReduceLiteOps/build_out/autogen"
)

message(STATUS "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
message(STATUS "CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}")
message(STATUS "REDUCE_LAB_BLOCK_DIM=${REDUCE_LAB_BLOCK_DIM}")
