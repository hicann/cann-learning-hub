cmake_minimum_required(VERSION 3.16)
project(moe_lite_runner CXX)

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

# If the package was accidentally installed as .../vendors/customize/vendors/customize,
# accept that layout as well, so the runner can still be built.
if(NOT EXISTS ${CUSTOM_OPP_PATH}/op_api/include AND EXISTS ${CUSTOM_OPP_PATH}/vendors/customize/op_api/include)
  set(CUSTOM_OPP_PATH ${CUSTOM_OPP_PATH}/vendors/customize)
endif()

include_directories(${ASCEND_HOME_PATH}/include)
include_directories(${ASCEND_HOME_PATH}/include/aclnn)
include_directories(${ASCEND_HOME_PATH}/include/aclnnop)
include_directories(${CUSTOM_OPP_PATH}/op_api/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
# Useful before install, when headers exist in the combined generated project.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/MoELiteOps/build_out/autogen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../custom_ops/generated/MoELiteOps/MoELiteOps/build_out/autogen)

link_directories(${ASCEND_HOME_PATH}/lib64)
link_directories(${CUSTOM_OPP_PATH}/op_api/lib)

add_executable(main_benchmark main_benchmark.cpp)
target_link_libraries(main_benchmark ascendcl nnopbase cust_opapi)
set_target_properties(main_benchmark PROPERTIES
  BUILD_RPATH "${CUSTOM_OPP_PATH}/op_api/lib;${ASCEND_HOME_PATH}/lib64"
)

add_executable(main_full_pipeline_benchmark main_full_pipeline_benchmark.cpp)
target_link_libraries(main_full_pipeline_benchmark ascendcl nnopbase cust_opapi)
set_target_properties(main_full_pipeline_benchmark PROPERTIES
  BUILD_RPATH "${CUSTOM_OPP_PATH}/op_api/lib;${ASCEND_HOME_PATH}/lib64"
)

message(STATUS "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
message(STATUS "CUSTOM_OPP_PATH=${CUSTOM_OPP_PATH}")
