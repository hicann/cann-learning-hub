cmake_minimum_required(VERSION 3.14)
project(ge_zero_copy_device LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Please source the CANN set_env.sh first.")
endif()
message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")

add_executable(ge_zero_copy_device main.cc)
target_compile_options(ge_zero_copy_device PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(ge_zero_copy_device PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(ge_zero_copy_device PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/graph"
  "${ASCEND_HOME_PATH}/include/ge"
  "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
)
target_link_directories(ge_zero_copy_device PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(ge_zero_copy_device PRIVATE
  -Wl,--no-as-needed
  ge_runner
  ge_compiler
  graph
  graph_base
  ascendcl
  acl_rt
  -Wl,--as-needed
)
