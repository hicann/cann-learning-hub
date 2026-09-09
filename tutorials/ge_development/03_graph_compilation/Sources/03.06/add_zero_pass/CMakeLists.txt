cmake_minimum_required(VERSION 3.14)
project(notebook_add_zero_pass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Please source the CANN set_env.sh first.")
endif()
message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")

# CANN 9.0 通过该 CMake 模块为 Pattern 和 Replacement 生成 ES 构图 API。
set(GENERATE_ES_PACKAGE_DIR "${ASCEND_HOME_PATH}/include/ge/cmake")
if(NOT EXISTS "${GENERATE_ES_PACKAGE_DIR}/generate_es_package.cmake")
  message(FATAL_ERROR "CANN ES API generator not found: ${GENERATE_ES_PACKAGE_DIR}/generate_es_package.cmake")
endif()
list(APPEND CMAKE_MODULE_PATH "${GENERATE_ES_PACKAGE_DIR}")
find_package(GenerateEsPackage REQUIRED)

set(ES_OUTPUT_PATH "${CMAKE_BINARY_DIR}/es_output")
add_library(opgraph_all INTERFACE)
set_target_properties(opgraph_all PROPERTIES
  INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${ASCEND_HOME_PATH}/opp/built-in/op_proto"
)
add_es_library(
  ES_LINKABLE_AND_ALL_TARGET es_all
  OPP_PROTO_TARGET opgraph_all
  OUTPUT_PATH "${ES_OUTPUT_PATH}"
)

add_library(notebook_add_zero_pass SHARED src/add_zero_pass.cpp)
target_compile_options(notebook_add_zero_pass PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(notebook_add_zero_pass PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(notebook_add_zero_pass PRIVATE
  "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/graph"
  "${ASCEND_HOME_PATH}/include/ge"
  "${ASCEND_HOME_PATH}/include/register"
  "${ASCEND_HOME_PATH}/include/external"
  "${ES_OUTPUT_PATH}/include/es_all"
)
target_link_directories(notebook_add_zero_pass PRIVATE
  "${ASCEND_HOME_PATH}/lib64/stub"
  "${ASCEND_HOME_PATH}/lib64"
)
target_link_libraries(notebook_add_zero_pass PRIVATE
  -Wl,--no-as-needed
  graph
  register
  ge_compiler
  es_all
  -Wl,--as-needed
)

# Pass 和生成的 libes_all.so 放在同一目录，无需安装到 CANN 目录。
set_target_properties(notebook_add_zero_pass PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${ES_OUTPUT_PATH}/lib64"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
