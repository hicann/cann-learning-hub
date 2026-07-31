cmake_minimum_required(VERSION 3.16)
project(offline_add_custom LANGUAGES CXX)

# The offline sample deliberately uses the CANN 9.0 V1 operator package
# (OpDef/tiling/kernel).  CompilableOp/PortableOp are not required here.
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH must point to the CANN 9.0.0 toolkit")
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" host_arch)
if(host_arch MATCHES "^(aarch64|arm64)$")
  set(cann_arch_dir aarch64-linux)
else()
  set(cann_arch_dir x86_64-linux)
endif()

# 根据 SOC_VERSION 动态推导 ASCEND_COMPUTE_UNIT，使算子包的 ops-info/kernel
# 目录与目标芯片的 SHORT_SOC_VERSION 对齐（ATC 按 SHORT_SOC_VERSION 查找算子库）。
# 必须在 find_package(ASC) 之前设置，否则 ASC 包会以 ascend910b 作为默认 cache 值。
# 也可通过 CMake 变量 ASCEND_COMPUTE_UNIT 显式覆盖。
if(NOT DEFINED ASCEND_COMPUTE_UNIT)
  set(_soc_version "$ENV{SOC_VERSION}")
  string(TOLOWER "${_soc_version}" _soc_version_lower)
  if(_soc_version_lower MATCHES "^ascend910_93")
    set(ASCEND_COMPUTE_UNIT ascend910_93)
  elseif(_soc_version_lower MATCHES "^ascend910b")
    set(ASCEND_COMPUTE_UNIT ascend910b)
  elseif(_soc_version_lower MATCHES "^ascend910")
    set(ASCEND_COMPUTE_UNIT ascend910b)
  elseif(_soc_version_lower MATCHES "^ascend310b")
    set(ASCEND_COMPUTE_UNIT ascend310b)
  elseif(_soc_version_lower MATCHES "^ascend310p")
    set(ASCEND_COMPUTE_UNIT ascend310p)
  elseif(_soc_version_lower)
    set(ASCEND_COMPUTE_UNIT "${_soc_version_lower}")
  else()
    set(ASCEND_COMPUTE_UNIT ascend910b)
    message(WARNING "SOC_VERSION is empty; defaulting ASCEND_COMPUTE_UNIT to ascend910b. "
      "Set SOC_VERSION env var or ASCEND_COMPUTE_UNIT CMake variable to override.")
  endif()
endif()
message(STATUS "ASCEND_COMPUTE_UNIT: ${ASCEND_COMPUTE_UNIT} (SOC_VERSION=$ENV{SOC_VERSION})")

find_package(ASC REQUIRED HINTS
  "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
  "${ASCEND_HOME_PATH}/${cann_arch_dir}/tikcpp/ascendc_kernel_cmake"
)

set(package_name customize)

npu_op_package(${package_name}
  TYPE RUN
  CONFIG
    INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)

set(COMMON_COMPILE_OPTIONS -Wall -Wextra -Wno-unused-parameter)
set(COMMON_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)

add_executable(offline_add_graph_build graph_build/main.cc)
target_compile_features(offline_add_graph_build PRIVATE cxx_std_17)
target_compile_options(offline_add_graph_build PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_definitions(offline_add_graph_build PRIVATE ${COMMON_DEFINITIONS})
target_include_directories(offline_add_graph_build PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ge
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/graph
  ${ASCEND_HOME_PATH}/include/ge
  ${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc
)
target_link_directories(offline_add_graph_build PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(offline_add_graph_build PRIVATE graph ge_compiler ascendcl graph_base)
set_target_properties(offline_add_graph_build PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(offline_add_model_exec model_exec/main.cc)
target_compile_features(offline_add_model_exec PRIVATE cxx_std_17)
target_compile_options(offline_add_model_exec PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_definitions(offline_add_model_exec PRIVATE ${COMMON_DEFINITIONS})
target_include_directories(offline_add_model_exec PRIVATE ${ASCEND_HOME_PATH}/include)
target_link_directories(offline_add_model_exec PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(offline_add_model_exec PRIVATE ascendcl)
set_target_properties(offline_add_model_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
