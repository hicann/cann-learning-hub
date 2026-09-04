cmake_minimum_required(VERSION 3.16)
project(DisGMRES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(DIS_GMRES_FORCE_STUB "Build without ACL/HCCL even when CANN is installed" OFF)
option(DIS_GMRES_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(ASCEND_HOME "$ENV{ASCEND_HOME}" CACHE PATH "Ascend toolkit root")
if(NOT ASCEND_HOME)
  set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
endif()

find_package(OpenMP)
find_path(ACL_INCLUDE_DIR acl/acl.h HINTS "${ASCEND_HOME}/include" "${ASCEND_HOME}/runtime/include")
find_path(HCCL_INCLUDE_DIR hccl/hccl.h HINTS "${ASCEND_HOME}/include" "${ASCEND_HOME}/runtime/include")
find_library(ASCENDCL_LIBRARY NAMES ascendcl HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/runtime/lib64")
find_library(HCCL_LIBRARY NAMES hccl HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/runtime/lib64")
find_library(ACL_RTC_LIBRARY NAMES acl_rtc HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/runtime/lib64")

if(NOT DIS_GMRES_FORCE_STUB AND ACL_INCLUDE_DIR AND HCCL_INCLUDE_DIR AND ASCENDCL_LIBRARY AND HCCL_LIBRARY AND ACL_RTC_LIBRARY)
  set(DIS_GMRES_HAS_CANN 1)
else()
  set(DIS_GMRES_HAS_CANN 0)
endif()

if(NOT DIS_GMRES_FORCE_STUB AND NOT DIS_GMRES_HAS_CANN)
  message(FATAL_ERROR "Real distributed GMRES requires ACL, HCCL and acl_rtc; use DIS_GMRES_FORCE_STUB=ON only for Host unit tests")
endif()

message(STATUS "ASCEND_HOME=${ASCEND_HOME}")
message(STATUS "Dis-GMRES ACL/HCCL backend=${DIS_GMRES_HAS_CANN} (1=real, 0=host stub)")
set(DIS_GMRES_HAS_CANN "${DIS_GMRES_HAS_CANN}" CACHE INTERNAL "Resolved real ACL/HCCL backend")

set(COURSE_SPARSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/sparse")

add_library(dis_gmres_core
  ${COURSE_SPARSE_ROOT}/src/csr_matrix.cpp
  src/csr_matrix.cpp
  src/spmv.cpp
  src/hccl_comm.cpp
  src/gmres.cpp
  src/npu_compute.cpp
  src/npu_gmres.cpp
)
target_include_directories(dis_gmres_core PUBLIC
  include
  ${COURSE_SPARSE_ROOT}/include
)
target_compile_definitions(dis_gmres_core PUBLIC DIS_GMRES_HAS_CANN=${DIS_GMRES_HAS_CANN})

if(OpenMP_CXX_FOUND)
  target_link_libraries(dis_gmres_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(dis_gmres_core PUBLIC DIS_GMRES_HAS_OPENMP=1)
else()
  target_compile_definitions(dis_gmres_core PUBLIC DIS_GMRES_HAS_OPENMP=0)
endif()

if(DIS_GMRES_HAS_CANN)
  target_include_directories(dis_gmres_core SYSTEM PUBLIC "${ACL_INCLUDE_DIR}" "${HCCL_INCLUDE_DIR}")
  target_link_libraries(dis_gmres_core PUBLIC "${ASCENDCL_LIBRARY}" "${HCCL_LIBRARY}" "${ACL_RTC_LIBRARY}")
  target_compile_definitions(dis_gmres_core PRIVATE
    DIS_GMRES_KERNEL_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/kernels/gmres_ops.cpp")
  set_target_properties(dis_gmres_core PROPERTIES BUILD_RPATH "${ASCEND_HOME}/lib64;${ASCEND_HOME}/runtime/lib64")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dis_gmres_core PRIVATE -Wall -Wextra -Wpedantic)
  if(DIS_GMRES_WARNINGS_AS_ERRORS)
    target_compile_options(dis_gmres_core PRIVATE -Werror)
  endif()
endif()

add_executable(dis_gmres src/main.cpp)
target_link_libraries(dis_gmres PRIVATE dis_gmres_core)

add_executable(matrix_generator tools/matrix_generator.cpp)
target_link_libraries(matrix_generator PRIVATE dis_gmres_core)

enable_testing()
add_executable(dis_gmres_tests tests/test_core.cpp)
target_link_libraries(dis_gmres_tests PRIVATE dis_gmres_core)
add_test(NAME dis_gmres_core_tests COMMAND dis_gmres_tests)
