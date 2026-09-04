cmake_minimum_required(VERSION 3.16)

project(AscendXyce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)
option(ASCEND_XYCE_HOST_ONLY "Build CPU baselines without the required NPU solver" OFF)
set(ASCEND_HOME "$ENV{ASCEND_HOME}" CACHE PATH "Ascend toolkit root")
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
endif()
find_path(ACL_INCLUDE_DIR acl/acl.h HINTS "${ASCEND_HOME}/include" "${ASCEND_HOME}/runtime/include")
find_path(HCCL_INCLUDE_DIR hccl/hccl.h HINTS "${ASCEND_HOME}/include" "${ASCEND_HOME}/runtime/include")
find_library(ASCENDCL_LIBRARY NAMES ascendcl HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/runtime/lib64")
find_library(HCCL_LIBRARY NAMES hccl HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/runtime/lib64")
find_library(ACL_RTC_LIBRARY NAMES acl_rtc HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/runtime/lib64")
if(ASCEND_XYCE_HOST_ONLY)
    set(ASCEND_XYCE_HAS_CANN 0)
elseif(ACL_INCLUDE_DIR AND HCCL_INCLUDE_DIR AND ASCENDCL_LIBRARY AND HCCL_LIBRARY AND ACL_RTC_LIBRARY)
    set(ASCEND_XYCE_HAS_CANN 1)
else()
    message(FATAL_ERROR "Ascend-Xyce requires ACL, HCCL and acl_rtc; ASCEND_XYCE_HOST_ONLY=ON is for CPU diagnostics only")
endif()

get_filename_component(COURSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(DIS_GMRES_ROOT "${COURSE_ROOT}/09_cann_application_tuning/01_distributed_gmres/src/dis_gmres")
set(COURSE_SPARSE_ROOT "${COURSE_ROOT}/common/sparse")
add_library(xyce_npu_solver
    ${COURSE_SPARSE_ROOT}/src/csr_matrix.cpp
    ${DIS_GMRES_ROOT}/src/csr_matrix.cpp ${DIS_GMRES_ROOT}/src/spmv.cpp
    ${DIS_GMRES_ROOT}/src/hccl_comm.cpp ${DIS_GMRES_ROOT}/src/gmres.cpp
    ${DIS_GMRES_ROOT}/src/npu_compute.cpp ${DIS_GMRES_ROOT}/src/npu_gmres.cpp)
target_include_directories(xyce_npu_solver PUBLIC
    "${DIS_GMRES_ROOT}"
    "${DIS_GMRES_ROOT}/include"
    "${COURSE_SPARSE_ROOT}/include")
target_compile_definitions(xyce_npu_solver PUBLIC DIS_GMRES_HAS_CANN=${ASCEND_XYCE_HAS_CANN})
if(ASCEND_XYCE_HAS_CANN)
    target_include_directories(xyce_npu_solver PUBLIC "${ACL_INCLUDE_DIR}" "${HCCL_INCLUDE_DIR}")
    target_link_libraries(xyce_npu_solver PUBLIC "${ASCENDCL_LIBRARY}" "${HCCL_LIBRARY}" "${ACL_RTC_LIBRARY}")
    target_compile_definitions(xyce_npu_solver PRIVATE DIS_GMRES_KERNEL_SOURCE="${DIS_GMRES_ROOT}/kernels/gmres_ops.cpp")
endif()

if(NOT DEFINED ASCEND_GMRES_DIR)
    if(DEFINED ENV{ASCEND_GMRES_DIR})
        set(ASCEND_GMRES_DIR "$ENV{ASCEND_GMRES_DIR}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backend/gmres/Ascend-GMRES/CMakeLists.txt")
        set(ASCEND_GMRES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backend/gmres/Ascend-GMRES")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Ascend-GMRES/CMakeLists.txt")
        set(ASCEND_GMRES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Ascend-GMRES")
    else()
        message(FATAL_ERROR "ASCEND_GMRES_DIR is not set and Ascend-GMRES was not found")
    endif()
endif()

message(STATUS "Using Ascend-GMRES backend: ${ASCEND_GMRES_DIR}")

set(ASCEND_GMRES_SOURCES
    ${ASCEND_GMRES_DIR}/src/csr_matrix.cpp
    ${ASCEND_GMRES_DIR}/src/gmres.cpp
    ${ASCEND_GMRES_DIR}/src/cpu_gmres.cpp
    ${ASCEND_GMRES_DIR}/src/cpu_omp_gmres.cpp
    ${ASCEND_GMRES_DIR}/src/npu_gmres.cpp
    ${ASCEND_GMRES_DIR}/spmv/csr_spmv.cpp
    ${ASCEND_GMRES_DIR}/spmv/npu_spmv.cpp
    ${ASCEND_GMRES_DIR}/blas/dot.cpp
    ${ASCEND_GMRES_DIR}/blas/axpy.cpp
    ${ASCEND_GMRES_DIR}/blas/norm.cpp
)

add_library(ascend_xyce_core
    src/xyce_adapter.cpp
    ${ASCEND_GMRES_SOURCES}
)

target_include_directories(ascend_xyce_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASCEND_GMRES_DIR}/include
    ${DIS_GMRES_ROOT}
    ${COURSE_SPARSE_ROOT}/include
)
target_link_libraries(ascend_xyce_core PUBLIC xyce_npu_solver)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ascend_xyce_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ascend_xyce_core PUBLIC ASCEND_GMRES_HAS_OPENMP=1)
else()
    target_compile_definitions(ascend_xyce_core PUBLIC ASCEND_GMRES_HAS_OPENMP=0)
endif()

add_executable(xyce_benchmark benchmark/xyce_benchmark.cpp)
target_link_libraries(xyce_benchmark PRIVATE ascend_xyce_core)
