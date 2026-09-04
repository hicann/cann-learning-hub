cmake_minimum_required(VERSION 3.16)

project(SpMV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)
option(SPMV_REAL_ASCENDC "Build the real ACL RTC Ascend C backend" ON)
get_filename_component(COURSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(COURSE_SPARSE_ROOT "${COURSE_ROOT}/common/sparse")

add_library(spmv_core
    ${COURSE_SPARSE_ROOT}/src/csr_matrix.cpp
    ${COURSE_SPARSE_ROOT}/src/cpu_backend.cpp
    src/cpu_omp_spmv.cpp
    src/npu_spmv.cpp
    src/npu_spmv_fp16.cpp
    src/npu_spmv_bf16.cpp
    src/npu_spmv_context.cpp
    src/npu_spmv_bf16_persistent.cpp
)

if(SPMV_REAL_ASCENDC)
    set(ASCEND_HOME "$ENV{ASCEND_HOME}" CACHE PATH "Ascend Toolkit prefix")
    if(NOT ASCEND_HOME)
        set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
    endif()
    find_path(ACL_INCLUDE_DIR acl/acl.h HINTS "${ASCEND_HOME}/include")
    find_library(ASCENDCL_LIBRARY ascendcl HINTS "${ASCEND_HOME}/lib64")
    find_library(ACL_RTC_LIBRARY acl_rtc HINTS "${ASCEND_HOME}/lib64")
    if(NOT ACL_INCLUDE_DIR OR NOT ASCENDCL_LIBRARY OR NOT ACL_RTC_LIBRARY)
        message(FATAL_ERROR "Real Ascend C backend requires CANN ACL and acl_rtc. Source set_env.sh, or explicitly use -DSPMV_REAL_ASCENDC=OFF for Host-prototype diagnostics.")
    endif()
    target_sources(spmv_core PRIVATE src/ascendc_spmv.cpp)
    target_include_directories(spmv_core PUBLIC "${ACL_INCLUDE_DIR}")
    target_link_libraries(spmv_core PUBLIC "${ASCENDCL_LIBRARY}" "${ACL_RTC_LIBRARY}")
    target_compile_definitions(spmv_core PUBLIC SPMV_REAL_ASCENDC=1
        ASCENDC_SPMV_KERNEL_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/kernels/spmv_fp32_kernel.cpp")
else()
    target_compile_definitions(spmv_core PUBLIC SPMV_REAL_ASCENDC=0)
endif()

target_include_directories(spmv_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${COURSE_SPARSE_ROOT}/include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(spmv_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(spmv_core PUBLIC SPMV_HAS_OPENMP=1)
else()
    target_compile_definitions(spmv_core PUBLIC SPMV_HAS_OPENMP=0)
endif()

add_executable(spmv_benchmark benchmark/spmv_benchmark.cpp)
target_link_libraries(spmv_benchmark PRIVATE spmv_core)

add_executable(uniform_generator ${COURSE_SPARSE_ROOT}/tools/matrix_generator.cpp)
target_compile_definitions(uniform_generator PRIVATE COURSE_SPARSE_GENERATOR_KIND=0)
target_link_libraries(uniform_generator PRIVATE spmv_core)

add_executable(longtail_generator ${COURSE_SPARSE_ROOT}/tools/matrix_generator.cpp)
target_compile_definitions(longtail_generator PRIVATE COURSE_SPARSE_GENERATOR_KIND=1)
target_link_libraries(longtail_generator PRIVATE spmv_core)

add_executable(block_generator ${COURSE_SPARSE_ROOT}/tools/matrix_generator.cpp)
target_compile_definitions(block_generator PRIVATE COURSE_SPARSE_GENERATOR_KIND=2)
target_link_libraries(block_generator PRIVATE spmv_core)
