cmake_minimum_required(VERSION 3.16)

project(AscendGMRES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)

get_filename_component(COURSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(COURSE_SPARSE_ROOT "${COURSE_ROOT}/common/sparse")

add_library(ascend_gmres_core
    ${COURSE_SPARSE_ROOT}/src/csr_matrix.cpp
    src/csr_matrix.cpp
    src/gmres.cpp
    src/cpu_gmres.cpp
    src/cpu_omp_gmres.cpp
    src/npu_gmres.cpp
    spmv/csr_spmv.cpp
    spmv/npu_spmv.cpp
    blas/dot.cpp
    blas/axpy.cpp
    blas/norm.cpp
)

target_include_directories(ascend_gmres_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${COURSE_SPARSE_ROOT}/include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ascend_gmres_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ascend_gmres_core PUBLIC ASCEND_GMRES_HAS_OPENMP=1)
else()
    target_compile_definitions(ascend_gmres_core PUBLIC ASCEND_GMRES_HAS_OPENMP=0)
endif()
