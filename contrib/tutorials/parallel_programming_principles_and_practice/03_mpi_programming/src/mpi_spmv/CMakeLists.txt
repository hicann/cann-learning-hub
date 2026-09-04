cmake_minimum_required(VERSION 3.16)

project(mpi_spmv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(MPI REQUIRED COMPONENTS CXX)

get_filename_component(COURSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(COURSE_SPARSE_ROOT "${COURSE_ROOT}/common/sparse")

add_library(spmv_core
    src/benchmark.cpp
    src/spmv_cpu.cpp
    ${COURSE_SPARSE_ROOT}/src/csr_matrix.cpp
)
target_include_directories(spmv_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${COURSE_SPARSE_ROOT}/include")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(spmv_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

add_executable(spmv_cpu src/cpu_main.cpp)
target_link_libraries(spmv_cpu PRIVATE spmv_core)

add_executable(spmv_mpi src/spmv_mpi.cpp)
target_link_libraries(spmv_mpi PRIVATE spmv_core MPI::MPI_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(spmv_mpi PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
