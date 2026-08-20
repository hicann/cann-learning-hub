cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(vector_add_student LANGUAGES ASC CXX)

set(NPU_ARCH dav-2201 CACHE STRING "Ascend C target architecture")

add_executable(vector_add_student vector_add_student.asc)
target_include_directories(vector_add_student PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vector_add_student PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)
