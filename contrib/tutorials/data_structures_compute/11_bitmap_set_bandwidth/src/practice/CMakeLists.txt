cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(bitmap_and_student LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(NPU_ARCH dav-2201 CACHE STRING "Ascend C target architecture")

add_executable(bitmap_and_student ../demo/bitmap_and.asc)
target_include_directories(bitmap_and_student PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bitmap_and_student PRIVATE STUDENT_MODE=1)
target_compile_options(bitmap_and_student PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)
