cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU ARCH, e.g. dav-3510")

find_package(ASC REQUIRED)
project(transpose_unaligned_sample LANGUAGES ASC CXX)

add_executable(demo
    transpose_unaligned_simt.asc
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES} --enable-simt --cce-res-usage>
)
