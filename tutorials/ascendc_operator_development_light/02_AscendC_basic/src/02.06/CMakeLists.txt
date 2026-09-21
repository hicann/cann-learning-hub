cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture: dav-2201, dav-3510")
find_package(ASC REQUIRED)
project(add_custom_template LANGUAGES ASC CXX)

add_executable(demo
    add_custom_template.asc
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    m
    dl
)
