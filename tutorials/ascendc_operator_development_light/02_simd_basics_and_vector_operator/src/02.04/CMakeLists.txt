cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(static_tensor_vector LANGUAGES ASC CXX)

# 默认编译 Add；习题和答案通过 -DKERNEL_SOURCE=/absolute/path/sub_static.asc 指定。
set(KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/add_static.asc" CACHE FILEPATH "Kernel source")
add_executable(demo "${KERNEL_SOURCE}")
target_compile_options(demo PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
