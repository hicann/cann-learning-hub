cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(bitmap_and_demo LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

# dav-2201 covers the Atlas A2/910B course environment. Override it only
# after recording the actual target, for example: -DNPU_ARCH=<target>.
set(NPU_ARCH dav-2201 CACHE STRING "Ascend C target architecture")

add_executable(bitmap_and bitmap_and.asc)
target_compile_options(bitmap_and PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)
