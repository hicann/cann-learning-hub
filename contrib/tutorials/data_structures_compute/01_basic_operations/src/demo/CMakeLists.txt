cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(vector_add_demo LANGUAGES ASC CXX)

# dav-2201 is the A2/910B3 course baseline; override only after recording
# the actual CANNLab hardware, for example: -DNPU_ARCH=<target>.
set(NPU_ARCH dav-2201 CACHE STRING "Ascend C target architecture")

add_executable(vector_add vector_add.asc)
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)
