cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU ARCH, e.g. dav-3510")

# find_package(ASC) 用于查找和配置 Ascend C 编译工具链
find_package(ASC REQUIRED)
# 指定项目支持的语言包括 ASC 和 CXX，ASC 表示支持使用毕昇编译器对 Ascend C 编程语言进行编译
project(transpose_ub_loop_sample LANGUAGES ASC CXX)

add_executable(demo
    simd_simt_transpose_ub_loop.asc
)

# 通过编译选项设置 NPU 架构
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)