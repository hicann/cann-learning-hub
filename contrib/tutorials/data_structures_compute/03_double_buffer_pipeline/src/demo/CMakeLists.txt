cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(vector_add_pipeline_demo LANGUAGES ASC CXX)

set(NPU_ARCH dav-2201 CACHE STRING "Ascend C target architecture")

function(add_pipeline_demo target_name buffer_num)
    add_executable(${target_name} vector_add_pipeline.asc)
    target_compile_definitions(${target_name} PRIVATE PIPELINE_BUFFER_NUM=${buffer_num})
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    )
endfunction()

add_pipeline_demo(vector_add_single_buffer 1)
add_pipeline_demo(vector_add_double_buffer 2)
