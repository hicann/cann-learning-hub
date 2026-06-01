cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(conv3d_backprop_filter_v2_demo_minimal LANGUAGES ASC CXX)

add_executable(minimal_demo
    minimal_demo.asc
)

target_include_directories(minimal_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/basic_api
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/basic_api/reg_compute
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/simt_api
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/tikcpp/tikcfw
)

target_link_directories(minimal_demo PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(minimal_demo PRIVATE
    tiling_api
    register
    platform
    ascendc_runtime
    runtime
    ascendcl
    error_manager
    profapi
    ge_common_base
    unified_dlog
    mmpa
    ascend_dump
    c_sec
    m
    dl
)

target_compile_options(minimal_demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)