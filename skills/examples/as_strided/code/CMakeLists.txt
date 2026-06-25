cmake_minimum_required(VERSION 3.16.0)

project(custom_project)

find_package(ASC REQUIRED)

set(ASCEND_COMPUTE_UNIT ascend910b)
set(package_name custom)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)