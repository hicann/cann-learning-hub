cmake_minimum_required(VERSION 3.16)
project(ge_three_operator_samples LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver" CACHE PATH "Ascend driver path")

function(configure_ge_demo target)
    target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CANN_HOME}/include
        ${CANN_HOME}/include/graph
        ${CANN_HOME}/opp/built-in/op_graph/inc
        ${CANN_HOME}/opp/built-in/op_proto/inc
    )
    target_link_directories(${target} PRIVATE
        ${CANN_HOME}/lib64
        ${ASCEND_DRIVER_PATH}/lib64
        ${ASCEND_DRIVER_PATH}/lib64/common
        ${ASCEND_DRIVER_PATH}/lib64/driver
    )
    target_link_libraries(${target} PRIVATE graph ge_runner graph_base)
    target_link_options(${target} PRIVATE
        -Wl,-rpath-link,${ASCEND_DRIVER_PATH}/lib64
        -Wl,-rpath-link,${ASCEND_DRIVER_PATH}/lib64/common
        -Wl,-rpath-link,${ASCEND_DRIVER_PATH}/lib64/driver)
endfunction()
