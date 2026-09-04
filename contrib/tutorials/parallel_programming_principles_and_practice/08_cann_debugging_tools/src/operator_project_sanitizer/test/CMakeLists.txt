cmake_minimum_required(VERSION 3.5.1)
project(acl_execute_add_msanitizer)

add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")

# set_env.sh normally provides both paths; the defaults match a standard Toolkit install.
# CANN root priority: -DASCEND_ROOT cache override, then ASCEND_HOME,
# ASCEND_HOME_PATH, and DDK_PATH. A3's set_env.sh
# exports ASCEND_HOME_PATH but leaves DDK_PATH/NPU_HOST_LIB empty, so the
# ASCEND_HOME_PATH-based selection below is what makes acl/acl.h findable.
set(ASCEND_ROOT "" CACHE PATH "CANN install root (must contain include and lib64)")
if(ASCEND_ROOT STREQUAL "")
    set(ASCEND_ROOT "$ENV{ASCEND_HOME}")
endif()
if(ASCEND_ROOT STREQUAL "")
    set(ASCEND_ROOT "$ENV{ASCEND_HOME_PATH}")
endif()
if(ASCEND_ROOT STREQUAL "")
    set(ASCEND_ROOT "$ENV{DDK_PATH}")
endif()
get_filename_component(ASCEND_ROOT "${ASCEND_ROOT}" ABSOLUTE)

# Headers: accept both <root>/include and <root>/runtime/include layouts.
set(ASCEND_INCLUDE_DIR "")
if(EXISTS "${ASCEND_ROOT}/include/acl/acl.h")
    set(ASCEND_INCLUDE_DIR "${ASCEND_ROOT}/include")
elseif(EXISTS "${ASCEND_ROOT}/runtime/include/acl/acl.h")
    set(ASCEND_INCLUDE_DIR "${ASCEND_ROOT}/runtime/include")
else()
    message(FATAL_ERROR
        "CANN headers not found: neither ${ASCEND_ROOT}/include/acl/acl.h nor "
        "${ASCEND_ROOT}/runtime/include/acl/acl.h exists. Fix ASCEND_ROOT or "
        "export ASCEND_HOME_PATH/ASCEND_HOME before configuring.")
endif()

# Host libraries: prefer the CANN root lib64/runtime/lib64, then the
# NPU_HOST_LIB environment variable, then the legacy <arch>-<os>/devlib layout.
set(ASCEND_HOST_LIB "")
if(EXISTS "${ASCEND_ROOT}/lib64")
    set(ASCEND_HOST_LIB "${ASCEND_ROOT}/lib64")
elseif(EXISTS "${ASCEND_ROOT}/runtime/lib64")
    set(ASCEND_HOST_LIB "${ASCEND_ROOT}/runtime/lib64")
elseif(NOT "$ENV{NPU_HOST_LIB}" STREQUAL "")
    set(ASCEND_HOST_LIB "$ENV{NPU_HOST_LIB}")
else()
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(ASCEND_HOST_LIB
        "${ASCEND_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/devlib")
endif()
get_filename_component(ASCEND_HOST_LIB "${ASCEND_HOST_LIB}" ABSOLUTE)
if(NOT EXISTS "${ASCEND_HOST_LIB}")
    message(FATAL_ERROR
        "CANN host library directory not found: ${ASCEND_HOST_LIB}. Check "
        "NPU_HOST_LIB or the lib64 layout under ${ASCEND_ROOT}.")
endif()

# The Notebook installs the custom OPP into an isolated temporary directory.
set(LAB05_CUSTOM_OPP_ROOT "" CACHE PATH "Custom OPP install root")
if(LAB05_CUSTOM_OPP_ROOT STREQUAL "")
    message(FATAL_ERROR "Set -DLAB05_CUSTOM_OPP_ROOT=<install-dir>")
endif()
set(CUSTOM_OP_API "${LAB05_CUSTOM_OPP_ROOT}/vendors/customize/op_api")

include_directories("${ASCEND_INCLUDE_DIR}" "${CUSTOM_OP_API}/include")
link_directories("${ASCEND_HOST_LIB}" "${CUSTOM_OP_API}/lib")

add_executable(execute_add_msanitizer main.cpp)
target_link_libraries(execute_add_msanitizer
    ascendcl cust_opapi acl_op_compiler nnopbase stdc++)
install(TARGETS execute_add_msanitizer DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
