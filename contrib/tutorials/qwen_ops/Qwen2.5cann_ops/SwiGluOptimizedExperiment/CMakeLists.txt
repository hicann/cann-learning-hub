cmake_minimum_required(VERSION 3.16)
project(SwiGluOptimizedExperiment LANGUAGES CXX)
include(GNUInstallDirs)

set(SOC_VERSION "" CACHE STRING "Ascend SoC version, such as ascend910b4 or ascend310p3")
if(NOT SOC_VERSION)
    message(FATAL_ERROR "SOC_VERSION not set. Use scripts/build.sh or pass -DSOC_VERSION=ascend910b4")
endif()

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE STRING "CANN root")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug/Release" FORCE)
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/swiglu_optimized_kernel.cpp
)

ascendc_library(swiglu_kernels_npu SHARED ${KERNEL_FILES})
ascendc_include_directories(swiglu_kernels_npu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

set(_HCC_CXX_ROOT "${ASCEND_CANN_PACKAGE_PATH}/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0")
if(EXISTS "${_HCC_CXX_ROOT}")
    ascendc_include_directories(swiglu_kernels_npu PRIVATE
        "${_HCC_CXX_ROOT}"
        "${_HCC_CXX_ROOT}/aarch64-target-linux-gnu"
    )
endif()

add_executable(swiglu_optimized_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/verify_kernel_launch/swiglu_optimized_standalone.cpp
)

target_compile_options(swiglu_optimized_standalone PRIVATE
    -O2
    -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=0
    -Wall
    -Wextra
)

target_include_directories(swiglu_optimized_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_BINARY_DIR}/include/swiglu_kernels_npu
    ${CMAKE_INSTALL_PREFIX}/include
)

target_link_libraries(swiglu_optimized_standalone PRIVATE
    ascendcl
    swiglu_kernels_npu
    dl
)

execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0], end='')"
    OUTPUT_VARIABLE TORCH_PATH
    RESULT_VARIABLE TORCH_FIND_RET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python3 -c "import torch_npu, pathlib; print(pathlib.Path(torch_npu.__file__).resolve().parent, end='')"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    RESULT_VARIABLE TORCH_NPU_FIND_RET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['include'], end='')"
    OUTPUT_VARIABLE PYTHON_INCLUDE
    RESULT_VARIABLE PYTHON_FIND_RET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_FIND_RET EQUAL 0 AND TORCH_NPU_FIND_RET EQUAL 0 AND EXISTS "${TORCH_PATH}" AND EXISTS "${TORCH_NPU_PATH}")
    set(TORCH_INCLUDE "${TORCH_PATH}/include")
    set(TORCH_API_INCLUDE "${TORCH_INCLUDE}/torch/csrc/api/include")
    set(TORCH_LIB_DIR "${TORCH_PATH}/lib")

    set(TORCH_NPU_INCLUDE "${TORCH_NPU_PATH}/include")
    set(TORCH_NPU_LIB_DIR "${TORCH_NPU_PATH}/lib")

    set(TORCH_ASC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/torch_extension/swiglu_torch_register.asc)
    set_source_files_properties(${TORCH_ASC_SRC} PROPERTIES LANGUAGE CXX)

    add_library(swiglu_torch_register SHARED
        ${TORCH_ASC_SRC}
    )

    target_compile_options(swiglu_torch_register PRIVATE
        -O2
        -std=c++17
        -D_GLIBCXX_USE_CXX11_ABI=1
        -x
        c++
    )

    target_include_directories(swiglu_torch_register PRIVATE
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/include/swiglu_kernels_npu
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TORCH_INCLUDE}
        ${TORCH_API_INCLUDE}
        ${TORCH_NPU_INCLUDE}
        ${TORCH_NPU_PATH}
        ${PYTHON_INCLUDE}
    )

    target_link_directories(swiglu_torch_register PRIVATE
        ${TORCH_LIB_DIR}
        ${TORCH_NPU_LIB_DIR}
    )

    target_link_libraries(swiglu_torch_register PRIVATE
        torch_cpu
        torch
        c10
        torch_npu
        ascendcl
        swiglu_kernels_npu
        dl
    )

    install(TARGETS swiglu_torch_register DESTINATION lib)

    find_program(PATCHELF_EXE patchelf)
    if(PATCHELF_EXE)
        install(CODE "execute_process(COMMAND ${PATCHELF_EXE} --set-rpath \"${TORCH_LIB_DIR}:${TORCH_NPU_LIB_DIR}:${CMAKE_INSTALL_PREFIX}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64\" \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/libswiglu_torch_register.so)")
    endif()
else()
    message(WARNING "PyTorch or torch_npu not found. Torch extension will not be built.")
endif()

install(TARGETS swiglu_kernels_npu DESTINATION lib)
install(TARGETS swiglu_optimized_standalone RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
