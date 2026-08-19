cmake_minimum_required(VERSION 3.16)
project(RopeBaselineExperiment LANGUAGES CXX)
include(GNUInstallDirs)

set(SOC_VERSION "" CACHE STRING "Ascend SoC version (e.g. Ascend910B4, Ascend310P3)")
if(NOT SOC_VERSION)
    message(FATAL_ERROR
        "SOC_VERSION not set. Pass -DSOC_VERSION=ascend310p3 "
        "or run via scripts/build.sh (auto-detects from npu-smi).")
endif()

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "CANN root")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug/Release" FORCE)
endif()

# ── ascendc_kernel_cmake 发现 ──────────────────────────────────
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

# ── 内核编译 (device 端, ascendc_library 管理) ─────────────────
set(KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/rope_baseline_kernel.cpp)
ascendc_library(rope_kernels_npu SHARED ${KERNEL_FILES})

# CANN 8.x HCC 头注入
set(_HCC_CXX_ROOT "${ASCEND_CANN_PACKAGE_PATH}/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0")
if(EXISTS "${_HCC_CXX_ROOT}")
    ascendc_include_directories(rope_kernels_npu PRIVATE
        "${_HCC_CXX_ROOT}"
        "${_HCC_CXX_ROOT}/aarch64-target-linux-gnu")
endif()

ascendc_include_directories(rope_kernels_npu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)

# ── 独立验证可执行文件 (host 端, 纯 C++ 无 ASC 依赖) ──────────
add_executable(rope_baseline_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/verify_kernel_launch/rope_baseline_standalone.cpp)
target_compile_options(rope_baseline_standalone PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra)
target_include_directories(rope_baseline_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_INSTALL_PREFIX}/include)
target_link_libraries(rope_baseline_standalone PRIVATE
    ascendcl rope_kernels_npu dl)

# ═══════════════════════════════════════════════════════════════
#  PyTorch TORCH_LIBRARY 算子注册 (.so)
# ═══════════════════════════════════════════════════════════════
# 定位 libtorch (pip-installed PyTorch)
execute_process(
    COMMAND python3 -c
        "import torch; print(torch.utils.cmake_prefix_path, end='')"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    RESULT_VARIABLE _torch_find_ret
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_torch_find_ret EQUAL 0 AND EXISTS "${TORCH_CMAKE_PREFIX}")
    # 手动定位 libtorch (不用 find_package(Torch) — 它绑定了 CUDA 依赖)
    execute_process(
        COMMAND python3 -c
            "import torch; print(torch.__path__[0], end='')"
        OUTPUT_VARIABLE TORCH_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(TORCH_INCLUDE "${TORCH_PATH}/include")
    set(TORCH_LIB_DIR "${TORCH_PATH}/lib")

    # .asc → C++: 强制 g++ 按 C++ 编译 (扩展名非标准)
    set(_ASC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/torch_extension/rope_basline_torch.asc)
    set_source_files_properties(${_ASC_SRC} PROPERTIES LANGUAGE CXX)

    add_library(rope_torch_register SHARED ${_ASC_SRC})

    target_compile_options(rope_torch_register PRIVATE
        -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1
        -x c++)    # 强制 .asc 按 C++ 编译

    target_include_directories(rope_torch_register PRIVATE
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TORCH_INCLUDE})

    target_link_directories(rope_torch_register PRIVATE ${TORCH_LIB_DIR})
    target_link_libraries(rope_torch_register PRIVATE
        torch_cpu torch c10
        ascendcl
        rope_kernels_npu)

    # RPATH: ascendc.cmake 内部 CMAKE_SKIP_RPATH=TRUE 禁了 CMake RPATH。
    #          绕过它, 用 patchelf 在 install 后直接注入 RUNPATH。
    find_program(PATCHELF_EXE patchelf)
    if(PATCHELF_EXE)
        install(CODE "execute_process(COMMAND ${PATCHELF_EXE}
            --set-rpath \"${TORCH_LIB_DIR}:${CMAKE_INSTALL_PREFIX}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64\"
            \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/librope_torch_register.so)")
    else()
        message(WARNING "patchelf not found; RPATH won't be set. Install it: apt install patchelf")
    endif()

    install(TARGETS rope_torch_register DESTINATION lib)
    message(STATUS "Torch extension: ENABLED (librope_torch_register.so)")
else()
    message(WARNING "Torch extension: DISABLED (PyTorch not found)")
endif()

# ── install ────────────────────────────────────────────────────
install(TARGETS rope_kernels_npu DESTINATION lib)
install(TARGETS rope_baseline_standalone RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
