# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file adapts the upstream ms_sanitizer sample under the CANN Open Software License Agreement Version 2.0.
# See LICENSE.asc-devkit and UPSTREAM.md in this directory.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture: dav-2201")
set(LAB05_FAULT_MODE "baseline" CACHE STRING "Fault mode: baseline, memcheck, racecheck, initcheck, synccheck")
option(LAB05_ENABLE_SANITIZER "Compile ASC code with --sanitizer instrumentation" ON)
set_property(CACHE LAB05_FAULT_MODE PROPERTY STRINGS baseline memcheck racecheck initcheck synccheck)
set(_lab05_fault_modes baseline memcheck racecheck initcheck synccheck)
list(FIND _lab05_fault_modes "${LAB05_FAULT_MODE}" _lab05_mode_index)
if(_lab05_mode_index EQUAL -1)
  message(FATAL_ERROR "LAB05_FAULT_MODE must be one of: ${_lab05_fault_modes}")
endif()

find_package(ASC REQUIRED)
project(lab05_debug_probe LANGUAGES ASC CXX)

add_executable(lab05_add_sanitizer add_sanitizer.asc)
target_compile_options(lab05_add_sanitizer PRIVATE
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
  -g
)
if(LAB05_ENABLE_SANITIZER)
  target_compile_options(lab05_add_sanitizer PRIVATE --sanitizer)
endif()
target_compile_definitions(lab05_add_sanitizer PRIVATE
  LAB05_FAULT_MODE_${LAB05_FAULT_MODE}=1
)
