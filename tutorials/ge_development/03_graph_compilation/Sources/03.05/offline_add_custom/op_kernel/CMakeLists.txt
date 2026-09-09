npu_op_kernel_sources(offline_add_kernels
  OP_TYPE AddCustom
  KERNEL_FILE add_custom.cpp
  KERNEL_DIR ./
)
npu_op_kernel_library(offline_add_kernels
  SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
  TILING_LIBRARY offline_add_optiling
)
npu_op_package_add(${package_name}
  LIBRARY offline_add_kernels
)
