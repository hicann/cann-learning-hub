# kernel源码
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

# kernel动态库
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# 添加到package中
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)