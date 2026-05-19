file(GLOB host_src ./*.cpp)

# host侧源码
npu_op_code_gen(
    SRC ${host_src}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

# tiling动态库
npu_op_library(cust_optiling TILING ${host_src})

# aclnn单算子调用的动态库
file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_src}
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
)