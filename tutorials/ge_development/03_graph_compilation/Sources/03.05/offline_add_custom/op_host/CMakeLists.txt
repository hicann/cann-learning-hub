file(GLOB host_ops_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

npu_op_code_gen(
  SRC ${host_ops_srcs}
  PACKAGE ${package_name}
  OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

npu_op_library(offline_add_optiling TILING ${host_ops_srcs})

file(GLOB generated_proto_src CONFIGURE_DEPENDS "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
set_source_files_properties(${generated_proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(offline_add_opproto GRAPH ${host_ops_srcs} ${generated_proto_src})

npu_op_package_add(${package_name}
  LIBRARY offline_add_optiling offline_add_opproto
)
