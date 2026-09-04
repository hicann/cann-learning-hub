# 样例来源

独立 ASC 样例基于 `cann/asc-devkit`：

- 仓库：<https://gitcode.com/cann/asc-devkit>
- 路径：`examples/01_simd_cpp_api/01_utilities/05_sanitizer/ms_sanitizer`
- 基线提交：`ac600d5f72c72cb46cae60cfacd3f26c3bcb2491`
- 许可证：[CANN Open Software License Agreement Version 2.0](LICENSE.asc-devkit)

课程在上游 Add 样例上增加了 baseline、memcheck、racecheck、initcheck、synccheck 五种模式。
`operator_project_sanitizer/` 的工程骨架来自仓库内标准 AddCustom 模板，并加入同样的四类故障和
`add_ops_compile_options` / `npu_op_kernel_options` 插桩方式。这些课程改动不是官方样例原样内容。
