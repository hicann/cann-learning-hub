本实验通过 ACLRT_LAUNCH_KERNEL 和 torch.library 调用自定义 kernel，不打包完整 OPP 工程。
op_host 目录仅保留说明，后续需要接入 aclnn/GE 图模式时再补充 host 侧 tiling 与算子原型注册。
