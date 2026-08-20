# 01.02 课后实践参考说明

`environment_check.py` 使用 `shutil.which` 在当前 `PATH` 中定位工具。工具存在时把结果转换为绝对路径，不存在时返回 `MISSING`。

参考实现不写死 CANNLab 中的安装路径，因此更换实例或镜像后仍可复用。工具被定位到只表示命令入口存在；ASC 是否正确加载、NPU 是否可用，仍应结合 CMake 配置结果和 `npu-smi info` 判断。
