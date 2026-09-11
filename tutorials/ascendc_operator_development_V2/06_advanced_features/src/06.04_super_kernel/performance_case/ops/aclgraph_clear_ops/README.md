# aclgraph_clear_ops

ACLGraph 路径使用的 pybind 自定义算子包。

## 产物

- Python 包：`op_extension`
- 导入行为：`import op_extension` 会加载 `op_extension.custom_ops_lib`，并注册 `torch.ops.ascendc_ops.clear_ops`
- 编译方式：`bisheng -x asc` 直接把 `csrc/*.asc` 编译为 Python extension

## 安装

```bash
bash examples/custom_sk/ops/aclgraph_clear_ops/install.sh
```

可选环境变量：

- `PYTHON`：指定 Python 解释器，默认 `python3`
- `SK_NPU_ARCHS`：指定 Bisheng `--npu-arch`，该样例一次只编译一个架构，默认 `dav-2201`

## 卸载

```bash
bash examples/custom_sk/ops/aclgraph_clear_ops/uninstall.sh
```
