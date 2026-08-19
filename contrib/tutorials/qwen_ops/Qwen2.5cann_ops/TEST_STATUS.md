# 验证状态（2026-07-15）

已完成的非设备验证：

- `bash scripts/verify_layout.sh`：通过；十个项目的源码、测试、构建脚本和 `out/` 产物均存在。
- `bash scripts/run_all_existing_tests.sh`：通过测试发现；列出全部原有 `tests/*.py` 入口（排除仅供导入的 `_setup_env.py`），包括 Qwen 原生对比脚本。
- 两个新集成入口的 `--help`：通过。
- 两个新集成脚本：`py_compile` 通过。

设备验证更新：在可访问宿主 NPU 的历史回归中，十个项目的 `tests/test_torch_op.py` 均已通过；GEMM/SwiGLU 的旧 PyTorch ABI 注册库已重建。本次宿主机的 8 张 Ascend 910B4 均可用，五基础版与五优化版统一接入均以 `--repeat 1` 返回 `0` 并生成 `results/latest.json`：两者 logits allclose 均通过。基础/优化自定义前向分别为 `50870.109/2657.420 ms`（同模型、5 tokens、单次样本），优化版为 `19.142665×` 更快；该数据含 bridge 且样本量为 1，不能作为稳定性能结论。完整结果及 JSON 数值见 `实验手册/2026-7-15全项目测试记录.md`。

在实际 NPU 主机的复现顺序：

```bash
cd /home/user/Qwen2.5cann_ops
export ASCEND_HOME_PATH=/home/user/Ascend/ascend-toolkit/cann-8.5.0
bash scripts/verify_layout.sh
bash scripts/run_all_existing_tests.sh --execute
Qwen2.5BaselineIntegrationExperiment/run.sh --repeat 10
Qwen2.5OptimizedIntegrationExperiment/run.sh --repeat 10
```

只有两个变体都以相同模型、prompt、repeat 在可用 NPU 上成功生成 `results/latest.json` 后，才能形成有效的基础/优化端到端性能结论。
