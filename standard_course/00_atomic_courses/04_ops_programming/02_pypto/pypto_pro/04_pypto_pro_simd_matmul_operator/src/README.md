# 第 4 章示例与审阅记录

本章以单 Tile → K 分块 → 多核/尾块串联 GEMM，另提供转置 B 的布局示例。正文直接展示完整 Kernel；[gemm_reference.py](./gemm_reference.py) 汇总同一组实现及独立 Host 校验入口，运行不依赖其他 Notebook 或 PyPTO/gym 源码仓路径。

在完成课程环境配置的 Ascend 950 上，从章节目录运行：

```bash
python src/gemm_reference.py
```

设备号读取 `TILE_FWK_DEVICE_ID`，未设置时为 0；运行前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。测试使用 1/4/8 核，目标设备需支持对应启动核数。此处未引入 CPU 仿真路径。

| 内容 | Notebook | 参考函数 | 设备校验用例 |
| --- | --- | --- | --- |
| 单 Tile | 04.02 | `mm_single` | 随机值、全 1，共 2 组 |
| K 分块 | 04.02 | `mm_k_split` | 随机值、K 三块常量，共 2 组 |
| 转置 B | 04.03 | `mm_transposed_b` | M=64,K=128,N=32，1 组 |
| 动态 M/N | 04.04 | `mm_dynamic` | 整除、M/N 尾块、空闲核和 1/4/8 核，共 6 组 |
| 独立实践 | 04.05 | 学生工作区与 `check_case` | 与参考实现共用相同验收逻辑 |

## 2026-11-02：第 4 章样板改写

已完成的教学补充：

- 导学简化标题与分节，使用两列导航、可检查的学习目标及起始入口。
- 单/双缓冲地址预算，完整 Kernel 与 Host 调用，流水依赖及常见错误推导。
- 4×4 矩阵、2×2 示意分形的 NZ/ZN 元素顺序；真实 40×56 有效输入的 48×64 存储预算；完整转置搬入实例。
- 500×300 输出的任务编号、核区间、任务 11 的输入输出范围与 5 个 Tile 的有效形状。
- 三道递进实践的独立工作区、校验器、验收标准及完整参考代码；未作答明确显示尚未完成。

源码依据（审阅时 PyPTO 为 `eb60d7b58de75f5edcfc1c78e563c210e840656e`，以下路径相对该源码仓）：

- `docs/zh/guide/programming_guide/pro/development/cube_computation.md`：数据流、K 累加、默认布局、转置、动态 M/N 与尾块。
- `docs/zh/guide/programming_guide/pro/development/tiling/tiling_parameter_definition.md`：TilingData 与 TilingKey 的约束。
- `docs/zh/api/pro_api/SIMD-API/memory_data_movement/load.md` 与 `cube_computation/matmul.md`：元素 offset、order、有效形状及累加语义。
- `python/pypto_pro/language/_api.py`、`python/pypto_pro/ir/op/block_ops.py`：API 参数、TileType、槽位容量与对齐规则。
- `python/tests/st/pypto_pro/frontend/matmul/test_matmul_api_basic.py`：单 Tile、非方阵转置、K 累加与尾块模式。
