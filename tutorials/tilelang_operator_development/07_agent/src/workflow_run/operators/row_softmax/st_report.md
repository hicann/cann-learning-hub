# row_softmax ST 验收报告

结论：**PASS**。本轮正确性 ST 共收集 90 项，全部通过。

## 契约

输入为连续二维 Ascend NPU `float32` 张量，`M∈[4,256]` 且 `M%4=0`，`N∈[128,2048]` 且 `N%64=0`，所有元素有限且绝对值不超过 80。输出应当新分配、连续，并与输入保持相同的 shape、dtype 和 device；输入内容不能被修改。算子只交付前向计算。

数值参考使用 CPU PyTorch `float64` Softmax，再转为 `float32`。逐元素容差为 `1e-6 + 1e-4 × abs(reference)`，同时检查行和、有限性与非负性。

## 覆盖结论

| 维度 | 结论 | 证据重点 |
|---|---|---|
| 功能 | covered | 指定案例、扩展 shape、重复调用与平移不变性 |
| 精度 | covered | 独立 CPU 参考、逐元素误差、行和与有限性 |
| 边界 | covered | 最小/最大合法域、调度波次与非 2 次幂向量块数 |
| 梯度 | not applicable | 需求只交付前向 |
| 状态修改 | covered | 输入位模式不变，旧输出不被后续调用破坏 |
| 异常拒绝 | covered | 非法 dtype、shape、布局和 builder 参数 |
| 布局与接口 | covered | shape、dtype、device、连续性和新分配语义 |
| 后端与分支 | covered | 实际执行 AscendC、tvm_ffi 与 SimdVF 路径 |
| 随机性 | not applicable | 算子没有随机计算；测试种子固定 |
| 执行证据 | covered | nodeid、执行结果和源码对应关系完整 |

## 实际执行

| 检查 | 结果 | 证据 |
|---|---:|---|
| 九个指定精度案例 | 9 passed | [执行日志](evidence/generation/accuracy.log)、[结构化结果](evidence/generation/result.json) |
| builder 回归初测 | 发现 1 项失败 | [失败日志](evidence/st/builder_initial.log) |
| 修复后完整 Level 1 | 90 collected，90 passed | [执行记录](evidence/st/level1.json) |

## 发现与修复

测试发现：先用整数参数编译，再传入浮点参数时，Python 的等价缓存键可能让非法参数命中缓存，从而绕过类型检查。修复方式是先验证公开 `build` 的参数，再进入私有编译缓存。

这项修改修复的是接口行为，没有改变算子计算、tile、buffer 或同步方式。九个指定案例和完整 ST 均通过，说明修复没有破坏已有功能。

## 关键交付物

- [初始实现](evidence/generation/initial/row_softmax.py)
- [pytest](test_row_softmax.py)
- [初始设计](evidence/design/design_v1.md)
- [完整执行记录](evidence/st/level1.json)
