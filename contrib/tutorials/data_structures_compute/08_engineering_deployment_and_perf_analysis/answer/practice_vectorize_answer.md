# 实践题参考答案：注意力算子向量化改造

> 题目见 08.03「综合编程实践题」。改造目标：将阶段 B（o = P × V）的
> 内层 k 循环从"逐元素 GM 标量读"改为"DataCopy 批量读 + 向量指令累加"，
> 使 msProf 的 `aiv_vec_ratio` 从 ≈0 显著上升。

## 1. 改造思路

原始（纯标量）阶段 B：

```cpp
for (uint32_t j = 0; j < dim; ++j) {          // 输出维度方向
    float s = 0.0f;
    for (uint32_t k = 0; k < seqLen; ++k) {   // 内层 k 循环：逐元素 GM 标量读 ×2
        s += scoresRow[k] * static_cast<float>(vGlobal.GetValue(k * dim + j));
    }
    oRow[j] = s;
}
```

向量化改造要点：

1. **交换循环顺序**：外层遍历 k，内层一次性处理整个 dim 向量（64 个元素）；
2. **V 全量预载 UB**：`DataCopy(vUb, vGlobal, seqLen * dim)` 一次搬入（64KB），
   k 循环内只做 UB→UB 的行拷贝（`vUb[k * dim]` 切片），消除 GM 标量读；
3. **标量 × 向量**：P[k] 是标量（UB 普通数组 `scoresRow[k]`），
   用 `Muls(vRowF, vRowF, pik, dim)` 广播乘，`Add(oRowF, oRowF, vRowF, dim)` 累加；
4. **half→float**：V 行是 half，先 `Cast` 到 float 再乘加。

## 2. 核心代码

```cpp
// 新增 UB 缓冲（Init 中初始化）：
//   AscendC::TBuf<AscendC::TPosition::VECCALC> vUbBuf;      // V 全量 [S, D] half
//   AscendC::TBuf<AscendC::TPosition::VECCALC> vRowFBuf;    // V 一行 [D] float
//   pipe.InitBuffer(vUbBuf, seqLen * dim * sizeof(half));
//   pipe.InitBuffer(vRowFBuf, dim * sizeof(float));

// 阶段 B（每行）：
// 先批量读 V 全量到 UB（每批一次，或整核一次）
AscendC::DataCopy(vUb, vGlobal, seqLen * dim);      // MTE2
AscendC::SyncAll();

for (uint32_t k = 0; k < seqLen; ++k) {
    AscendC::DataCopy(vRowH, vUb[k * dim], dim);    // UB→UB 行拷贝
    AscendC::Cast(vRowF, vRowH, AscendC::RoundMode::CAST_NONE, dim);
    float pik = scoresRow[k];                        // P 标量（UB 数组）
    AscendC::Muls(vRowF, vRowF, pik, dim);           // 广播乘
    AscendC::Add(oRowF, oRowF, vRowF, dim);          // 向量累加
}
AscendC::SyncAll();
AscendC::Cast(oRowH, oRowF, AscendC::RoundMode::CAST_NONE, dim);
AscendC::SyncAll();
AscendC::DataCopy(oGlobal[row * dim], oRowH, dim);  // MTE3 写回
```

## 3. 验证步骤

```bash
# 1) 修改 kernel 后重新编译安装
cd src/attention_op
source $ASCEND_HOME_PATH/set_env.sh
bash scripts/build_ops.sh

# 2) 精度回归（必须 PASS）
source scripts/env_custom_opp.sh
aclnn_runner/build/main_attention_benchmark data 512 64

# 3) 性能对比（看 aiv_vec_ratio 变化）
bash scripts/run_profiling.sh 512 --output prof
python3 -c "
import glob, csv
p = glob.glob('prof/prof_512/OPPROF_*')[-1]
r = next(csv.DictReader(open(f'{p}/PipeUtilization.csv')))
print('aiv_vec_ratio =', r['aiv_vec_ratio'])   # 改造前 ≈ 0
"
```

## 4. 环境注意事项（重要）

- 本实验环境的 CANN 工具链对「**循环内标量读取 + 向量指令混用**」存在编译器缺陷
  （08.02 中纯标量实现正是为此）。若改造后出现输出异常（如行结果呈均匀分布），
  请检查是否命中该组合；可尝试：
  1. 将 `pik` 改为从 GM 标量读（`vGlobal`/`qGlobal.GetValue`，避开 UB 标量读）；
  2. 或拆分循环（先取标量数组，再进纯向量循环）；
  3. 或将 k 循环改为编译期常量边界 / 模板展开。
- 若最终受环境限制无法向量化到 PASS，**请记录实验现象与 msProf 指标差异**，
  并在总结中说明"向量化思路 + 环境限制结论"，同样视为完成实践题。

## 5. 预期结果

| 指标 | 纯标量（08.02） | 向量化后（预期） |
| -- | -- | -- |
| aiv_vec_ratio | ≈ 0 | 明显 > 0（向量指令占比上升） |
| Task Duration | ~85 ms（512） | 下降（GM 标量读消失） |
| 精度 | PASS（maxAbsErr ≈ 6e-5） | PASS（误差可能略增，仍在阈值内） |
