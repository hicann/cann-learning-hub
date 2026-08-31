# ReduceLab 章节实践答案

## 实践 1：补全 ReduceSum 数据搬运逻辑

### 任务要求

补全 ReduceSum 算子中的输入数据读取、局部规约和结果写回逻辑。

### 参考思路

1. 每个核通过 `GetBlockIdx()` 确定自己负责的数据范围 `[start, end)`
2. 用 `GetValue()` 从 GM 逐元素读取输入数据
3. 转换为 float 进行累加（避免 FP16 精度损失）
4. 用 `SetValue()` 将局部结果写入输出 tensor `y[blockId]`

### 关键代码位置

参考文件：
```bash
cat answer/reduce_sum_todo_answer.cpp
```

---

## 实践 2：实现 ReduceMax

### 任务要求

在 ReduceSum 的基础上，将规约逻辑修改为 ReduceMax。

### 参考思路

1. 初始化局部最大值为 `-1e30f`（近似 -∞）
2. 遍历输入数据，用 `if (v > localMax)` 比较
3. 保留最大值
4. 将最大值写回输出 `y[blockId]`

### 关键代码位置

参考文件：
```bash
cat answer/reduce_max_answer.cpp
```

---

## 实践 3：验证实验结果

### 任务要求

运行 benchmark，确认 3 个算子全部 PASS。

### 验证方法

运行后对比：
```bash
cat answer/expected_output.txt
```

如果 NPU 输出与预期输出一致（3/3 PASS），则说明实验通过。

---

## 课后测试答案

### 选择题
1. D（910B 有 20+个 AI Core）
2. C（0..19，910B 有 20+核）
3. B（避免 FP16 精度损失）
4. B（D-cache/UB 硬件错误 507015）

### 填空题
5. 一致
6. int64_t
7. 小根堆

### 简答题
8. GM → DataCopy → UB → 计算 → DataCopy → GM
9. 部分芯片 D-cache 对 workspace GM 区域写入有限制，输出 tensor 由 CANN 框架管理
10. custom_ops/json + scripts/build_ops.sh + aclnn_runner/main_xxx_benchmark.cpp
